import base64
import copy
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None

try:
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    httpx = None

try:
    from trajdata import MapAPI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    MapAPI = None

from tbsim.policies.base import Policy
from tbsim.policies.common import Action
from tbsim.utils.geometry_utils import detect_collision, transform_points_tensor
from tbsim.utils.trajdata_utils import extend_ego_plan


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _wrap_float_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def _tensor_bool_mask(value, device: torch.device, expected_len: int) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        mask = value.to(device=device).bool().flatten()
    else:
        mask = torch.as_tensor(value, device=device).bool().flatten()
    if mask.numel() != expected_len:
        return None
    return mask


def _extract_json_object(text: str) -> Optional[Dict]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _make_openai_timeout(timeout_s: float) -> Any:
    timeout = max(1.0, float(timeout_s))
    short_timeout = min(15.0, timeout)
    if httpx is None:
        return timeout
    return httpx.Timeout(
        timeout,
        connect=short_timeout,
        write=short_timeout,
        pool=short_timeout,
    )


def _velocity_world_from_speed_yaw(speed_mps: float, yaw_world: float) -> List[float]:
    speed = float(speed_mps)
    yaw = float(yaw_world)
    return [
        float(speed * math.cos(yaw)),
        float(speed * math.sin(yaw)),
    ]


LOCAL_ACTION_TYPES = (
    "turn_left_to_target",
    "turn_right_to_target",
    "lane_change_to_target",
    "maneuver_to_target",
    "slow_down",
    "speed_up",
)

TARGET_DRIVEN_ACTION_TYPES = (
    "turn_left_to_target",
    "turn_right_to_target",
    "lane_change_to_target",
    "maneuver_to_target",
)

LONGITUDINAL_ACTION_TYPES = (
    "slow_down",
    "speed_up",
)


class QwenChatClient:
    def __init__(
        self,
        api_keys_path: str,
        model: str,
        base_url: str,
        temperature: float,
        reasoning_effort: Optional[str],
        enable_thinking: bool,
        thinking_budget: Optional[int],
        timeout_s: float,
        api_key_env: Optional[str] = None,
        api_key_json_field: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if OpenAI is None:
            raise ModuleNotFoundError("openai is not installed (pip install openai)")

        base_url_lower = str(base_url).strip().lower()
        if "openrouter.ai" in base_url_lower:
            provider_name = "openrouter"
        elif "deepseek.com" in base_url_lower:
            provider_name = "deepseek"
        else:
            provider_name = "dashscope"
        env_candidates: List[str] = []
        json_field_candidates: List[str] = []
        if api_key_env:
            env_candidates.append(str(api_key_env).strip())
        if api_key_json_field:
            json_field_candidates.append(str(api_key_json_field).strip())
        if provider_name == "openrouter":
            env_candidates.append("OPENROUTER_API_KEY")
            json_field_candidates.append("openrouter_api_key")
        elif provider_name == "deepseek":
            env_candidates.append("DEEPSEEK_API_KEY")
            json_field_candidates.append("deepseek_api_key")
        else:
            env_candidates.append("DASHSCOPE_API_KEY")
            json_field_candidates.append("dashscope_api_key")

        api_key = None
        for env_name in env_candidates:
            if not env_name:
                continue
            api_key = os.getenv(env_name)
            if api_key:
                break
        if not api_key and api_keys_path:
            with open(api_keys_path, "r", encoding="utf-8") as f:
                key_config = json.load(f)
            for field_name in json_field_candidates:
                if not field_name:
                    continue
                api_key = key_config.get(field_name)
                if api_key:
                    break
        if not api_key:
            raise ValueError(
                f"Missing {provider_name} API key. Checked env vars {env_candidates} and json fields {json_field_candidates}."
            )

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": _make_openai_timeout(timeout_s),
            "max_retries": 0,
        }
        if extra_headers:
            client_kwargs["default_headers"] = dict(extra_headers)
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.provider_name = provider_name
        self.enable_thinking = bool(enable_thinking)
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None and int(thinking_budget) > 0 else None
        self.timeout_s = float(timeout_s)
        self.reasoning_effort = (
            str(reasoning_effort).strip().lower() if reasoning_effort is not None else ""
        )
        self.last_reasoning_content = ""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        user_image_data_url: Optional[str] = None,
    ) -> str:
        user_content: Any = user_prompt
        if user_image_data_url:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": user_image_data_url}},
            ]
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if self.provider_name == "deepseek":
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if self.enable_thinking else "disabled"}
            }
            if self.enable_thinking and self.reasoning_effort in {"high", "max"}:
                request_kwargs["reasoning_effort"] = self.reasoning_effort
        elif self.enable_thinking and self.provider_name == "dashscope":
            extra_body = {"enable_thinking": True}
            if self.thinking_budget is not None:
                extra_body["thinking_budget"] = int(self.thinking_budget)
            request_kwargs["extra_body"] = extra_body
            request_kwargs["stream"] = True

        response = self.client.chat.completions.create(**request_kwargs)
        if request_kwargs.get("stream"):
            answer_chunks: List[str] = []
            reasoning_chunks: List[str] = []
            stream_start = time.time()
            for chunk in response:
                if time.time() - stream_start > self.timeout_s:
                    raise TimeoutError(f"LLM stream exceeded timeout_s={self.timeout_s:g}")
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content:
                    reasoning_chunks.append(str(reasoning_content))
                content_delta = getattr(delta, "content", None)
                if content_delta:
                    answer_chunks.append(str(content_delta))
            self.last_reasoning_content = "".join(reasoning_chunks)
            return "".join(answer_chunks)

        # Some OpenAI-compatible providers in our workflow reject or mishandle
        # reasoning-specific fields. Keep reasoning_effort in config for
        # compatibility, and only forward provider-specific fields above.
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(item.get("text", ""))
            return "\n".join([chunk for chunk in chunks if chunk])
        return str(content)


class AnchorDriveLLMAdversaryPolicy(Policy):
    """
    Hybrid policy for AnchorDrive Stage 1 in safe-sim.

    Background agents use a base policy trajectory.
    Adversarial agents are replaced by LLM-generated trajectories conditioned on ego_plan.
    """

    def __init__(self, device, eval_config, background_policy: Optional[Policy] = None):
        super().__init__(device)
        self.eval_config = eval_config
        self.llm_cfg = eval_config.llm
        self.background_policy = background_policy
        self.horizon = int(self.llm_cfg.trajectory_horizon)
        self.future_context_steps = int(self.llm_cfg.future_context_steps)
        self.max_nearby_agents = int(self.llm_cfg.max_nearby_agents)
        self.max_ref_points = int(self.llm_cfg.max_reference_points)
        self.max_retries = int(self.llm_cfg.max_retries)
        self.max_step_distance = float(self.llm_cfg.max_step_distance)
        self.route_num_points = int(getattr(self.llm_cfg, "route_num_points", 6))
        self.route_spacing_m = float(getattr(self.llm_cfg, "route_spacing_m", 5.0))
        self.send_route_information = bool(getattr(self.llm_cfg, "send_route_information", True))
        self.min_start_move_distance = float(getattr(self.llm_cfg, "min_start_move_distance", 0.08))
        self.min_heading_update_distance = float(getattr(self.llm_cfg, "min_heading_update_distance", 0.15))
        self.enable_macro_actions = bool(getattr(self.llm_cfg, "enable_macro_actions", True))
        self.default_lane_width_m = float(getattr(self.llm_cfg, "default_lane_width_m", 3.6))
        self.default_lane_change_duration_steps = int(getattr(self.llm_cfg, "default_lane_change_duration_steps", 12))
        self.default_macro_hold_steps = int(getattr(self.llm_cfg, "default_macro_hold_steps", 6))
        self.max_abs_accel_mps2 = float(getattr(self.llm_cfg, "max_abs_accel_mps2", 4.0))
        self.max_abs_yaw_rate_rps = float(getattr(self.llm_cfg, "max_abs_yaw_rate_rps", 0.8))
        self.max_speed_mps = float(getattr(self.llm_cfg, "max_speed_mps", 25.0))
        self.validation_max_abs_accel_mps2 = float(
            getattr(self.llm_cfg, "validation_max_abs_accel_mps2", 12.0)
        )
        self.validation_max_abs_yaw_rate_rps = float(
            getattr(self.llm_cfg, "validation_max_abs_yaw_rate_rps", 2.0)
        )
        self.validation_max_speed_mps = float(
            getattr(self.llm_cfg, "validation_max_speed_mps", 40.0)
        )
        self.validation_reject_dynamics_exceeded = bool(
            getattr(self.llm_cfg, "validation_reject_dynamics_exceeded", False)
        )
        self.validation_reject_waypoint_segment_jumps = bool(
            getattr(self.llm_cfg, "validation_reject_waypoint_segment_jumps", True)
        )
        self.llmscenario_disable_online_local_guard = bool(
            getattr(self.llm_cfg, "llmscenario_disable_online_local_guard", True)
        )
        self.llmscenario_enable_local_reality_assessor = bool(
            getattr(self.llm_cfg, "llmscenario_enable_local_reality_assessor", True)
        )
        self.llmscenario_local_reality_max_replans = max(
            0,
            int(getattr(self.llm_cfg, "llmscenario_local_reality_max_replans", 1)),
        )
        self.validation_max_waypoint_speed_jump_mps = float(
            getattr(self.llm_cfg, "validation_max_waypoint_speed_jump_mps", 6.0)
        )
        self.validation_max_waypoint_heading_jump_rad = float(
            getattr(self.llm_cfg, "validation_max_waypoint_heading_jump_rad", 0.8)
        )
        self.reachability_max_abs_accel_mps2 = float(
            self.llm_cfg.get("reachability_max_abs_accel_mps2", 5.0)
        )
        self.reachability_max_abs_yaw_rate_rps = float(
            self.llm_cfg.get("reachability_max_abs_yaw_rate_rps", 0.5)
        )
        self.reachability_distance_tolerance_m = float(
            self.llm_cfg.get("reachability_distance_tolerance_m", 0.75)
        )
        self.reachability_angle_tolerance_rad = float(
            self.llm_cfg.get("reachability_angle_tolerance_rad", 0.05)
        )
        self.nearby_radius_m = float(getattr(self.llm_cfg, "nearby_radius_m", 20.0))
        self.map_info_mode = str(getattr(self.llm_cfg, "map_info_mode", "local_radius")).strip().lower()
        self.map_lane_radius_m = float(getattr(self.llm_cfg, "map_lane_radius_m", self.nearby_radius_m))
        self.send_nearby_agents = bool(getattr(self.llm_cfg, "send_nearby_agents", True))
        self.send_nearby_future_trajectory = bool(
            getattr(self.llm_cfg, "send_nearby_future_trajectory", True)
        )
        self.enable_plan_assessor = bool(getattr(self.llm_cfg, "enable_plan_assessor", False))
        self.plan_assessor_max_replans = max(0, int(getattr(self.llm_cfg, "plan_assessor_max_replans", 2)))
        self.enable_local_collision_termination = bool(
            getattr(self.llm_cfg, "enable_local_collision_termination", True)
        )
        self.executed_steps_per_call = max(1, int(getattr(self.eval_config, "n_step_action", 5)))
        self.waypoint_horizon_steps = max(
            1,
            int(self.llm_cfg.get("waypoint_horizon_steps", self.executed_steps_per_call)),
        )
        self.waypoint_interval_steps = max(
            1,
            int(self.llm_cfg.get("waypoint_interval_steps", 5)),
        )
        configured_call_interval = getattr(self.llm_cfg, "call_interval_steps", None)
        if configured_call_interval is None:
            configured_call_interval = self.executed_steps_per_call
        self.call_interval_steps = max(1, int(configured_call_interval))
        configured_goal = str(getattr(self.llm_cfg, "scenario_goal", "")).strip()
        objective_goal = str(getattr(self.llm_cfg, "objective", "")).strip()
        self.scenario_goal = configured_goal or objective_goal or "adv collide with ego"
        self.manual_response_path = str(getattr(self.llm_cfg, "manual_response_path", "") or "").strip()
        self._manual_response_cache: Optional[Any] = None
        self.prompt_version = str(getattr(self.llm_cfg, "prompt_version", "v1") or "v1").strip().lower()
        self._llmscenario_prompt_mode = self._is_llmscenario_prompt_version(self.prompt_version)
        if self._llmscenario_prompt_mode:
            # LLMScenario uses post-generation evaluation/feedback rather than a
            # separate accept/reject plan assessor. Keep this baseline isolated
            # from AnchorDrive's assessor mechanism.
            self.enable_plan_assessor = False
            self.plan_assessor_max_replans = 0
            if self.llmscenario_disable_online_local_guard:
                self.validation_reject_waypoint_segment_jumps = False
                self.validation_reject_dynamics_exceeded = False
                self.enable_local_collision_termination = False
        self.send_bev_to_llm = bool(getattr(self.llm_cfg, "send_bev_to_llm", False))
        self.bev_max_side = int(getattr(self.llm_cfg, "bev_max_side", 512))
        self.bev_image_format = str(getattr(self.llm_cfg, "bev_image_format", "jpeg")).lower()
        self.bev_jpeg_quality = int(getattr(self.llm_cfg, "bev_jpeg_quality", 85))
        self.save_bev_debug = bool(getattr(self.llm_cfg, "save_bev_debug", False))
        self.results_dir = Path(self.llm_cfg.results_dir or eval_config.results_dir)
        bev_debug_dir_cfg = getattr(self.llm_cfg, "bev_debug_dir", None)
        self.bev_debug_dir = Path(bev_debug_dir_cfg) if bev_debug_dir_cfg else self.results_dir / "llm_bev"
        self.log_path = self.results_dir / "llm_logs" / "anchordrive_stage1_llm.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_bev_debug:
            self.bev_debug_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[QwenChatClient] = None
        self._map_api = None
        self._vector_maps: Dict[str, Any] = {}
        self._all_lane_prompt_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._prompt_sample_written = False
        self._assessor_prompt_sample_written = False
        self._last_llm_output_by_agent: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._last_plan_cache_by_agent: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._last_failure_feedback_by_agent: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def eval(self):
        if self.background_policy is not None:
            self.background_policy.eval()
        return self

    def _get_client(self) -> QwenChatClient:
        if self._client is None:
            extra_headers: Dict[str, str] = {}
            http_referer = str(self.llm_cfg["http_referer"] if "http_referer" in self.llm_cfg else "" or "").strip()
            app_title = str(self.llm_cfg["app_title"] if "app_title" in self.llm_cfg else "" or "").strip()
            reasoning_effort = self.llm_cfg["reasoning_effort"] if "reasoning_effort" in self.llm_cfg else None
            api_key_env = self.llm_cfg["api_key_env"] if "api_key_env" in self.llm_cfg else None
            api_key_json_field = self.llm_cfg["api_key_json_field"] if "api_key_json_field" in self.llm_cfg else None
            if http_referer:
                extra_headers["HTTP-Referer"] = http_referer
            if app_title:
                extra_headers["X-OpenRouter-Title"] = app_title
            self._client = QwenChatClient(
                api_keys_path=self.llm_cfg.api_keys_path,
                model=self.llm_cfg.model,
                base_url=self.llm_cfg.base_url,
                temperature=float(self.llm_cfg.temperature),
                reasoning_effort=reasoning_effort,
                enable_thinking=bool(getattr(self.llm_cfg, "enable_thinking", False)),
                thinking_budget=getattr(self.llm_cfg, "thinking_budget", None),
                timeout_s=float(self.llm_cfg.timeout_s),
                api_key_env=api_key_env,
                api_key_json_field=api_key_json_field,
                extra_headers=extra_headers or None,
            )
        return self._client

    def _load_manual_response_text(self, step_index: int) -> Optional[str]:
        if not self.manual_response_path:
            return None
        if self._manual_response_cache is None:
            path = Path(self.manual_response_path).expanduser()
            raw_text = path.read_text(encoding="utf-8")
            try:
                self._manual_response_cache = json.loads(raw_text)
            except json.JSONDecodeError:
                self._manual_response_cache = raw_text
        payload = self._manual_response_cache
        if isinstance(payload, dict):
            responses_by_step = payload.get("responses_by_step")
            if isinstance(responses_by_step, dict):
                selected = responses_by_step.get(str(int(step_index)))
                if selected is None:
                    selected = responses_by_step.get("default")
                payload = selected
            elif isinstance(payload.get("response_text"), str):
                return str(payload["response_text"])
        if payload is None:
            return None
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)

    def _append_log(self, payload: Dict) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _write_prompt_exchange(
        self,
        step_index: int,
        system_prompt: str,
        user_prompt: str,
        response_text: str = "",
    ) -> None:
        step_index = int(step_index)
        prompt_path = self.log_path.parent / f"prompt_sample_step{step_index}.txt"
        response_path = self.log_path.parent / f"response_sample_step{step_index}.txt"
        prompt_text = (
            "=== system_prompt ===\n"
            f"{system_prompt}\n\n"
            "=== user_prompt ===\n"
            f"{user_prompt}\n"
        )
        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            if str(response_text):
                with open(response_path, "w", encoding="utf-8") as f:
                    f.write(str(response_text))
        except OSError:
            pass

    def _maybe_write_prompt_sample(
        self,
        step_index: int,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        if int(step_index) != 0 or self._prompt_sample_written:
            return
        try:
            self._write_prompt_exchange(
                step_index=step_index,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text="",
            )
            self._prompt_sample_written = True
        except OSError:
            pass

    def _maybe_write_assessor_sample(
        self,
        step_index: int,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
    ) -> None:
        if int(step_index) != 0 or self._assessor_prompt_sample_written:
            return
        prompt_path = self.log_path.parent / "assessor_prompt_sample_step0.txt"
        response_path = self.log_path.parent / "assessor_response_sample_step0.txt"
        prompt_text = (
            "=== system_prompt ===\n"
            f"{system_prompt}\n\n"
            "=== user_prompt ===\n"
            f"{user_prompt}\n"
        )
        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            with open(response_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            self._assessor_prompt_sample_written = True
        except OSError:
            pass

    def _bev_tensor_to_rgb_uint8(self, bev_tensor: torch.Tensor) -> np.ndarray:
        bev = bev_tensor.detach().to(device="cpu").float()
        if bev.ndim == 4:
            bev = bev[0]
        if bev.ndim == 2:
            bev = bev.unsqueeze(0)
        elif bev.ndim == 3:
            if bev.shape[0] <= 8 and bev.shape[1] > 8 and bev.shape[2] > 8:
                pass
            elif bev.shape[2] <= 8 and bev.shape[0] > 8 and bev.shape[1] > 8:
                bev = bev.permute(2, 0, 1)
            else:
                if bev.shape[0] <= bev.shape[-1]:
                    pass
                else:
                    bev = bev.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported BEV ndim={bev.ndim}")

        if bev.shape[0] == 1:
            bev = bev.repeat(3, 1, 1)
        elif bev.shape[0] == 2:
            bev = torch.cat([bev, bev[-1:]], dim=0)
        else:
            bev = bev[:3]

        bev = bev.permute(1, 2, 0).contiguous()
        finite_mask = torch.isfinite(bev)
        if not bool(finite_mask.any()):
            raise ValueError("BEV contains no finite values")
        bev = torch.where(finite_mask, bev, torch.zeros_like(bev))

        min_v = float(bev.min().item())
        max_v = float(bev.max().item())
        if 0.0 <= min_v and max_v <= 1.0:
            bev = bev * 255.0
        elif -1.0 <= min_v and max_v <= 1.0:
            bev = (bev + 1.0) * 127.5
        elif 0.0 <= min_v and max_v <= 255.0:
            pass
        else:
            scale = max(max_v - min_v, 1e-6)
            bev = (bev - min_v) * (255.0 / scale)

        bev = torch.clamp(bev, 0.0, 255.0).to(dtype=torch.uint8)
        return bev.numpy()

    def _build_bev_data_url(
        self,
        obs: Dict,
        ctrl_idx: int,
        step_index: int,
        scene_index: int,
        ego_idx: Optional[int] = None,
        overlay_mode: str = "planner",
        rollout_summary: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        bev_meta: Dict[str, Any] = {
            "send_bev_to_llm": bool(self.send_bev_to_llm),
            "save_bev_debug": bool(self.save_bev_debug),
            "bev_attached": False,
            "bev_status": "disabled",
            "bev_original_shape": None,
            "bev_encoded_shape": None,
            "bev_image_format": None,
            "bev_image_bytes": 0,
            "bev_saved_path": None,
            "bev_save_status": "disabled",
            "bev_save_error": None,
            "bev_error": None,
            "bev_overlay_mode": str(overlay_mode),
            "bev_overlay_status": "not_attempted",
        }
        if not self.send_bev_to_llm:
            return None, bev_meta

        image_tensor = obs.get("image")
        if image_tensor is None:
            bev_meta["bev_status"] = "missing_obs_image"
            return None, bev_meta
        if not isinstance(image_tensor, torch.Tensor):
            bev_meta["bev_status"] = "obs_image_not_tensor"
            return None, bev_meta
        if ctrl_idx < 0 or ctrl_idx >= image_tensor.shape[0]:
            bev_meta["bev_status"] = "ctrl_idx_out_of_range"
            return None, bev_meta

        try:
            bev_np = self._bev_tensor_to_rgb_uint8(image_tensor[ctrl_idx])
            bev_meta["bev_original_shape"] = [int(x) for x in bev_np.shape]
            image = Image.fromarray(bev_np, mode="RGB")
            self._overlay_llm_bev_guidance(
                image=image,
                obs=obs,
                ctrl_idx=ctrl_idx,
                ego_idx=ego_idx,
                overlay_mode=overlay_mode,
                rollout_summary=rollout_summary,
                bev_meta=bev_meta,
            )

            if self.bev_max_side > 0:
                max_side = max(image.size)
                if max_side > self.bev_max_side:
                    scale = float(self.bev_max_side) / float(max_side)
                    new_size = (
                        max(1, int(round(image.size[0] * scale))),
                        max(1, int(round(image.size[1] * scale))),
                    )
                    image = image.resize(new_size, Image.BILINEAR)

            fmt = self.bev_image_format
            if fmt in {"jpg", "jpeg"}:
                pil_fmt = "JPEG"
                mime = "image/jpeg"
                save_kwargs = {
                    "quality": int(_clip(float(self.bev_jpeg_quality), 50.0, 100.0)),
                    "optimize": True,
                }
            else:
                pil_fmt = "PNG"
                mime = "image/png"
                save_kwargs = {}

            buf = io.BytesIO()
            image.save(buf, format=pil_fmt, **save_kwargs)
            payload = buf.getvalue()
            data_url = f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"

            bev_meta["bev_attached"] = True
            bev_meta["bev_status"] = "ok"
            bev_meta["bev_encoded_shape"] = [int(image.size[1]), int(image.size[0]), 3]
            bev_meta["bev_image_format"] = pil_fmt.lower()
            bev_meta["bev_image_bytes"] = int(len(payload))
            if self.save_bev_debug:
                ext = "jpg" if pil_fmt.lower() == "jpeg" else "png"
                file_name = f"scene{int(scene_index):04d}_step{int(step_index):04d}_ctrl{int(ctrl_idx):02d}.{ext}"
                save_path = self.bev_debug_dir / file_name
                try:
                    with open(save_path, "wb") as f:
                        f.write(payload)
                    bev_meta["bev_saved_path"] = str(save_path)
                    bev_meta["bev_save_status"] = "ok"
                except OSError as exc:
                    bev_meta["bev_save_status"] = "save_failed"
                    bev_meta["bev_save_error"] = f"{type(exc).__name__}: {exc}"
            return data_url, bev_meta
        except Exception as exc:  # pragma: no cover
            bev_meta["bev_status"] = "encode_failed"
            bev_meta["bev_error"] = f"{type(exc).__name__}: {exc}"
            return None, bev_meta

    def _get_ego_plan_world_tensor(
        self,
        obs: Dict,
        ego_idx: int,
        horizon: int,
    ) -> Optional[torch.Tensor]:
        ego_plan = obs.get("ego_plan")
        if ego_plan is None:
            return None
        ego_plan = ego_plan.to(self.device)
        if ego_plan.ndim == 2:
            ego_plan = ego_plan.unsqueeze(0)
        ego_indices = obs.get("ego_idx")
        if ego_indices is not None:
            if isinstance(ego_indices, torch.Tensor):
                ego_indices_list = ego_indices.detach().cpu().tolist()
            else:
                ego_indices_list = list(ego_indices)
            try:
                ego_plan = ego_plan[ego_indices_list.index(ego_idx) : ego_indices_list.index(ego_idx) + 1]
            except ValueError:
                ego_plan = ego_plan[:1]
        else:
            ego_plan = ego_plan[:1]

        target_horizon = max(1, int(horizon))
        dt = float(obs["dt"][ego_idx].item()) if "dt" in obs else 0.1
        if ego_plan.shape[1] < target_horizon:
            ego_plan = extend_ego_plan(ego_plan, target_horizon, dt=dt)
        ego_plan = ego_plan[0, :target_horizon]
        return self._to_world(ego_plan, obs["world_from_agent"][ego_idx].to(self.device))

    def _get_raster_from_world_matrix(
        self,
        obs: Dict,
        owner_idx: int,
    ) -> Optional[torch.Tensor]:
        raster_from_world = obs.get("raster_from_world")
        if isinstance(raster_from_world, torch.Tensor) and owner_idx < raster_from_world.shape[0]:
            return raster_from_world[owner_idx].to(self.device)
        raster_from_agent = obs.get("raster_from_agent")
        agent_from_world = obs.get("agent_from_world")
        if (
            isinstance(raster_from_agent, torch.Tensor)
            and isinstance(agent_from_world, torch.Tensor)
            and owner_idx < raster_from_agent.shape[0]
            and owner_idx < agent_from_world.shape[0]
        ):
            return torch.matmul(
                raster_from_agent[owner_idx].to(self.device),
                agent_from_world[owner_idx].to(self.device),
            )
        return None

    def _world_points_to_raster_np(
        self,
        points_world: torch.Tensor,
        raster_from_world: torch.Tensor,
    ) -> np.ndarray:
        if points_world.ndim == 1:
            points_world = points_world.unsqueeze(0)
        raster_points = transform_points_tensor(
            points_world.to(self.device, dtype=torch.float32),
            raster_from_world.to(self.device, dtype=torch.float32),
        )
        return raster_points.detach().cpu().numpy()

    @staticmethod
    def _draw_bev_polyline(
        draw: ImageDraw.ImageDraw,
        points_xy: np.ndarray,
        color: Tuple[int, int, int, int],
        width: int,
        image_size: Tuple[int, int],
    ) -> None:
        if points_xy.ndim != 2 or points_xy.shape[0] == 0 or points_xy.shape[1] < 2:
            return
        finite = np.isfinite(points_xy[:, 0]) & np.isfinite(points_xy[:, 1])
        points_xy = points_xy[finite]
        if points_xy.shape[0] == 0:
            return
        width_px, height_px = image_size
        margin = float(max(width_px, height_px) * 2)
        in_reasonable_range = (
            (points_xy[:, 0] >= -margin)
            & (points_xy[:, 0] <= width_px + margin)
            & (points_xy[:, 1] >= -margin)
            & (points_xy[:, 1] <= height_px + margin)
        )
        points_xy = points_xy[in_reasonable_range]
        if points_xy.shape[0] >= 2:
            draw.line(
                [tuple(map(float, point)) for point in points_xy],
                fill=color,
                width=width,
                joint="curve",
            )
        radius = max(2, width)
        for point in points_xy[:: max(1, points_xy.shape[0] // 8)]:
            x, y = float(point[0]), float(point[1])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    @staticmethod
    def _draw_bev_label(
        draw: ImageDraw.ImageDraw,
        xy: Tuple[float, float],
        text: str,
        color: Tuple[int, int, int, int],
    ) -> None:
        x, y = float(xy[0]), float(xy[1])
        draw.text((x + 5, y + 5), text, fill=(0, 0, 0, 220))
        draw.text((x + 4, y + 4), text, fill=color)

    def _overlay_llm_bev_guidance(
        self,
        image: Image.Image,
        obs: Dict,
        ctrl_idx: int,
        ego_idx: Optional[int],
        overlay_mode: str,
        rollout_summary: Optional[Dict[str, Any]],
        bev_meta: Dict[str, Any],
    ) -> None:
        if ego_idx is None:
            bev_meta["bev_overlay_status"] = "missing_ego_idx"
            return
        raster_from_world = self._get_raster_from_world_matrix(obs, ctrl_idx)
        if raster_from_world is None:
            bev_meta["bev_overlay_status"] = "missing_raster_transform"
            return
        try:
            draw = ImageDraw.Draw(image, "RGBA")
            line_width = max(3, int(round(max(image.size) / 160)))

            ego_plan_world = self._get_ego_plan_world_tensor(
                obs=obs,
                ego_idx=ego_idx,
                horizon=min(int(self.horizon), int(obs["target_positions"].shape[1])),
            )
            if ego_plan_world is not None:
                ego_plan_raster = self._world_points_to_raster_np(ego_plan_world, raster_from_world)
                self._draw_bev_polyline(
                    draw=draw,
                    points_xy=ego_plan_raster,
                    color=(255, 215, 0, 235),
                    width=line_width,
                    image_size=image.size,
                )
                if ego_plan_raster.shape[0] > 0:
                    self._draw_bev_label(draw, tuple(ego_plan_raster[0]), "EGO plan", (255, 215, 0, 255))

            actor_positions = {
                "ADV": (ctrl_idx, (255, 60, 60, 245)),
                "EGO": (ego_idx, (45, 160, 255, 245)),
            }
            for label, (agent_idx, color) in actor_positions.items():
                curr_raster = self._world_points_to_raster_np(
                    obs["centroid"][agent_idx : agent_idx + 1].to(self.device),
                    raster_from_world,
                )
                if curr_raster.shape[0] == 0 or not np.isfinite(curr_raster[0]).all():
                    continue
                x, y = float(curr_raster[0, 0]), float(curr_raster[0, 1])
                radius = max(5, line_width * 2)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
                self._draw_bev_label(draw, (x, y), label, color)

            if str(overlay_mode).lower() == "assessor" and isinstance(rollout_summary, dict):
                overlay_specs = [
                    ("adv_positions_world_tensor", "ADV rollout", (255, 80, 220, 235)),
                    ("ego_positions_world_tensor", "EGO rollout", (0, 235, 255, 235)),
                ]
                for tensor_key, label, color in overlay_specs:
                    points_world = rollout_summary.get(tensor_key)
                    if not isinstance(points_world, torch.Tensor):
                        continue
                    points_raster = self._world_points_to_raster_np(points_world, raster_from_world)
                    self._draw_bev_polyline(
                        draw=draw,
                        points_xy=points_raster,
                        color=color,
                        width=line_width,
                        image_size=image.size,
                    )
                    if points_raster.shape[0] > 0:
                        self._draw_bev_label(draw, tuple(points_raster[-1]), label, color)

            bev_meta["bev_overlay_status"] = "ok"
        except Exception as exc:  # pragma: no cover
            bev_meta["bev_overlay_status"] = "overlay_failed"
            bev_meta["bev_overlay_error"] = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _to_world(points_local: torch.Tensor, world_from_agent: torch.Tensor) -> torch.Tensor:
        return transform_points_tensor(points_local, world_from_agent)

    @staticmethod
    def _to_local(points_world: torch.Tensor, agent_from_world: torch.Tensor) -> torch.Tensor:
        return transform_points_tensor(points_world, agent_from_world)

    @staticmethod
    def _compute_local_yaws(
        points_world: torch.Tensor,
        current_yaw: torch.Tensor,
        min_heading_update_distance: float,
    ) -> torch.Tensor:
        if points_world.shape[0] == 1:
            world_heading = current_yaw.reshape(1)
        else:
            delta = points_world[1:] - points_world[:-1]
            raw_heading = torch.atan2(delta[:, 1], delta[:, 0])
            step_norm = torch.norm(delta, dim=-1)
            heading = torch.empty_like(raw_heading)
            last_heading = current_yaw.reshape(()).to(dtype=raw_heading.dtype)
            for i in range(raw_heading.shape[0]):
                if float(step_norm[i].item()) >= min_heading_update_distance:
                    last_heading = raw_heading[i]
                heading[i] = last_heading
            world_heading = torch.cat([heading, heading[-1:]], dim=0)
        return _wrap_to_pi(world_heading - current_yaw)

    def _trim_leading_stationary_points(
        self,
        world_points: torch.Tensor,
        current_world: torch.Tensor,
    ) -> torch.Tensor:
        if world_points.shape[0] == 0:
            return world_points
        offsets = torch.norm(world_points - current_world, dim=-1)
        moving_indices = torch.nonzero(offsets > self.min_start_move_distance, as_tuple=False).reshape(-1)
        if moving_indices.numel() == 0:
            return world_points
        first_moving_idx = int(moving_indices[0].item())
        if first_moving_idx <= 0:
            return world_points
        return world_points[first_moving_idx:]

    def _get_centerline_world_tensor(self, obs: Dict, agent_idx: int) -> Optional[torch.Tensor]:
        extras = obs.get("extras", {})
        centerline_xy = extras.get("centerline_xy")
        if centerline_xy is None:
            return None
        local_centerline = centerline_xy[agent_idx]
        if local_centerline.ndim != 2 or local_centerline.shape[-1] != 2:
            return None
        valid = torch.isfinite(local_centerline).all(dim=-1)
        local_centerline = local_centerline[valid]
        if local_centerline.shape[0] < 2:
            return None
        world_centerline = self._to_world(local_centerline, obs["world_from_agent"][agent_idx])
        return world_centerline

    def _sample_route_near_current_position(
        self,
        path_world: torch.Tensor,
        current_pos: torch.Tensor,
        num_points: Optional[int] = None,
        spacing_m: Optional[float] = None,
    ) -> torch.Tensor:
        if path_world.ndim != 2 or path_world.shape[-1] != 2 or path_world.shape[0] < 2:
            return path_world
        count = max(1, int(self.route_num_points if num_points is None else num_points))
        spacing = max(0.1, float(self.route_spacing_m if spacing_m is None else spacing_m))
        s_nodes, _, _ = self._centerline_geometry(path_world)
        closest_idx = int(torch.argmin(torch.norm(path_world - current_pos.unsqueeze(0), dim=-1)).item())
        start_s = s_nodes[closest_idx]
        query_s = start_s + torch.arange(
            count,
            device=path_world.device,
            dtype=path_world.dtype,
        ) * spacing
        sampled_world, _ = self._interp_path_by_s(path_world, s_nodes, query_s)
        return sampled_world

    @staticmethod
    def _centerline_geometry(path_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seg = path_xy[1:] - path_xy[:-1]
        seg_len = torch.norm(seg, dim=-1).clamp_min(1e-6)
        tangent = seg / seg_len.unsqueeze(-1)
        tangent = torch.cat([tangent, tangent[-1:]], dim=0)
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
        s = torch.zeros(path_xy.shape[0], dtype=path_xy.dtype, device=path_xy.device)
        s[1:] = torch.cumsum(seg_len, dim=0)
        return s, tangent, normal

    @staticmethod
    def _interp_path_by_s(path_xy: torch.Tensor, s_nodes: torch.Tensor, query_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_s = float(s_nodes[-1].item())
        q = torch.clamp(query_s, 0.0, max_s)
        idx_right = torch.searchsorted(s_nodes, q, right=False)
        idx_right = torch.clamp(idx_right, 1, s_nodes.shape[0] - 1)
        idx_left = idx_right - 1
        s0 = s_nodes[idx_left]
        s1 = s_nodes[idx_right]
        alpha = (q - s0) / (s1 - s0 + 1e-6)
        p0 = path_xy[idx_left]
        p1 = path_xy[idx_right]
        pos = p0 + (p1 - p0) * alpha.unsqueeze(-1)
        return pos, idx_left

    def _estimate_lateral_offset(
        self,
        current_pos: torch.Tensor,
        path_xy: torch.Tensor,
        normals: torch.Tensor,
    ) -> float:
        d = torch.norm(path_xy - current_pos.unsqueeze(0), dim=-1)
        idx = int(torch.argmin(d).item())
        vec = current_pos - path_xy[idx]
        return float(torch.dot(vec, normals[idx]).item())

    @staticmethod
    def _parse_json_maybe(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _get_agent_current_lane_id(self, obs: Dict, agent_idx: int) -> Optional[Any]:
        extras = obs.get("extras", {})
        ref_polyline_ids = extras.get("ref_polyline_ids")
        if ref_polyline_ids is None:
            return None
        if isinstance(ref_polyline_ids, torch.Tensor):
            if agent_idx < 0 or agent_idx >= ref_polyline_ids.shape[0]:
                return None
            raw_value = ref_polyline_ids[agent_idx]
        elif isinstance(ref_polyline_ids, (list, tuple)):
            if agent_idx < 0 or agent_idx >= len(ref_polyline_ids):
                return None
            raw_value = ref_polyline_ids[agent_idx]
        else:
            return None
        parsed = self._parse_json_maybe(raw_value)
        if isinstance(parsed, list) and parsed:
            return parsed[0]
        return None

    @staticmethod
    def _sample_polyline(polyline_xy: torch.Tensor, num_points: int) -> torch.Tensor:
        if polyline_xy.shape[0] <= num_points:
            return polyline_xy
        indices = torch.linspace(
            0,
            polyline_xy.shape[0] - 1,
            steps=num_points,
            dtype=torch.float32,
            device=polyline_xy.device,
        )
        indices = torch.round(indices).long()
        return polyline_xy[indices]

    @staticmethod
    def _polyline_yaws(polyline_xy: torch.Tensor) -> List[float]:
        if polyline_xy.ndim != 2 or polyline_xy.shape[-1] != 2 or polyline_xy.shape[0] < 2:
            return []
        delta = polyline_xy[1:] - polyline_xy[:-1]
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        yaw = torch.cat([yaw, yaw[-1:]], dim=0)
        return [float(x) for x in yaw.detach().cpu().tolist()]

    @staticmethod
    def _sample_numpy_polyline(polyline_xy: np.ndarray, num_points: int) -> np.ndarray:
        if polyline_xy.ndim != 2 or polyline_xy.shape[-1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        xy = np.asarray(polyline_xy[:, :2], dtype=np.float32)
        if xy.shape[0] <= num_points:
            return xy
        indices = np.linspace(0, xy.shape[0] - 1, num=num_points)
        indices = np.round(indices).astype(np.int64)
        return xy[indices]

    @staticmethod
    def _polyline_yaws_numpy(polyline_xy: np.ndarray) -> List[float]:
        if polyline_xy.ndim != 2 or polyline_xy.shape[-1] < 2 or polyline_xy.shape[0] < 2:
            return []
        delta = polyline_xy[1:, :2] - polyline_xy[:-1, :2]
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        yaw = np.concatenate([yaw, yaw[-1:]], axis=0)
        return [float(x) for x in yaw.tolist()]

    def _estimate_lane_width_from_map_lane(self, lane: Any) -> float:
        left_edge = getattr(lane, "left_edge", None)
        right_edge = getattr(lane, "right_edge", None)
        if left_edge is None or right_edge is None:
            return float(self.default_lane_width_m)
        try:
            left_xy = self._sample_numpy_polyline(left_edge.interpolate(num_pts=12).xy, num_points=12)
            right_xy = self._sample_numpy_polyline(right_edge.interpolate(num_pts=12).xy, num_points=12)
            n = min(left_xy.shape[0], right_xy.shape[0])
            if n < 2:
                return float(self.default_lane_width_m)
            width = np.linalg.norm(left_xy[:n] - right_xy[:n], axis=-1)
            width = width[np.isfinite(width)]
            if width.size == 0:
                return float(self.default_lane_width_m)
            return float(np.clip(width.mean(), 2.0, 8.0))
        except Exception:
            return float(self.default_lane_width_m)

    @staticmethod
    def _infer_turn_type_from_lane(lane: Any) -> str:
        center = getattr(lane, "center", None)
        if center is None:
            return "unknown"
        try:
            headings = np.asarray(center.h, dtype=np.float32)
            if headings.shape[0] < 2:
                return "unknown"
            delta = float((headings[-1] - headings[0] + np.pi) % (2 * np.pi) - np.pi)
            abs_delta = abs(delta)
            if abs_delta < 0.35:
                return "straight"
            if abs_delta > 2.6:
                return "u_turn"
            return "left" if delta > 0.0 else "right"
        except Exception:
            return "unknown"

    @staticmethod
    def _pick_single_lane_id(lane_ids: Any) -> Optional[Any]:
        if lane_ids is None:
            return None
        if isinstance(lane_ids, (set, list, tuple)):
            if len(lane_ids) == 0:
                return None
            return sorted(lane_ids)[0]
        return lane_ids

    def _get_vector_map(self, map_name: Any) -> Optional[Any]:
        if MapAPI is None or map_name is None:
            return None
        map_name = str(map_name)
        if not map_name:
            return None
        if map_name not in self._vector_maps:
            if self._map_api is None:
                cache_path = Path("~/.unified_data_cache").expanduser()
                self._map_api = MapAPI(cache_path, keep_in_memory=True)
            try:
                self._vector_maps[map_name] = self._map_api.get_map(map_name, scene_cache=None)
            except Exception:
                self._vector_maps[map_name] = None
        return self._vector_maps.get(map_name)

    def _serialize_map_lanes(self, lanes: List[Any]) -> List[Dict[str, Any]]:
        all_lanes: List[Dict[str, Any]] = []
        seen_lane_ids = set()
        for lane in lanes:
            lane_id = getattr(lane, "id", None)
            if lane_id in seen_lane_ids:
                continue
            seen_lane_ids.add(lane_id)
            center = getattr(lane, "center", None)
            if center is None:
                continue
            sampled_world = self._sample_numpy_polyline(center.xy, num_points=6)
            if sampled_world.shape[0] < 2:
                continue
            all_lanes.append(
                {
                    "lane_id": lane_id,
                    "centerline_world": sampled_world.tolist(),
                    "centerline_yaw": self._polyline_yaws_numpy(sampled_world),
                    "lane_width_m": self._estimate_lane_width_from_map_lane(lane),
                    "turn_type": self._infer_turn_type_from_lane(lane),
                    "neighbors": {
                        "left_lane_id": self._pick_single_lane_id(getattr(lane, "adj_lanes_left", None)),
                        "right_lane_id": self._pick_single_lane_id(getattr(lane, "adj_lanes_right", None)),
                    },
                }
            )
        all_lanes.sort(key=lambda lane_item: str(lane_item.get("lane_id")))
        return all_lanes

    def _get_local_map_lanes(
        self,
        obs: Dict,
        adv_idx: int,
        ego_idx: int,
    ) -> List[Dict[str, Any]]:
        map_names = obs.get("map_names")
        if map_names is None or adv_idx < 0 or ego_idx < 0 or adv_idx >= len(map_names) or ego_idx >= len(map_names):
            return self._get_candidate_lanes(obs, adv_idx)
        adv_map_name = str(map_names[adv_idx])
        ego_map_name = str(map_names[ego_idx])
        if adv_map_name != ego_map_name:
            return self._get_candidate_lanes(obs, adv_idx)
        map_name = adv_map_name

        vec_map = self._get_vector_map(map_name)
        if vec_map is None or getattr(vec_map, "lanes", None) is None:
            return self._get_candidate_lanes(obs, adv_idx)

        query_radius = float(self.map_lane_radius_m)
        adv_xy = obs["centroid"][adv_idx].detach().cpu().numpy()
        ego_xy = obs["centroid"][ego_idx].detach().cpu().numpy()
        adv_xyz = np.array([float(adv_xy[0]), float(adv_xy[1]), 0.0], dtype=np.float32)
        ego_xyz = np.array([float(ego_xy[0]), float(ego_xy[1]), 0.0], dtype=np.float32)

        lanes_near_adv = vec_map.get_lanes_within(adv_xyz, query_radius)
        lanes_near_ego = vec_map.get_lanes_within(ego_xyz, query_radius)
        combined_lanes = lanes_near_adv + lanes_near_ego
        if len(combined_lanes) == 0:
            return self._get_candidate_lanes(obs, adv_idx)
        return self._serialize_map_lanes(combined_lanes)

    def _get_current_lane_map_lanes(
        self,
        obs: Dict,
        adv_idx: int,
        ego_idx: int,
    ) -> List[Dict[str, Any]]:
        map_names = obs.get("map_names")
        if map_names is None or adv_idx < 0 or ego_idx < 0 or adv_idx >= len(map_names) or ego_idx >= len(map_names):
            return self._get_candidate_lanes(obs, adv_idx)
        adv_map_name = str(map_names[adv_idx])
        ego_map_name = str(map_names[ego_idx])
        if adv_map_name != ego_map_name:
            return self._get_candidate_lanes(obs, adv_idx)

        vec_map = self._get_vector_map(adv_map_name)
        if vec_map is None:
            return self._get_candidate_lanes(obs, adv_idx)

        lane_ids: List[str] = []
        adv_lane_id = self._get_agent_current_lane_id(obs, adv_idx)
        ego_lane_id = self._get_agent_current_lane_id(obs, ego_idx)
        for lane_id in [adv_lane_id, ego_lane_id]:
            if lane_id is None:
                continue
            lane_id = str(lane_id)
            try:
                lane_obj = vec_map.get_road_lane(lane_id)
            except Exception:
                continue
            lane_ids.append(lane_obj.id)

        if len(lane_ids) == 0:
            return self._get_candidate_lanes(obs, adv_idx)
        lane_objs = [vec_map.get_road_lane(lane_id) for lane_id in lane_ids]
        return self._serialize_map_lanes(lane_objs)

    def _get_map_information(
        self,
        obs: Dict,
        adv_idx: int,
        ego_idx: int,
    ) -> Dict[str, Any]:
        mode = self.map_info_mode
        if mode == "none":
            lanes: List[Dict[str, Any]] = []
        elif mode == "current_lanes":
            lanes = self._get_current_lane_map_lanes(obs, adv_idx=adv_idx, ego_idx=ego_idx)
        else:
            lanes = self._get_local_map_lanes(obs, adv_idx=adv_idx, ego_idx=ego_idx)
        return {
            "candidate_lanes": lanes,
            "mode": mode,
            "radius_m": float(self.map_lane_radius_m) if mode == "local_radius" else None,
        }

    def _map_info_description(self) -> str:
        if self.map_info_mode == "none":
            return "Do not display any lane information in map information."
        if self.map_info_mode == "current_lanes":
            return "In map information, only the lanes where the current EGO and ADV are respectively located are displayed."
        return (
            f'In map information, only display the current ADV radius {self.map_lane_radius_m:g}m and the current EGO radius {self.map_lane_radius_m:g}lanes appearing within m, and take the union of the two.'
        )

    def _get_candidate_lanes(self, obs: Dict, agent_idx: int) -> List[Dict[str, Any]]:
        extras = obs.get("extras", {})
        world_tf = obs["world_from_agent"][agent_idx]
        candidate_paths: List[Tuple[Optional[Any], torch.Tensor]] = []

        centerline_xy = extras.get("centerline_xy")
        if centerline_xy is not None and agent_idx < centerline_xy.shape[0]:
            center_local = centerline_xy[agent_idx]
            valid = torch.isfinite(center_local).all(dim=-1)
            center_local = center_local[valid]
            if center_local.ndim == 2 and center_local.shape[-1] == 2 and center_local.shape[0] >= 2:
                lane_id = self._get_agent_current_lane_id(obs, agent_idx)
                candidate_paths.append((lane_id, center_local))

        all_poss_refs = extras.get("all_poss_refs")
        all_ref_polyline_ids = extras.get("all_ref_polyline_ids")
        parsed_all_ref_ids = None
        if isinstance(all_ref_polyline_ids, torch.Tensor) and agent_idx < all_ref_polyline_ids.shape[0]:
            parsed_all_ref_ids = self._parse_json_maybe(all_ref_polyline_ids[agent_idx])
            if not isinstance(parsed_all_ref_ids, list):
                parsed_all_ref_ids = None
        elif isinstance(all_ref_polyline_ids, (list, tuple)) and agent_idx < len(all_ref_polyline_ids):
            parsed_all_ref_ids = self._parse_json_maybe(all_ref_polyline_ids[agent_idx])
            if not isinstance(parsed_all_ref_ids, list):
                parsed_all_ref_ids = None
        if (
            all_poss_refs is not None
            and hasattr(all_poss_refs, "ndim")
            and all_poss_refs.ndim >= 4
            and agent_idx < all_poss_refs.shape[0]
        ):
            refs_local = all_poss_refs[agent_idx]
            for ridx in range(refs_local.shape[0]):
                local_ref = refs_local[ridx]
                valid = torch.isfinite(local_ref).all(dim=-1)
                local_ref = local_ref[valid]
                if local_ref.ndim != 2 or local_ref.shape[-1] != 2 or local_ref.shape[0] < 2:
                    continue
                lane_id: Optional[Any] = None
                if isinstance(parsed_all_ref_ids, list) and ridx < len(parsed_all_ref_ids):
                    lane_path = parsed_all_ref_ids[ridx]
                    if isinstance(lane_path, list) and lane_path:
                        lane_id = lane_path[0]
                candidate_paths.append((lane_id, local_ref))

        candidate_lanes: List[Dict[str, Any]] = []
        seen = set()
        current_pos = obs["centroid"][agent_idx].to(world_tf.device, dtype=world_tf.dtype)
        for idx, (lane_id, local_path) in enumerate(candidate_paths):
            sampled_world_full = self._to_world(local_path, world_tf)
            sampled_world = self._sample_route_near_current_position(sampled_world_full, current_pos)
            if sampled_world.shape[0] < 2:
                continue
            key = tuple(torch.round(sampled_world[:2].reshape(-1) * 10).long().detach().cpu().tolist())
            if key in seen:
                continue
            seen.add(key)
            lane_id_value = lane_id if lane_id is not None else f"cand_{idx}"
            candidate_lanes.append(
                {
                    "lane_id": lane_id_value,
                    "centerline_world": sampled_world.detach().cpu().tolist(),
                    "centerline_yaw": self._polyline_yaws(sampled_world),
                    "lane_width_m": float(self.default_lane_width_m),
                    "turn_type": "unknown",
                    "neighbors": {
                        "left_lane_id": None,
                        "right_lane_id": None,
                    },
                }
            )
        return candidate_lanes

    def _rollout_world_points_from_controls(
        self,
        ctrl_idx: int,
        obs: Dict,
        ax_seq: torch.Tensor,
        ay_seq: torch.Tensor,
        target_horizon: int,
        initial_velocity_world: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        curr_pos = obs["centroid"][ctrl_idx].clone().to(self.device, dtype=torch.float32)
        if initial_velocity_world is None:
            curr_yaw = float(obs["yaw"][ctrl_idx].item())
            curr_speed = float(obs["curr_speed"][ctrl_idx].item())
            initial_velocity_world = _velocity_world_from_speed_yaw(curr_speed, curr_yaw)
        curr_vel = torch.tensor(initial_velocity_world, dtype=torch.float32, device=self.device)
        points = torch.zeros((target_horizon, 2), dtype=torch.float32, device=self.device)
        velocities = torch.zeros((target_horizon, 2), dtype=torch.float32, device=self.device)
        for i in range(target_horizon):
            curr_vel[0] = curr_vel[0] + float(ax_seq[i].item()) * dt
            curr_vel[1] = curr_vel[1] + float(ay_seq[i].item()) * dt
            curr_speed_norm = float(torch.linalg.norm(curr_vel).item())
            if curr_speed_norm > self.max_speed_mps:
                curr_vel = curr_vel * (self.max_speed_mps / max(curr_speed_norm, 1e-6))
            curr_pos = curr_pos + curr_vel * dt
            points[i] = curr_pos
            velocities[i] = curr_vel
        return points, velocities

    @staticmethod
    def _blend_world_yaws(primary_yaw: float, secondary_yaw: float, secondary_weight: float) -> float:
        secondary_weight = _clip(float(secondary_weight), 0.0, 1.0)
        primary_weight = 1.0 - secondary_weight
        x = primary_weight * math.cos(primary_yaw) + secondary_weight * math.cos(secondary_yaw)
        y = primary_weight * math.sin(primary_yaw) + secondary_weight * math.sin(secondary_yaw)
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            return _wrap_float_to_pi(primary_yaw)
        return _wrap_float_to_pi(math.atan2(y, x))

    def _compute_reachable_target_constraint(
        self,
        ctrl_idx: int,
        obs: Dict,
        step_index: int,
    ) -> Dict[str, Any]:
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        curr_pos = obs["centroid"][ctrl_idx].clone().to(self.device, dtype=torch.float32)
        curr_yaw = float(obs["yaw"][ctrl_idx].item())
        adv_motion_state = self._get_effective_adv_motion_state(
            obs=obs,
            ctrl_idx=ctrl_idx,
            step_index=step_index,
        )
        curr_speed = float(adv_motion_state["speed_mps"])
        total_time_s = float(self.executed_steps_per_call) * float(dt)
        theta_max_rad = float(self.reachability_max_abs_yaw_rate_rps) * total_time_s
        r_out = curr_speed * total_time_s + 0.5 * float(self.reachability_max_abs_accel_mps2) * total_time_s ** 2
        r_in = max(
            0.0,
            curr_speed * total_time_s - 0.5 * float(self.reachability_max_abs_accel_mps2) * total_time_s ** 2,
        )
        angle_offsets = np.linspace(-theta_max_rad, theta_max_rad, num=7, dtype=np.float32)
        outer_arc_sample_points_world: List[List[float]] = []
        inner_arc_sample_points_world: List[List[float]] = []
        for offset in angle_offsets.tolist():
            yaw = curr_yaw + float(offset)
            outer_arc_sample_points_world.append(
                [
                    float(curr_pos[0].item() + r_out * math.cos(yaw)),
                    float(curr_pos[1].item() + r_out * math.sin(yaw)),
                ]
            )
            if r_in > 1e-4:
                inner_arc_sample_points_world.append(
                    [
                        float(curr_pos[0].item() + r_in * math.cos(yaw)),
                        float(curr_pos[1].item() + r_in * math.sin(yaw)),
                    ]
                )
        return {
            "planning_duration_s": float(total_time_s),
            "step_dt_s": float(dt),
            "n_step_action": int(self.executed_steps_per_call),
            "current_position_world": curr_pos.detach().cpu().tolist(),
            "current_yaw_world": float(curr_yaw),
            "current_speed_mps": float(curr_speed),
            "max_abs_yaw_rate_rps": float(self.reachability_max_abs_yaw_rate_rps),
            "max_abs_accel_mps2": float(self.reachability_max_abs_accel_mps2),
            "theta_max_rad": float(theta_max_rad),
            "reachable_distance_range_m": [float(r_in), float(r_out)],
            "outer_arc_sample_points_world": outer_arc_sample_points_world,
            "inner_arc_sample_points_world": inner_arc_sample_points_world,
        }

    def _rollout_world_points_to_target(
        self,
        ctrl_idx: int,
        obs: Dict,
        target_position_world: List[float],
        target_yaw_world: float,
        target_horizon: int,
        step_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        curr_pos = obs["centroid"][ctrl_idx].clone().to(self.device, dtype=torch.float32)
        curr_yaw = float(obs["yaw"][ctrl_idx].item())
        adv_motion_state = self._get_effective_adv_motion_state(
            obs=obs,
            ctrl_idx=ctrl_idx,
            step_index=step_index,
        )
        initial_velocity_world = adv_motion_state["velocity_world"]
        rollout_steps = max(1, int(target_horizon))
        points = torch.zeros((rollout_steps, 2), dtype=torch.float32, device=self.device)
        yaws_world = torch.zeros((rollout_steps,), dtype=torch.float32, device=self.device)
        velocities = torch.zeros((rollout_steps, 2), dtype=torch.float32, device=self.device)
        current_speed = float(math.hypot(float(initial_velocity_world[0]), float(initial_velocity_world[1])))
        total_time_s = max(float(rollout_steps) * float(dt), 1e-3)
        target_point = torch.tensor(target_position_world, dtype=torch.float32, device=self.device)
        target_yaw = float(_wrap_float_to_pi(target_yaw_world))
        target_distance = float(torch.linalg.norm(target_point - curr_pos).item())
        nominal_accel_mps2 = _clip(
            2.0 * (target_distance - current_speed * total_time_s) / max(total_time_s ** 2, 1e-6),
            -self.reachability_max_abs_accel_mps2,
            self.reachability_max_abs_accel_mps2,
        )
        terminal_speed = _clip(
            current_speed + nominal_accel_mps2 * total_time_s,
            0.0,
            self.max_speed_mps,
        )
        start_tangent = torch.tensor(initial_velocity_world, dtype=torch.float32, device=self.device) * total_time_s
        end_tangent = torch.tensor(
            [terminal_speed * math.cos(target_yaw), terminal_speed * math.sin(target_yaw)],
            dtype=torch.float32,
            device=self.device,
        ) * total_time_s
        tangent_cap = max(target_distance * 1.25 + 1.0, 1.0)
        start_norm = float(torch.linalg.norm(start_tangent).item())
        if start_norm > tangent_cap:
            start_tangent = start_tangent * (tangent_cap / max(start_norm, 1e-6))
        end_norm = float(torch.linalg.norm(end_tangent).item())
        if end_norm > tangent_cap:
            end_tangent = end_tangent * (tangent_cap / max(end_norm, 1e-6))

        def hermite_position(s: float) -> torch.Tensor:
            h00 = 2.0 * s ** 3 - 3.0 * s ** 2 + 1.0
            h10 = s ** 3 - 2.0 * s ** 2 + s
            h01 = -2.0 * s ** 3 + 3.0 * s ** 2
            h11 = s ** 3 - s ** 2
            return h00 * curr_pos + h10 * start_tangent + h01 * target_point + h11 * end_tangent

        def hermite_velocity(s: float) -> torch.Tensor:
            dh00 = 6.0 * s ** 2 - 6.0 * s
            dh10 = 3.0 * s ** 2 - 4.0 * s + 1.0
            dh01 = -6.0 * s ** 2 + 6.0 * s
            dh11 = 3.0 * s ** 2 - 2.0 * s
            vel = (
                dh00 * curr_pos
                + dh10 * start_tangent
                + dh01 * target_point
                + dh11 * end_tangent
            ) / total_time_s
            speed = float(torch.linalg.norm(vel).item())
            if speed > self.max_speed_mps:
                vel = vel * (self.max_speed_mps / max(speed, 1e-6))
            return vel

        prev_point = curr_pos
        total_distance_travelled_m = 0.0
        for step_idx in range(rollout_steps):
            s = float(step_idx + 1) / float(rollout_steps)
            point = hermite_position(s)
            velocity_world = hermite_velocity(s)
            speed = float(torch.linalg.norm(velocity_world).item())
            if speed > 1e-4:
                yaw_world = math.atan2(float(velocity_world[1].item()), float(velocity_world[0].item()))
            else:
                yaw_world = _wrap_float_to_pi(curr_yaw + s * _wrap_float_to_pi(target_yaw - curr_yaw))
            points[step_idx] = point
            yaws_world[step_idx] = float(_wrap_float_to_pi(yaw_world))
            velocities[step_idx] = velocity_world
            total_distance_travelled_m += float(torch.linalg.norm(point - prev_point).item())
            prev_point = point

        points[-1] = target_point
        yaws_world[-1] = float(target_yaw)
        if rollout_steps == 1:
            velocities[-1] = (points[-1] - curr_pos) / max(float(dt), 1e-6)
        else:
            velocities[-1] = (points[-1] - points[-2]) / max(float(dt), 1e-6)
        last_speed = float(torch.linalg.norm(velocities[-1]).item())
        if last_speed > self.max_speed_mps:
            velocities[-1] = velocities[-1] * (self.max_speed_mps / max(last_speed, 1e-6))

        rollout_meta = {
            "target_position_world_used": copy.deepcopy(target_position_world),
            "target_yaw_world_used": float(target_yaw),
            "returned_steps": int(rollout_steps),
            "step_dt_s": float(dt),
            "end_reason": "reachable_target_rollout",
            "initial_velocity_world_used": list(initial_velocity_world),
            "initial_speed_mps_used": float(adv_motion_state["speed_mps"]),
            "initial_speed_source": str(adv_motion_state.get("source", "unknown")),
            "trajectory_mode": "cubic_hermite_reachable_target",
            "nominal_accel_mps2_used": float(nominal_accel_mps2),
            "terminal_speed_mps_used": float(terminal_speed),
            "total_distance_travelled_m": float(total_distance_travelled_m),
        }
        return points, yaws_world, velocities, rollout_meta

    def _rollout_world_points_longitudinal_action(
        self,
        ctrl_idx: int,
        obs: Dict,
        accel_mps2: float,
        target_horizon: int,
        step_index: int,
        action_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        curr_pos = obs["centroid"][ctrl_idx].clone().to(self.device, dtype=torch.float32)
        curr_yaw = float(obs["yaw"][ctrl_idx].item())
        adv_motion_state = self._get_effective_adv_motion_state(
            obs=obs,
            ctrl_idx=ctrl_idx,
            step_index=step_index,
        )
        velocity_world = torch.tensor(
            adv_motion_state["velocity_world"], dtype=torch.float32, device=self.device
        )
        current_speed = float(adv_motion_state["speed_mps"])
        rollout_steps = max(1, int(target_horizon))
        accel_cmd = float(_clip(accel_mps2, -self.max_abs_accel_mps2, self.max_abs_accel_mps2))

        if action_type == "slow_down":
            accel_cmd = min(accel_cmd, 0.0)
        elif action_type == "speed_up":
            accel_cmd = max(accel_cmd, 0.0)

        points = torch.zeros((rollout_steps, 2), dtype=torch.float32, device=self.device)
        yaws_world = torch.full((rollout_steps,), float(curr_yaw), dtype=torch.float32, device=self.device)
        velocities = torch.zeros((rollout_steps, 2), dtype=torch.float32, device=self.device)
        total_distance_travelled_m = 0.0
        prev_pos = curr_pos.clone()
        heading_dir = torch.tensor(
            [math.cos(curr_yaw), math.sin(curr_yaw)], dtype=torch.float32, device=self.device
        )
        if current_speed > 1e-4:
            heading_dir = velocity_world / max(float(torch.linalg.norm(velocity_world).item()), 1e-6)

        speed = float(current_speed)
        for step_idx in range(rollout_steps):
            speed = _clip(speed + accel_cmd * dt, 0.0, self.max_speed_mps)
            velocity_world = heading_dir * speed
            curr_pos = curr_pos + velocity_world * dt
            points[step_idx] = curr_pos
            velocities[step_idx] = velocity_world
            total_distance_travelled_m += float(torch.linalg.norm(curr_pos - prev_pos).item())
            prev_pos = curr_pos.clone()

        rollout_meta = {
            "returned_steps": int(rollout_steps),
            "step_dt_s": float(dt),
            "end_reason": "longitudinal_action_rollout",
            "initial_velocity_world_used": copy.deepcopy(adv_motion_state["velocity_world"]),
            "initial_speed_mps_used": float(adv_motion_state["speed_mps"]),
            "initial_speed_source": str(adv_motion_state.get("source", "unknown")),
            "trajectory_mode": f"{action_type}_longitudinal",
            "longitudinal_accel_mps2_used": float(accel_cmd),
            "terminal_speed_mps_used": float(speed),
            "total_distance_travelled_m": float(total_distance_travelled_m),
            "target_position_world_used": points[-1].detach().cpu().tolist(),
            "target_yaw_world_used": float(curr_yaw),
        }
        return points, yaws_world, velocities, rollout_meta

    def _compute_waypoint_steps(
        self,
        total_horizon_steps: Optional[int] = None,
        interval_steps: Optional[int] = None,
    ) -> List[int]:
        total_steps = max(1, int(total_horizon_steps or self.waypoint_horizon_steps))
        stride = max(1, int(interval_steps or self.waypoint_interval_steps))
        steps = list(range(stride, total_steps + 1, stride))
        if not steps or steps[-1] != total_steps:
            steps.append(total_steps)
        return steps

    def _rollout_world_points_from_waypoints(
        self,
        ctrl_idx: int,
        obs: Dict,
        waypoints_world: List[Dict[str, Any]],
        rollout_horizon_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        current_pos = obs["centroid"][ctrl_idx].clone().to(self.device, dtype=torch.float32)
        current_yaw = float(obs["yaw"][ctrl_idx].item())
        rollout_steps = max(1, int(rollout_horizon_steps))

        points = torch.zeros((rollout_steps, 2), dtype=torch.float32, device=self.device)
        prev_step = 0
        prev_pos = current_pos.clone()
        for waypoint in waypoints_world:
            target_step = max(prev_step + 1, int(waypoint["step"]))
            target_step = min(target_step, rollout_steps)
            target_pos = torch.tensor(
                waypoint["position_world"], dtype=torch.float32, device=self.device
            )
            segment_len = max(1, target_step - prev_step)
            for step_number in range(prev_step + 1, target_step + 1):
                alpha = float(step_number - prev_step) / float(segment_len)
                points[step_number - 1] = prev_pos + alpha * (target_pos - prev_pos)
            prev_step = target_step
            prev_pos = target_pos
            if prev_step >= rollout_steps:
                break

        if prev_step < rollout_steps:
            points[prev_step:] = prev_pos

        yaws_world = self._compute_world_yaws_from_points(points, current_yaw=current_yaw)
        velocities = torch.zeros((rollout_steps, 2), dtype=torch.float32, device=self.device)
        if rollout_steps == 1:
            velocities[0] = (points[0] - current_pos) / max(float(dt), 1e-6)
        else:
            velocities[0] = (points[0] - current_pos) / max(float(dt), 1e-6)
            velocities[1:] = (points[1:] - points[:-1]) / max(float(dt), 1e-6)

        total_distance_travelled_m = float(
            torch.norm(
                torch.diff(torch.cat([current_pos.unsqueeze(0), points], dim=0), dim=0),
                dim=-1,
            ).sum().item()
        )
        rollout_meta = {
            "returned_steps": int(rollout_steps),
            "step_dt_s": float(dt),
            "end_reason": "waypoint_rollout",
            "trajectory_mode": "piecewise_linear_waypoints",
            "waypoints_world_used": copy.deepcopy(waypoints_world),
            "waypoint_horizon_steps_used": int(rollout_steps),
            "total_distance_travelled_m": float(total_distance_travelled_m),
        }
        return points, yaws_world, velocities, rollout_meta

    def _validate_waypoint_segment_dynamics(
        self,
        ctrl_idx: int,
        obs: Dict,
        waypoints_world: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        dt = max(float(dt), 1e-6)
        prev_step = 0
        prev_pos = obs["centroid"][ctrl_idx].detach().to(self.device, dtype=torch.float32)
        prev_speed = float(obs["curr_speed"][ctrl_idx].item()) if "curr_speed" in obs else 0.0
        prev_heading = float(obs["yaw"][ctrl_idx].item())

        segment_summaries: List[Dict[str, Any]] = []
        speed_jumps: List[float] = []
        heading_jumps: List[float] = []
        segment_accels: List[float] = []
        segment_yaw_rates: List[float] = []
        for segment_idx, waypoint in enumerate(waypoints_world):
            step = max(prev_step + 1, int(waypoint["step"]))
            target_pos = torch.tensor(
                waypoint["position_world"],
                dtype=torch.float32,
                device=self.device,
            )
            delta = target_pos - prev_pos
            distance = float(torch.linalg.norm(delta).item())
            duration_s = max(float(step - prev_step) * dt, dt)
            speed = distance / duration_s
            if distance > 1e-4:
                heading = math.atan2(float(delta[1].item()), float(delta[0].item()))
            else:
                heading = prev_heading
            speed_delta = float(speed - prev_speed)
            heading_delta = _wrap_float_to_pi(float(heading - prev_heading))
            speed_jump = abs(speed_delta)
            heading_jump = abs(heading_delta)
            accel_mps2 = speed_delta / duration_s
            yaw_rate_rps = heading_delta / duration_s
            speed_jumps.append(float(speed_jump))
            heading_jumps.append(float(heading_jump))
            segment_accels.append(float(accel_mps2))
            segment_yaw_rates.append(float(yaw_rate_rps))
            segment_summaries.append(
                {
                    "segment_index": int(segment_idx),
                    "from_step": int(prev_step),
                    "to_step": int(step),
                    "duration_s": float(duration_s),
                    "distance_m": float(distance),
                    "speed_mps": float(speed),
                    "speed_delta_mps": float(speed_delta),
                    "speed_jump_mps": float(speed_jump),
                    "heading_world_rad": float(heading),
                    "heading_delta_rad": float(heading_delta),
                    "heading_jump_rad": float(heading_jump),
                    "segment_accel_mps2": float(accel_mps2),
                    "segment_yaw_rate_rps": float(yaw_rate_rps),
                }
            )
            prev_step = int(step)
            prev_pos = target_pos
            prev_speed = float(speed)
            prev_heading = float(heading)

        max_speed_jump = max(speed_jumps) if speed_jumps else 0.0
        max_heading_jump = max(heading_jumps) if heading_jumps else 0.0
        max_abs_accel = max((abs(value) for value in segment_accels), default=0.0)
        max_abs_yaw_rate = max((abs(value) for value in segment_yaw_rates), default=0.0)
        speed_exceeded = bool(max_speed_jump > self.validation_max_waypoint_speed_jump_mps)
        heading_exceeded = bool(max_heading_jump > self.validation_max_waypoint_heading_jump_rad)
        accel_exceeded = bool(max_abs_accel > self.validation_max_abs_accel_mps2)
        yaw_rate_exceeded = bool(max_abs_yaw_rate > self.validation_max_abs_yaw_rate_rps)
        status = "failed" if accel_exceeded or yaw_rate_exceeded else "ok"
        return {
            "waypoint_segment_validation_status": status,
            "waypoint_segment_validation_failure_reason": (
                "waypoint_segment_dynamics_exceeded" if status == "failed" else None
            ),
            "waypoint_segment_validation_mode": "segment_accel_yaw_rate",
            "dynamics_measurement_source": "sparse_waypoint_segments",
            "validation_reject_waypoint_segment_jumps": bool(self.validation_reject_waypoint_segment_jumps),
            "validation_reject_waypoint_segment_dynamics": bool(self.validation_reject_waypoint_segment_jumps),
            "validation_max_abs_accel_limit_mps2": float(self.validation_max_abs_accel_mps2),
            "validation_max_abs_yaw_rate_limit_rps": float(self.validation_max_abs_yaw_rate_rps),
            "validation_max_waypoint_speed_jump_limit_mps": float(self.validation_max_waypoint_speed_jump_mps),
            "validation_max_waypoint_heading_jump_limit_rad": float(self.validation_max_waypoint_heading_jump_rad),
            "max_abs_accel_mps2": float(max_abs_accel),
            "max_abs_yaw_rate_rps": float(max_abs_yaw_rate),
            "max_waypoint_segment_abs_accel_mps2": float(max_abs_accel),
            "max_waypoint_segment_abs_yaw_rate_rps": float(max_abs_yaw_rate),
            "max_waypoint_speed_jump_mps": float(max_speed_jump),
            "max_waypoint_heading_jump_rad": float(max_heading_jump),
            "validation_accel_exceeded": accel_exceeded,
            "validation_yaw_rate_exceeded": yaw_rate_exceeded,
            "waypoint_segment_accel_exceeded": accel_exceeded,
            "waypoint_segment_yaw_rate_exceeded": yaw_rate_exceeded,
            "waypoint_speed_jump_exceeded": speed_exceeded,
            "waypoint_heading_jump_exceeded": heading_exceeded,
            "waypoint_segment_summaries": segment_summaries,
        }

    @staticmethod
    def _infer_active_stage_id_from_queue(
        waypoint_queue: Optional[List[Dict[str, Any]]],
        fallback_stage_id: Optional[int] = None,
    ) -> Optional[int]:
        if isinstance(waypoint_queue, list):
            for waypoint in waypoint_queue:
                if not isinstance(waypoint, dict):
                    continue
                stage_id = _safe_int(waypoint.get("stage_id"), None)
                if stage_id is not None:
                    return int(stage_id)
        if fallback_stage_id is None:
            return None
        return int(fallback_stage_id)

    @staticmethod
    def _shift_remaining_waypoint_plan(
        waypoint_queue: Optional[List[Dict[str, Any]]],
        executed_steps: int,
    ) -> List[Dict[str, Any]]:
        if not isinstance(waypoint_queue, list):
            return []
        remaining: List[Dict[str, Any]] = []
        executed_steps = max(0, int(executed_steps))
        for waypoint in waypoint_queue:
            if not isinstance(waypoint, dict):
                continue
            step_value = _safe_int(waypoint.get("step"), None)
            if step_value is None or step_value <= executed_steps:
                continue
            shifted = copy.deepcopy(waypoint)
            shifted["step"] = int(step_value - executed_steps)
            remaining.append(shifted)
        return remaining

    def _normalize_macro_action(self, parsed: Dict) -> Optional[Dict[str, Any]]:
        raw_macro = parsed.get("macro_action")
        params: Dict[str, Any] = {}
        action_name: Optional[str] = None

        if isinstance(raw_macro, str):
            action_name = raw_macro.strip()
        elif isinstance(raw_macro, dict):
            action_name = str(raw_macro.get("name", "")).strip()
            params.update(raw_macro)
            if isinstance(raw_macro.get("params"), dict):
                params.update(raw_macro["params"])
        elif isinstance(parsed.get("action"), str):
            action_name = str(parsed.get("action")).strip()

        if isinstance(parsed.get("params"), dict):
            params.update(parsed["params"])
        for key in [
            "accel_mps2",
            "target_speed_mps",
            "target_offset_m",
            "lane_change_direction",
            "lane_change_duration_steps",
            "hold_steps",
            "aggressiveness",
        ]:
            if key in parsed and key not in params:
                params[key] = parsed.get(key)

        if not action_name:
            return None

        action_name = action_name.lower().strip()
        valid_names = {
            "keep_lane",
            "accelerate_to_conflict",
            "brake_to_yield_then_cut",
            "start_lane_change_left",
            "start_lane_change_right",
            "continue_lane_change",
            "abort_lane_change",
            "merge_and_brake",
        }
        if action_name not in valid_names:
            return None

        lane_dir_raw = str(params.get("lane_change_direction", "")).lower().strip()
        if lane_dir_raw not in {"left", "right"}:
            if action_name.endswith("_left"):
                lane_dir_raw = "left"
            elif action_name.endswith("_right"):
                lane_dir_raw = "right"
            else:
                lane_dir_raw = "left"

        accel = _safe_float(params.get("accel_mps2"), None)
        target_speed = _safe_float(params.get("target_speed_mps"), None)
        target_offset = _safe_float(params.get("target_offset_m"), None)
        lane_change_duration = _safe_int(
            params.get("lane_change_duration_steps"),
            self.default_lane_change_duration_steps,
        )
        hold_steps = _safe_int(params.get("hold_steps"), self.default_macro_hold_steps)
        aggressiveness = _safe_float(params.get("aggressiveness"), 0.5)
        if target_offset is not None:
            target_offset = _clip(float(target_offset), -1.5 * self.default_lane_width_m, 1.5 * self.default_lane_width_m)

        return {
            "name": action_name,
            "accel_mps2": accel,
            "target_speed_mps": target_speed,
            "target_offset_m": target_offset,
            "lane_change_direction": lane_dir_raw,
            "lane_change_duration_steps": max(2, int(lane_change_duration)),
            "hold_steps": max(1, int(hold_steps)),
            "aggressiveness": _clip(float(aggressiveness), 0.0, 1.0),
            "raw_macro_action": raw_macro,
        }

    def _macro_accel_profile(
        self,
        macro: Dict[str, Any],
        current_speed: float,
        dt: float,
        horizon: int,
    ) -> torch.Tensor:
        accel = torch.zeros(horizon, dtype=torch.float32, device=self.device)
        name = str(macro["name"])
        hold_steps = int(macro["hold_steps"])
        aggr = float(macro["aggressiveness"])
        lane_change_steps = int(macro["lane_change_duration_steps"])

        if macro["accel_mps2"] is not None:
            base_acc = _clip(float(macro["accel_mps2"]), -self.max_abs_accel_mps2, self.max_abs_accel_mps2)
        else:
            defaults = {
                "keep_lane": 0.0,
                "accelerate_to_conflict": 1.2 + 1.2 * aggr,
                "brake_to_yield_then_cut": -1.5 - 1.0 * aggr,
                "start_lane_change_left": 0.4,
                "start_lane_change_right": 0.4,
                "continue_lane_change": 0.2,
                "abort_lane_change": -0.4,
                "merge_and_brake": -1.8 - 0.8 * aggr,
            }
            base_acc = defaults.get(name, 0.0)

        if name in {"keep_lane", "accelerate_to_conflict", "start_lane_change_left", "start_lane_change_right", "continue_lane_change"}:
            accel[:] = base_acc
        elif name == "abort_lane_change":
            accel[:hold_steps] = base_acc
            accel[hold_steps:] = 0.0
        elif name == "merge_and_brake":
            accel[:] = base_acc
        elif name == "brake_to_yield_then_cut":
            brake_steps = max(2, min(horizon - 1, hold_steps))
            accel[:brake_steps] = base_acc
            accel[brake_steps : min(horizon, brake_steps + lane_change_steps)] = 0.6 + 0.6 * aggr
            if brake_steps + lane_change_steps < horizon:
                accel[brake_steps + lane_change_steps :] = 0.0
        accel = torch.clamp(accel, -self.max_abs_accel_mps2, self.max_abs_accel_mps2)

        target_speed = macro.get("target_speed_mps")
        if target_speed is not None:
            v = torch.zeros(horizon, dtype=torch.float32, device=self.device)
            curr = float(max(0.0, current_speed))
            tgt = _clip(float(target_speed), 0.0, self.max_speed_mps)
            if name == "accelerate_to_conflict":
                # Avoid over-speeding through the conflict area before ego reaches it.
                tgt = min(tgt, max(12.0, current_speed + 6.0))
            for i in range(horizon):
                curr = max(0.0, curr + float(accel[i].item()) * dt)
                if (accel[i] >= 0 and curr >= tgt) or (accel[i] < 0 and curr <= tgt):
                    curr = tgt
                    accel[i:] = 0.0
                v[i] = curr
        return accel

    def _macro_offset_profile(
        self,
        macro: Dict[str, Any],
        current_offset_m: float,
        horizon: int,
    ) -> torch.Tensor:
        name = str(macro["name"])
        target_offset_raw = macro.get("target_offset_m")
        lane_change_steps = int(macro["lane_change_duration_steps"])
        hold_steps = int(macro["hold_steps"])
        lane_dir = str(macro.get("lane_change_direction", "left")).lower()
        offsets = torch.full((horizon,), float(current_offset_m), dtype=torch.float32, device=self.device)
        default_sign = 1.0 if lane_dir == "left" else -1.0
        default_target = default_sign * 0.9 * self.default_lane_width_m
        target_offset = float(target_offset_raw) if target_offset_raw is not None else default_target

        def _ramp(start: float, end: float, steps: int) -> torch.Tensor:
            steps = max(1, min(horizon, steps))
            if steps == 1:
                return torch.tensor([end], dtype=torch.float32, device=self.device)
            return torch.linspace(start, end, steps, device=self.device, dtype=torch.float32)

        if name == "keep_lane":
            return offsets
        if name == "accelerate_to_conflict":
            ramp_steps = max(4, min(horizon, lane_change_steps))
            ramp = _ramp(current_offset_m, target_offset, ramp_steps)
            offsets[: ramp.shape[0]] = ramp
            if ramp.shape[0] < horizon:
                offsets[ramp.shape[0] :] = target_offset
            return offsets
        if name in {"start_lane_change_left", "start_lane_change_right", "continue_lane_change"}:
            ramp = _ramp(current_offset_m, target_offset, lane_change_steps)
            offsets[: ramp.shape[0]] = ramp
            if ramp.shape[0] < horizon:
                offsets[ramp.shape[0] :] = target_offset
            return offsets
        if name == "abort_lane_change":
            ramp = _ramp(current_offset_m, 0.0, lane_change_steps)
            offsets[: ramp.shape[0]] = ramp
            if ramp.shape[0] < horizon:
                offsets[ramp.shape[0] :] = 0.0
            return offsets
        if name == "merge_and_brake":
            ramp = _ramp(current_offset_m, 0.0, lane_change_steps)
            offsets[: ramp.shape[0]] = ramp
            if ramp.shape[0] < horizon:
                offsets[ramp.shape[0] :] = 0.0
            return offsets
        if name == "brake_to_yield_then_cut":
            stable = max(1, min(horizon, hold_steps))
            offsets[:stable] = current_offset_m
            if stable < horizon:
                ramp = _ramp(current_offset_m, target_offset, lane_change_steps)
                end_idx = min(horizon, stable + ramp.shape[0])
                offsets[stable:end_idx] = ramp[: end_idx - stable]
                if end_idx < horizon:
                    offsets[end_idx:] = target_offset
            return offsets
        return offsets

    def _resolve_target_offset(
        self,
        macro: Dict[str, Any],
        current_offset_m: float,
        base_xy: torch.Tensor,
        normal_query: torch.Tensor,
        ego_plan_world: Optional[torch.Tensor],
    ) -> Tuple[float, str]:
        name = str(macro.get("name", ""))
        explicit_target = macro.get("target_offset_m")
        if explicit_target is not None:
            return float(explicit_target), "llm_explicit"

        lane_dir = str(macro.get("lane_change_direction", "left")).lower()
        dir_sign = 1.0 if lane_dir == "left" else -1.0
        default_target = dir_sign * 0.9 * self.default_lane_width_m

        if name not in {"accelerate_to_conflict", "start_lane_change_left", "start_lane_change_right", "continue_lane_change", "brake_to_yield_then_cut"}:
            return default_target, "default_non_conflict"

        if ego_plan_world is None or ego_plan_world.ndim != 2 or ego_plan_world.shape[0] < 2:
            return default_target, "default_no_ego_plan"

        m = min(base_xy.shape[0], ego_plan_world.shape[0], normal_query.shape[0])
        if m < 2:
            return default_target, "default_short_horizon"

        # Candidate offsets: left, center, right (in centerline-normal space).
        candidates = [
            -0.9 * self.default_lane_width_m,
            0.0,
            0.9 * self.default_lane_width_m,
        ]
        best_offset = default_target
        best_score = float("inf")
        for cand in candidates:
            adv_pts = base_xy[:m] + normal_query[:m] * float(cand)
            dist_aligned = torch.norm(adv_pts - ego_plan_world[:m], dim=-1)
            score = float(dist_aligned.min().item()) + 0.15 * float(dist_aligned.mean().item())
            if score < best_score:
                best_score = score
                best_offset = float(cand)

        # Keep current side if the selected one provides negligible improvement.
        if abs(best_offset - current_offset_m) < 0.15:
            return float(current_offset_m), "auto_keep_current"
        return best_offset, "auto_from_ego_plan"

    @staticmethod
    def _trajectory_alignment_score(
        adv_world_points: torch.Tensor,
        ego_plan_world: Optional[torch.Tensor],
    ) -> float:
        if ego_plan_world is None or adv_world_points.ndim != 2 or ego_plan_world.ndim != 2:
            return float("inf")
        m = min(adv_world_points.shape[0], ego_plan_world.shape[0])
        if m < 2:
            return float("inf")
        dist = torch.norm(adv_world_points[:m] - ego_plan_world[:m], dim=-1)
        return float(dist.min().item()) + 0.2 * float(dist.mean().item())

    def _generate_world_points_from_macro(
        self,
        macro: Dict[str, Any],
        obs: Dict,
        ctrl_idx: int,
        target_horizon: int,
        ego_plan_world: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], str]:
        if target_horizon < 2:
            return None, "target_horizon_too_small"

        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        current_pos = obs["centroid"][ctrl_idx]
        current_speed = float(obs["curr_speed"][ctrl_idx].item())
        current_yaw = float(obs["yaw"][ctrl_idx].item())
        centerline_world = self._get_centerline_world_tensor(obs, ctrl_idx)

        if centerline_world is None or centerline_world.shape[0] < 2:
            # Fallback: straight rollout in current heading.
            accel = self._macro_accel_profile(macro, current_speed=current_speed, dt=dt, horizon=target_horizon)
            v = torch.zeros(target_horizon, dtype=torch.float32, device=self.device)
            curr_v = max(0.0, current_speed)
            for i in range(target_horizon):
                curr_v = max(0.0, min(self.max_speed_mps, curr_v + float(accel[i].item()) * dt))
                v[i] = curr_v
            dist = torch.cumsum(v * dt, dim=0)
            direction = torch.tensor([math.cos(current_yaw), math.sin(current_yaw)], device=self.device, dtype=torch.float32)
            points = current_pos.unsqueeze(0) + dist.unsqueeze(-1) * direction.unsqueeze(0)
            return points, "ok_macro_straight_fallback"

        s_nodes, _, normals = self._centerline_geometry(centerline_world)
        closest_idx = int(torch.argmin(torch.norm(centerline_world - current_pos.unsqueeze(0), dim=-1)).item())
        s0 = float(s_nodes[closest_idx].item())
        current_offset = self._estimate_lateral_offset(current_pos, centerline_world, normals)

        def _rollout_with_macro(macro_eval: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], str]:
            accel = self._macro_accel_profile(macro_eval, current_speed=current_speed, dt=dt, horizon=target_horizon)
            speeds = torch.zeros(target_horizon, dtype=torch.float32, device=self.device)
            curr_v = max(0.0, current_speed)
            for i in range(target_horizon):
                curr_v = max(0.0, min(self.max_speed_mps, curr_v + float(accel[i].item()) * dt))
                speeds[i] = curr_v
            ds = speeds * dt
            s_query = s0 + torch.cumsum(ds, dim=0)
            base_xy, idx_left = self._interp_path_by_s(centerline_world, s_nodes, s_query)
            normal_query = normals[idx_left]

            resolved_target_offset, offset_src = self._resolve_target_offset(
                macro=macro_eval,
                current_offset_m=current_offset,
                base_xy=base_xy,
                normal_query=normal_query,
                ego_plan_world=ego_plan_world,
            )
            macro_used = dict(macro_eval)
            macro_used["target_offset_m"] = _clip(
                float(resolved_target_offset),
                -1.5 * self.default_lane_width_m,
                1.5 * self.default_lane_width_m,
            )
            macro_used["target_offset_source"] = offset_src
            offsets = self._macro_offset_profile(macro_used, current_offset_m=current_offset, horizon=target_horizon)
            world_points = base_xy + offsets.unsqueeze(-1) * normal_query
            return world_points, macro_used, offset_src

        macro_name = str(macro.get("name", ""))
        if macro_name == "accelerate_to_conflict" and ego_plan_world is not None and ego_plan_world.shape[0] >= 2:
            ego_dt = max(dt, 1e-3)
            ego_seg = torch.norm(ego_plan_world[1:] - ego_plan_world[:-1], dim=-1)
            ego_speed_mean = float((ego_seg / ego_dt).mean().item()) if ego_seg.numel() > 0 else 8.0
            speed_candidates = [
                max(4.0, ego_speed_mean - 2.0),
                max(5.0, ego_speed_mean),
                max(6.0, ego_speed_mean + 2.0),
                max(7.0, ego_speed_mean + 4.0),
                max(8.0, ego_speed_mean + 6.0),
            ]
            if macro.get("target_speed_mps") is not None:
                speed_candidates.append(float(macro["target_speed_mps"]))
            speed_candidates = sorted({round(_clip(v, 2.0, self.max_speed_mps), 2) for v in speed_candidates})

            best_world_points: Optional[torch.Tensor] = None
            best_macro_used: Optional[Dict[str, Any]] = None
            best_offset_src = "auto_from_ego_plan"
            best_speed = None
            best_score = float("inf")
            for cand_speed in speed_candidates:
                macro_eval = dict(macro)
                macro_eval["target_speed_mps"] = float(cand_speed)
                world_points_i, macro_used_i, offset_src_i = _rollout_with_macro(macro_eval)
                score_i = self._trajectory_alignment_score(world_points_i, ego_plan_world)
                if score_i < best_score:
                    best_score = score_i
                    best_world_points = world_points_i
                    best_macro_used = macro_used_i
                    best_offset_src = offset_src_i
                    best_speed = float(cand_speed)
            if best_world_points is not None and best_macro_used is not None and best_speed is not None:
                return best_world_points, (
                    f"ok_macro:{best_offset_src}:offset={best_macro_used['target_offset_m']:.3f}"
                    f":target_speed={best_speed:.2f}:align_score={best_score:.3f}"
                )

        world_points, macro_used, offset_src = _rollout_with_macro(macro)
        return world_points, f"ok_macro:{offset_src}:offset={macro_used['target_offset_m']:.3f}"

    def _get_ref_centerline_world(self, obs: Dict, agent_idx: int) -> List[List[float]]:
        extras = obs.get("extras", {})
        centerline_xy = extras.get("centerline_xy")
        if centerline_xy is None:
            return []
        local_centerline = centerline_xy[agent_idx]
        valid = torch.isfinite(local_centerline).all(dim=-1)
        local_centerline = local_centerline[valid]
        if local_centerline.ndim != 2 or local_centerline.shape[-1] != 2 or local_centerline.shape[0] < 2:
            return []
        world_centerline = self._to_world(local_centerline, obs["world_from_agent"][agent_idx])
        current_pos = obs["centroid"][agent_idx].to(world_centerline.device, dtype=world_centerline.dtype)
        sampled_world = self._sample_route_near_current_position(world_centerline, current_pos)
        return sampled_world.detach().cpu().tolist()

    def _get_candidate_routes(self, obs: Dict, agent_idx: int) -> List[Dict[str, Any]]:
        candidate_lanes = self._get_candidate_lanes(obs, agent_idx)
        candidate_routes: List[Dict[str, Any]] = []
        for idx, lane in enumerate(candidate_lanes):
            candidate_routes.append(
                {
                    "route_id": lane.get("lane_id", f"cand_{idx}"),
                    "reference_route_world": lane.get("centerline_world", []),
                    "reference_route_yaw": lane.get("centerline_yaw", []),
                }
            )
        return candidate_routes

    def _get_route_info(self, obs: Dict, agent_idx: int) -> Dict[str, Any]:
        route_world = self._get_ref_centerline_world(obs, agent_idx)
        route_yaw: List[float] = []
        if len(route_world) >= 2:
            route_np = np.asarray(route_world, dtype=np.float32)
            route_yaw = self._polyline_yaws_numpy(route_np)
        return {
            "track_id": int(obs["track_id"][agent_idx].item()),
            "current_lane_id": self._get_agent_current_lane_id(obs, agent_idx),
            "reference_route_world": route_world,
            "reference_route_yaw": route_yaw,
            "candidate_routes": self._get_candidate_routes(obs, agent_idx),
        }

    def _get_agent_future_world(self, obs: Dict, agent_idx: int, steps: int) -> List[List[float]]:
        local_future = obs["target_positions"][agent_idx, :steps]
        world_future = self._to_world(local_future, obs["world_from_agent"][agent_idx])
        return world_future.detach().cpu().tolist()

    def _build_nominal_adv_trajectory(
        self,
        obs: Dict,
        ctrl_idx: int,
        horizon_steps: int,
        dt: float,
    ) -> Dict[str, Any]:
        """Build a simple rule/controller-like ADV nominal rollout.

        This is used by the LLMScenario-style baseline: the LLM receives a
        sane route-following or constant-velocity trajectory and rewrites it,
        instead of planning from a blank page.
        """
        horizon_steps = max(1, int(horizon_steps))
        current_pos = obs["centroid"][ctrl_idx].to(self.device, dtype=torch.float32)
        current_yaw = float(obs["yaw"][ctrl_idx].item())
        current_speed = max(0.0, float(obs["curr_speed"][ctrl_idx].item()))
        dt = max(float(dt), 1e-6)
        step_tensor = torch.arange(
            1,
            horizon_steps + 1,
            device=self.device,
            dtype=torch.float32,
        )

        source = "constant_velocity"
        points_world: torch.Tensor
        yaws_world: torch.Tensor
        route_world = self._get_centerline_world_tensor(obs, ctrl_idx)
        if route_world is not None and route_world.shape[0] >= 2:
            try:
                s_nodes, tangent, _ = self._centerline_geometry(route_world)
                closest_idx = int(torch.argmin(torch.norm(route_world - current_pos.unsqueeze(0), dim=-1)).item())
                query_s = s_nodes[closest_idx] + current_speed * dt * step_tensor
                points_world, idx_left = self._interp_path_by_s(route_world, s_nodes, query_s)
                yaws_world = torch.atan2(tangent[idx_left, 1], tangent[idx_left, 0])
                source = "route_following_centerline"
            except Exception:
                velocity_world = torch.tensor(
                    _velocity_world_from_speed_yaw(current_speed, current_yaw),
                    dtype=torch.float32,
                    device=self.device,
                )
                points_world = current_pos.unsqueeze(0) + step_tensor.unsqueeze(-1) * dt * velocity_world.unsqueeze(0)
                yaws_world = torch.full((horizon_steps,), current_yaw, dtype=torch.float32, device=self.device)
        else:
            velocity_world = torch.tensor(
                _velocity_world_from_speed_yaw(current_speed, current_yaw),
                dtype=torch.float32,
                device=self.device,
            )
            points_world = current_pos.unsqueeze(0) + step_tensor.unsqueeze(-1) * dt * velocity_world.unsqueeze(0)
            yaws_world = torch.full((horizon_steps,), current_yaw, dtype=torch.float32, device=self.device)

        waypoint_steps = self._compute_waypoint_steps(
            total_horizon_steps=horizon_steps,
            interval_steps=self.waypoint_interval_steps,
        )
        sparse = []
        for step in waypoint_steps:
            idx = min(max(int(step) - 1, 0), points_world.shape[0] - 1)
            sparse.append(
                {
                    "step": int(step),
                    "position_world": points_world[idx].detach().cpu().tolist(),
                    "yaw_world": float(yaws_world[idx].detach().cpu().item()),
                }
            )

        return {
            "source": source,
            "assumption": (
                "route-following constant-speed rollout when a centerline is available; "
                "otherwise constant-velocity rollout from current yaw/speed"
            ),
            "dense_trajectory_world": points_world.detach().cpu().tolist(),
            "dense_yaw_world": yaws_world.detach().cpu().tolist(),
            "sparse_trajectory": sparse,
        }

    @staticmethod
    def _angle_to_world_direction_label(angle_rad: float) -> str:
        labels = ["Right", "upper right", "Top", "upper left", "Left", "Bottom left", "Bottom", "lower right"]
        wrapped = _wrap_float_to_pi(float(angle_rad))
        bin_idx = int(np.round(wrapped / (math.pi / 4.0))) % 8
        return labels[bin_idx]

    def _build_ego_adv_interaction_analysis_text(
        self,
        adv_pos: torch.Tensor,
        adv_yaw: float,
        ego_pos: torch.Tensor,
        ego_yaw: float,
    ) -> str:
        adv_xy = adv_pos.detach().cpu().numpy().astype(np.float64)
        ego_xy = ego_pos.detach().cpu().numpy().astype(np.float64)
        rel_adv_to_ego = ego_xy - adv_xy
        rel_ego_to_adv = adv_xy - ego_xy
        distance_m = float(np.linalg.norm(rel_adv_to_ego))
        bearing_adv_to_ego = math.atan2(float(rel_adv_to_ego[1]), float(rel_adv_to_ego[0]))
        bearing_ego_to_adv = math.atan2(float(rel_ego_to_adv[1]), float(rel_ego_to_adv[0]))
        heading_diff_rad = _wrap_float_to_pi(float(ego_yaw) - float(adv_yaw))
        heading_diff_abs_rad = abs(heading_diff_rad)

        return (
            f"Position analysis:\n- EGO position is at {ego_xy.round(3).tolist()}, ADV position at {adv_xy.round(3).tolist()}。\n- relative_position_world = ego_position_world - adv_position_world = {rel_adv_to_ego.round(3).tolist()}, where dx = {rel_adv_to_ego[0]:.3f} m，dy = {rel_adv_to_ego[1]:.3f} m, representing the raw world-coordinate displacement from ADV to EGO.\n- Therefore EGO is at ADV's{self._angle_to_world_direction_label(bearing_adv_to_ego)}direction, ADV is in EGO's{self._angle_to_world_direction_label(bearing_ego_to_adv)}direction, the distance between their centers is {distance_m:.3f} m.\nDirection analysis:\n- ADV current yaw_world = {float(adv_yaw):.6f} rad, direction is {self._angle_to_world_direction_label(float(adv_yaw))}.\n- EGO current yaw_world = {float(ego_yaw):.6f} rad, direction is {self._angle_to_world_direction_label(float(ego_yaw))}。\n- heading_diff_rad = wrap_to_pi(ego_yaw - adv_yaw) = {heading_diff_rad:.6f} rad, the absolute difference is {heading_diff_abs_rad:.6f} rad。"
        )

    def _get_scene_indices(self, obs: Dict, scene_idx: int) -> torch.Tensor:
        scene_index = obs["scene_index"].reshape(-1)
        return torch.nonzero(scene_index == scene_idx, as_tuple=False).reshape(-1)

    def _get_effective_adv_motion_state(
        self,
        obs: Dict,
        ctrl_idx: int,
        step_index: int,
    ) -> Dict[str, Any]:
        curr_yaw = float(obs["yaw"][ctrl_idx].item())
        raw_speed = float(obs["curr_speed"][ctrl_idx].item())
        raw_velocity = _velocity_world_from_speed_yaw(raw_speed, curr_yaw)
        scene_idx = int(obs["scene_index"][ctrl_idx].item())
        ctrl_track_id = int(obs["track_id"][ctrl_idx].item())
        cache_key = (scene_idx, ctrl_track_id)
        cached_plan = self._last_plan_cache_by_agent.get(cache_key)
        if cached_plan is None:
            return {
                "speed_mps": raw_speed,
                "velocity_world": raw_velocity,
                "source": "obs_curr_speed",
            }

        elapsed_steps = int(step_index) - int(cached_plan.get("step_index", -1))
        if elapsed_steps <= 0:
            return {
                "speed_mps": raw_speed,
                "velocity_world": raw_velocity,
                "source": "obs_curr_speed",
            }

        velocity_world_used = cached_plan.get("velocity_world_used")
        trajectory_world_used = cached_plan.get("trajectory_world_used")
        if (
            not isinstance(velocity_world_used, list)
            or len(velocity_world_used) < elapsed_steps
            or not isinstance(trajectory_world_used, list)
            or len(trajectory_world_used) < elapsed_steps
        ):
            return {
                "speed_mps": raw_speed,
                "velocity_world": raw_velocity,
                "source": "obs_curr_speed",
            }

        cached_velocity = velocity_world_used[elapsed_steps - 1]
        cached_position = trajectory_world_used[elapsed_steps - 1]
        if (
            not isinstance(cached_velocity, (list, tuple))
            or len(cached_velocity) != 2
            or not isinstance(cached_position, (list, tuple))
            or len(cached_position) != 2
        ):
            return {
                "speed_mps": raw_speed,
                "velocity_world": raw_velocity,
                "source": "obs_curr_speed",
            }

        current_position = obs["centroid"][ctrl_idx].detach().to(self.device, dtype=torch.float32)
        predicted_position = torch.tensor(cached_position, dtype=torch.float32, device=self.device)
        position_error_m = float(torch.norm(current_position - predicted_position).item())
        cached_speed = float(math.hypot(float(cached_velocity[0]), float(cached_velocity[1])))

        should_use_cached = (
            position_error_m <= max(2.0, self.max_step_distance * 2.0)
            and cached_speed > 0.25
            and raw_speed < max(0.2, 0.5 * cached_speed)
        )
        if not should_use_cached:
            return {
                "speed_mps": raw_speed,
                "velocity_world": raw_velocity,
                "source": "obs_curr_speed",
                "raw_speed_mps": raw_speed,
                "predicted_speed_mps": cached_speed,
                "predicted_position_error_m": position_error_m,
            }

        return {
            "speed_mps": cached_speed,
            "velocity_world": [float(cached_velocity[0]), float(cached_velocity[1])],
            "source": "cached_plan_velocity",
            "raw_speed_mps": raw_speed,
            "predicted_position_error_m": position_error_m,
        }

    def _build_prompt_context(
        self,
        obs: Dict,
        ctrl_idx: int,
        ego_idx: int,
        step_index: int,
        planning_state: Optional[Dict[str, Any]] = None,
        failure_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        ctrl_pos = obs["centroid"][ctrl_idx]
        ctrl_yaw = obs["yaw"][ctrl_idx]
        ego_pos = obs["centroid"][ego_idx]
        adv_motion_state = self._get_effective_adv_motion_state(
            obs=obs,
            ctrl_idx=ctrl_idx,
            step_index=step_index,
        )
        adv_speed = float(adv_motion_state["speed_mps"])
        adv_velocity_world = list(adv_motion_state["velocity_world"])
        ego_speed = float(obs["curr_speed"][ego_idx].item())
        ego_velocity_world = _velocity_world_from_speed_yaw(ego_speed, float(obs["yaw"][ego_idx].item()))
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        planner_horizon = min(
            max(1, int(self.horizon)),
            int(obs["target_positions"].shape[1]),
        )
        execution_horizon = min(
            int(self.executed_steps_per_call),
            int(obs["target_positions"].shape[1]),
        )

        ego_plan = obs.get("ego_plan")
        if ego_plan is None:
            raise ValueError("ego_plan is missing from agent observation")
        ego_plan = ego_plan.to(self.device)
        if ego_plan.ndim == 2:
            ego_plan = ego_plan.unsqueeze(0)
        ego_indices = obs.get("ego_idx")
        if ego_indices is None:
            ego_plan = ego_plan[:1]
        else:
            if isinstance(ego_indices, torch.Tensor):
                ego_indices = ego_indices.detach().cpu().tolist()
            else:
                ego_indices = list(ego_indices)
            ego_plan_row = ego_indices.index(ego_idx)
            ego_plan = ego_plan[ego_plan_row : ego_plan_row + 1]
        if ego_plan.shape[1] < planner_horizon:
            ego_plan = extend_ego_plan(ego_plan, planner_horizon, dt=dt)
        ego_plan = ego_plan[:, :planner_horizon]
        ego_plan_world = self._to_world(
            ego_plan[0],
            obs["world_from_agent"][ego_idx].to(self.device),
        )
        ego_plan_steps = self._compute_waypoint_steps(
            total_horizon_steps=planner_horizon,
            interval_steps=self.waypoint_interval_steps,
        )
        ego_plan_world_sparse = ego_plan_world[[step - 1 for step in ego_plan_steps]]

        scene_idx = int(obs["scene_index"][ctrl_idx].item())
        scene_agent_indices = self._get_scene_indices(obs, scene_idx)
        distances = torch.norm(obs["centroid"][scene_agent_indices] - ctrl_pos, dim=-1)
        sorted_indices = scene_agent_indices[torch.argsort(distances)]

        nearby_agents = []
        if self.send_nearby_agents:
            for idx in sorted_indices.tolist():
                if idx == ctrl_idx or idx == ego_idx:
                    continue
                distance_to_adv = float(torch.norm(obs["centroid"][idx] - ctrl_pos, dim=-1).item())
                if distance_to_adv > self.nearby_radius_m:
                    continue
                agent_yaw = float(obs["yaw"][idx].item())
                agent_speed = float(obs["curr_speed"][idx].item())
                nearby_agent = {
                    "track_id": int(obs["track_id"][idx].item()),
                    "position_world": obs["centroid"][idx].detach().cpu().tolist(),
                    "yaw_world": agent_yaw,
                    "speed_mps": agent_speed,
                    "velocity_world": _velocity_world_from_speed_yaw(agent_speed, agent_yaw),
                    "extent_m": obs["extent"][idx, :2].detach().cpu().tolist(),
                    "current_lane_id": self._get_agent_current_lane_id(obs, idx),
                }
                if self.send_nearby_future_trajectory:
                    nearby_agent["future_trajectory_world"] = self._get_agent_future_world(
                        obs, idx, planner_horizon
                    )
                nearby_agents.append(nearby_agent)

        planning_state = planning_state if isinstance(planning_state, dict) else {}
        remaining_waypoint_plan = planning_state.get("remaining_waypoint_plan")
        stage_plan = planning_state.get("stage_plan")

        return {
            "scenario_goal": str(self.scenario_goal),
            "trajectory_horizon": int(planner_horizon),
            "execution_horizon": int(execution_horizon),
            "waypoint_horizon_steps": int(self.waypoint_horizon_steps),
            "waypoint_interval_steps": int(self.waypoint_interval_steps),
            "waypoint_steps": self._compute_waypoint_steps(),
            "step_dt_s": float(dt),
            "reachable_target_constraint": self._compute_reachable_target_constraint(
                ctrl_idx=ctrl_idx,
                obs=obs,
                step_index=step_index,
            ),
            "map_information": self._get_map_information(obs, adv_idx=ctrl_idx, ego_idx=ego_idx),
            "route_information": (
                {
                    "adv_route": self._get_route_info(obs, ctrl_idx),
                    "ego_route": self._get_route_info(obs, ego_idx),
                }
                if self.send_route_information
                else {}
            ),
            "adv_current_state": {
                "track_id": int(obs["track_id"][ctrl_idx].item()),
                "position_world": ctrl_pos.detach().cpu().tolist(),
                "yaw_world": float(ctrl_yaw.item()),
                "speed_mps": adv_speed,
                "velocity_world": adv_velocity_world,
                "extent_m": obs["extent"][ctrl_idx, :2].detach().cpu().tolist(),
                "current_lane_id": self._get_agent_current_lane_id(obs, ctrl_idx),
            },
            "ego_current_state": {
                "track_id": int(obs["track_id"][ego_idx].item()),
                "position_world": ego_pos.detach().cpu().tolist(),
                "yaw_world": float(obs["yaw"][ego_idx].item()),
                "speed_mps": ego_speed,
                "velocity_world": ego_velocity_world,
                "extent_m": obs["extent"][ego_idx, :2].detach().cpu().tolist(),
                "current_lane_id": self._get_agent_current_lane_id(obs, ego_idx),
            },
            "ego_adv_relative_state": {
                "relative_position_world": (ego_pos - ctrl_pos).detach().cpu().tolist(),
                "relative_velocity_world": [
                    float(ego_velocity_world[0] - adv_velocity_world[0]),
                    float(ego_velocity_world[1] - adv_velocity_world[1]),
                ],
                "distance_m": float(torch.norm(ego_pos - ctrl_pos, dim=-1).item()),
            },
            "ego_planned_trajectory_steps": [int(step) for step in ego_plan_steps],
            "ego_planned_trajectory_world": ego_plan_world_sparse.detach().cpu().tolist(),
            "adv_nominal_trajectory": self._build_nominal_adv_trajectory(
                obs=obs,
                ctrl_idx=ctrl_idx,
                horizon_steps=planner_horizon,
                dt=dt,
            ),
            "ego_adv_interaction_analysis": self._build_ego_adv_interaction_analysis_text(
                adv_pos=ctrl_pos,
                adv_yaw=float(ctrl_yaw.item()),
                ego_pos=ego_pos,
                ego_yaw=float(obs["yaw"][ego_idx].item()),
            ),
            "nearby_agents": nearby_agents,
            "stage_plan": copy.deepcopy(stage_plan) if isinstance(stage_plan, list) else None,
            "remaining_waypoint_plan": (
                copy.deepcopy(remaining_waypoint_plan)
                if isinstance(remaining_waypoint_plan, list)
                else None
            ),
            "executed_steps": int(execution_horizon),
            "previous_failure_feedback": (
                copy.deepcopy(failure_feedback) if isinstance(failure_feedback, dict) else None
            ),
        }

    @staticmethod
    def _is_llmscenario_prompt_version(prompt_version: str) -> bool:
        return str(prompt_version or "").strip().lower() in {
            "llmscenario",
            "llmscenario_trajrewrite",
            "llmscenario_trajectory_rewrite",
            "trajectory_rewrite",
        }

    def _use_llmscenario_style_stage1_prompt(self) -> bool:
        return self._is_llmscenario_prompt_version(getattr(self, "prompt_version", ""))

    def _build_llmscenario_style_stage1_prompt(self, context: Dict) -> Tuple[str, str]:
        execution_horizon = int(context.get("execution_horizon", self.executed_steps_per_call))
        waypoint_interval_steps = int(
            context.get("waypoint_interval_steps", self.waypoint_interval_steps)
        )
        waypoint_steps = [
            int(step)
            for step in context.get("waypoint_steps", self._compute_waypoint_steps())
        ]
        send_route_information = bool(self.send_route_information)
        payload = {
            "planning_contract": {
                "baseline": "LLMScenario-style trajectory rewriting",
                "trajectory_horizon": int(context.get("trajectory_horizon", self.horizon)),
                "execution_horizon": int(execution_horizon),
                "waypoint_interval_steps": int(waypoint_interval_steps),
                "allowed_waypoint_steps": waypoint_steps,
                "step_dt_s": float(context.get("step_dt_s", 0.1)),
                "assessor_enabled": False,
                "local_reality_feedback_max_replans": int(self.llmscenario_local_reality_max_replans),
                "contract_note": (
                    "A rule/controller-like nominal ADV trajectory is provided. "
                    "Rewrite only the sparse future points needed to satisfy the scenario goal. "
                    "If previous local reality feedback is provided, revise the trajectory once using that feedback."
                ),
            },
            "scenario_goal": context.get("scenario_goal"),
            "adv_current_state": context.get("adv_current_state"),
            "ego_current_state": context.get("ego_current_state"),
            "ego_adv_relative_state": context.get("ego_adv_relative_state"),
            "ego_planned_trajectory_steps": context.get("ego_planned_trajectory_steps"),
            "ego_planned_trajectory_world": context.get("ego_planned_trajectory_world"),
            "adv_nominal_trajectory": context.get("adv_nominal_trajectory"),
            "ego_adv_interaction_analysis": context.get("ego_adv_interaction_analysis"),
            "map_information": context.get("map_information"),
            "route_information": context.get("route_information") if send_route_information else {},
            "nearby_agents": context.get("nearby_agents"),
            "previous_feedback_from_execution": context.get("previous_failure_feedback"),
        }

        system_prompt = (
            "You are implementing an LLMScenario-style driving scenario generator.\n"
            "You control only the adversarial vehicle (ADV). EGO is controlled by its planner.\n"
            "You are given a nominal ADV trajectory from a simple rule/controller planner. "
            "Your job is to rewrite that trajectory into sparse world waypoints that better satisfy the scenario goal, "
            "while keeping the motion plausible and close to normal driving unless the goal requires a risky interaction.\n"
            "This baseline does not use a separate plan assessor. It may receive local evaluation feedback "
            "from a previous failed generation attempt and should revise the rewritten trajectory accordingly. "
            "Output must be valid JSON only."
        )
        user_prompt = (
            "Follow the LLMScenario-style workflow: task awareness, object/interaction recognition, "
            "risk reasoning, then trajectory rewriting.\n\n"
            "Rules:\n"
            "1. Start from adv_nominal_trajectory. Prefer minimal, semantically meaningful edits instead of planning from scratch.\n"
            "2. Rewrite the nominal trajectory to make ADV follow the scenario_goal. If the goal requires cut-in, blocking, overtaking, braking, or head-on intrusion, the waypoints must show that interaction explicitly.\n"
            "3. If previous_feedback_from_execution is non-empty, treat it as LLMScenario evaluation feedback and fix the cited realism issue before outputting the revised trajectory.\n"
            "4. The output waypoint steps must be exactly selected from allowed_waypoint_steps, strictly increasing, and the first step must equal waypoint_interval_steps.\n"
            f"5. The final waypoint step must be >= {execution_horizon}.\n"
            "6. Every waypoint must contain step, stage_id, and position_world. Use stage_id=1 if a single-stage rewrite is enough.\n"
            "7. The rewritten waypoints should remain vehicle-like: no teleporting, reverse jumps, impossible turns, or unnecessary off-road motion.\n"
            "8. If the scenario goal is not achievable from the nominal trajectory, make the strongest plausible progress toward it, then continue normal driving.\n"
            "9. If a BEV image is attached: red is ADV, blue is EGO, and yellow is EGO's planned trajectory. The image is only geometry context; use the world coordinates for output.\n"
            "10. Output JSON only, no Markdown.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "scenario_goal_restatement": "...",\n'
            '  "llmscenario_cot_summary": {\n'
            '    "task_awareness": "...",\n'
            '    "interaction_recognition": "...",\n'
            '    "risk_reasoning": "..."\n'
            "  },\n"
            '  "stage_plan": [\n'
            '    {"stage_id": 1, "stage_name": "rewrite_nominal_trajectory", "goal_description": "...", "completion_condition": "..."}\n'
            "  ],\n"
            '  "waypoint_plan": [\n'
            '    {"step": 5, "stage_id": 1, "position_world": [x, y]},\n'
            '    {"step": 10, "stage_id": 1, "position_world": [x, y]}\n'
            "  ],\n"
            '  "plan_analysis": {\n'
            '    "nominal_changes_summary": "...",\n'
            '    "trajectory_action_summary": "...",\n'
            '    "collision_expectation": {\n'
            '      "will_collide_with_ego": false,\n'
            '      "collision_location": "...",\n'
            '      "collision_reason": "...",\n'
            '      "why_match_scenario_goal": "..."\n'
            "    }\n"
            "  }\n"
            "}\n\n"
            "Scene input:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        return system_prompt, user_prompt

    def _use_baselinec_style_stage1_prompt(self) -> bool:
        prompt_version = str(getattr(self, "prompt_version", "v1") or "v1").strip().lower()
        return prompt_version in {
            "baselinec",
            "baselinec_waypoint",
            "baselinec_anchor",
            "stage1_anchor",
            "anchor_waypoint",
        }

    def _build_baselinec_style_stage1_prompt(self, context: Dict) -> Tuple[str, str]:
        execution_horizon = int(context.get("execution_horizon", self.executed_steps_per_call))
        waypoint_interval_steps = int(
            context.get("waypoint_interval_steps", self.waypoint_interval_steps)
        )
        waypoint_steps = [
            int(step)
            for step in context.get("waypoint_steps", self._compute_waypoint_steps())
        ]
        step_dt_s = float(context.get("step_dt_s", 0.1))
        send_route_information = bool(self.send_route_information)
        payload = {
            "planning_contract": {
                "prompt_version": str(getattr(self, "prompt_version", "baselinec_waypoint")),
                "trajectory_horizon": int(context.get("trajectory_horizon", self.horizon)),
                "execution_horizon": int(execution_horizon),
                "waypoint_interval_steps": int(waypoint_interval_steps),
                "allowed_waypoint_steps": waypoint_steps,
                "step_dt_s": float(step_dt_s),
                "send_route_information": bool(send_route_information),
                "send_bev_to_llm": bool(context.get("send_bev_to_llm", self.send_bev_to_llm)),
                "bev_attached": bool(context.get("bev_attached", False)),
                "bev_status": context.get("bev_status"),
                "output_note": (
                    "adv_anchor_plan is Stage1's sparse waypoint plan; the system interpolates these anchors into a dense rollout and hands it to the plan assessor for checking."
                ),
            },
            "scenario_goal": context.get("scenario_goal"),
            "adv_current_state": context.get("adv_current_state"),
            "ego_current_state": context.get("ego_current_state"),
            "ego_adv_relative_state": context.get("ego_adv_relative_state"),
            "ego_planned_trajectory_steps": context.get("ego_planned_trajectory_steps"),
            "ego_planned_trajectory_world": context.get("ego_planned_trajectory_world"),
            "ego_adv_interaction_analysis": context.get("ego_adv_interaction_analysis"),
            "map_information": context.get("map_information"),
            "route_information": context.get("route_information") if send_route_information else {},
            "nearby_agents": context.get("nearby_agents"),
            "reachable_target_constraint": context.get("reachable_target_constraint"),
            "remaining_anchor_plan": context.get("remaining_waypoint_plan"),
            "previous_failure_feedback": context.get("previous_failure_feedback"),
        }

        system_prompt = (
            "You are the ADV sparse waypoint/anchor planner in a closed-loop adversarial scenario.\nYou only control ADV; EGO is controlled by the planner, and background vehicles move along fixed trajectories.\nYour task is not to output a per-frame trajectory, nor to write a complex stage plan, but to output a small number of sparse waypoint anchors so that the local Stage1 rollout and subsequent diffusion can generate trajectories along these geometric constraints.\nThe goal is to advance the high-risk interaction between ADV and EGO around scenario_goal while keeping vehicle motion natural, reachable, and continuous.\nThe output must be JSON, and only contain scenario_goal_restatement, anchor_plan_reason, adv_anchor_plan, plan_analysis."
        )
        user_prompt = (
            f"""Based on the information below, generate a set of Stage1 sparse waypoint anchors for ADV.\nCore requirements:\n1. First understand the action semantics of scenario_goal, then plan waypoints; do not mechanically chase a single future point of EGO.\n2. adv_anchor_plan is this round's Stage1 waypoint plan. Each anchor must contain step, position_world, reason.\n3. step must be strictly increasing, and can only use allowed_waypoint_steps.\n4. Allowed steps can only come from: {waypoint_steps}.\n5. The step of the last anchor must be >= {execution_horizon}.\n6. The number of anchors should cover key geometric stages, such as start, turning, cut-in, conflict point, post-collision continuation, or normal driving; do not provide only one endpoint.\n7. position_world must be world coordinates [x, y], in meters.\n8. If remaining_anchor_plan is non-empty, it means there are still unexecuted anchors from the previous round. You should prioritize correcting and continuing based on it; if it conflicts with the current state or scenario_goal, you may adjust it.\n9. If previous_failure_feedback is non-empty, you must incorporate the assessor's feedback and avoid repeating the same type of error.\n10. Route information is a reference for road direction, not an absolute hard constraint. If route is missing or does not have the direction required by scenario_goal, do not give up the goal; continue planning based on the map, vehicle yaw/speed, and EGO future trajectory.\n11. Each step in this round = {step_dt_s:g}s. The first segment must respect ADV's current yaw and speed; when starting from rest or at low speed, the first anchor should not be too far, and must not be placed behind or to the side-rear of the vehicle or in the opposite direction.\n12. Self-check by reachability: for each anchor segment, |accel| should as much as possible not exceed {self.reachability_max_abs_accel_mps2:g} m/s^2, |yaw_rate| should not exceed {self.reachability_max_abs_yaw_rate_rps:g} rad/s; between adjacent anchor segments there must be no sudden speed change, lateral jump, hard stop, reverse interleaving, or sudden U-turn.\n13. Keep the same scale as the Stage1 local check: locally, acceleration/yaw_rate are computed by sparse anchor segment, and the segment-level |accel| must not exceed {self.validation_max_abs_accel_mps2:g} m/s^2, segment-level |yaw_rate| should not exceed {self.validation_max_abs_yaw_rate_rps:g} rad/s.\n14. When judging collisions, you must look at the spatiotemporal relationship: if you think step=t will collide, ADV near t should form a box overlap or be very close to the EGO's position at t.\n15. If a collision cannot be achieved in the current round or a subsequent reasonable implementation, or if the EGO has clearly avoided successfully, then drive normally along a reasonable route; do not use trajectories that violate dynamics to compensate.\n16. If this round includes a BEV image: the red ADV is the controlled vehicle, the blue EGO is the attacked vehicle, and the yellow line is the EGO plan. The ADV's future trajectory is not drawn during the planning stage.\n17. Output only JSON; do not output extra explanation, Markdown, or code blocks.\n\nOutput format:\n{{\n  "scenario_goal_restatement": "...",\n  "anchor_plan_reason": "...",\n  "adv_anchor_plan": [\n    {{"step": 5, "position_world": [x, y], "reason": "..."}},\n    {{"step": 10, "position_world": [x, y], "reason": "..."}}\n  ],\n  "plan_analysis": {{\n    "trajectory_action_summary": "...",\n    "collision_expectation": {{\n      "will_collide_with_ego": false,\n      "collision_location": "...",\n      "collision_reason": "...",\n      "why_match_scenario_goal": "..."\n    }}\n  }}\n}}\n\nScenario input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"""
        )
        return system_prompt, user_prompt

    def _build_prompt(self, context: Dict) -> Tuple[str, str]:
        if self._use_llmscenario_style_stage1_prompt():
            return self._build_llmscenario_style_stage1_prompt(context)
        if self._use_baselinec_style_stage1_prompt():
            return self._build_baselinec_style_stage1_prompt(context)

        horizon = int(context.get("trajectory_horizon", self.horizon))
        execution_horizon = int(context.get("execution_horizon", self.executed_steps_per_call))
        waypoint_horizon_steps = int(
            context.get("waypoint_horizon_steps", self.waypoint_horizon_steps)
        )
        waypoint_interval_steps = int(
            context.get("waypoint_interval_steps", self.waypoint_interval_steps)
        )
        waypoint_steps = [
            int(step)
            for step in context.get("waypoint_steps", self._compute_waypoint_steps())
        ]
        step_dt_s = float(context.get("step_dt_s", 0.1))
        route_num_points = int(self.route_num_points)
        route_spacing_m = float(self.route_spacing_m)
        send_route_information = bool(self.send_route_information)
        section_task = {
            "scenario_goal": context.get("scenario_goal", "adv collide with ego"),
        }
        section_map = {
            "candidate_lanes": context.get("map_information", {}).get("candidate_lanes", []),
        }
        raw_section_route = context.get("route_information", {}) if send_route_information else {}
        section_route = {}
        if send_route_information and isinstance(raw_section_route, dict):
            for actor_key in ("adv_route", "ego_route"):
                actor_route = raw_section_route.get(actor_key, {})
                if not isinstance(actor_route, dict):
                    continue
                current_routes: List[Dict[str, Any]] = []
                seen_route_signatures = set()
                reference_route_world = actor_route.get("reference_route_world", [])
                reference_route_yaw = actor_route.get("reference_route_yaw", [])
                route_sources: List[Tuple[List[Any], List[Any]]] = []
                if reference_route_world:
                    route_sources.append((reference_route_world, reference_route_yaw))
                for route_item in actor_route.get("candidate_routes", []):
                    if not isinstance(route_item, dict):
                        continue
                    route_world = route_item.get("reference_route_world", [])
                    if not route_world:
                        continue
                    route_sources.append(
                        {
                            "route_world": route_world,
                            "route_yaw": route_item.get("reference_route_yaw", []),
                        }
                    )
                normalized_sources: List[Dict[str, Any]] = []
                for item in route_sources:
                    if isinstance(item, tuple):
                        route_world, route_yaw = item
                        normalized_sources.append(
                            {
                                "route_world": route_world,
                                "route_yaw": route_yaw,
                            }
                        )
                    elif isinstance(item, dict):
                        normalized_sources.append(item)
                for route_item in normalized_sources:
                    route_world = route_item.get("route_world", [])
                    route_yaw = route_item.get("route_yaw", [])
                    signature = (
                        tuple(tuple(round(float(coord), 3) for coord in point) for point in route_world),
                        tuple(round(float(yaw), 4) for yaw in route_yaw),
                    )
                    if signature in seen_route_signatures:
                        continue
                    seen_route_signatures.add(signature)
                    current_routes.append(
                        {
                            "route_id": len(current_routes) + 1,
                            "route_world": route_world,
                            "route_yaw": route_yaw,
                        }
                    )
                section_route[actor_key] = {
                    "track_id": actor_route.get("track_id"),
                    "current_lane_id": actor_route.get("current_lane_id"),
                    "current_routes": current_routes,
                }
        map_info_description = self._map_info_description()
        section_vehicle = {
            "adv_current_state": context.get("adv_current_state", {}),
            "ego_current_state": context.get("ego_current_state", {}),
            "ego_adv_relative_state": context.get("ego_adv_relative_state", {}),
            "ego_planned_trajectory_steps": context.get("ego_planned_trajectory_steps", []),
            "ego_planned_trajectory_world": context.get("ego_planned_trajectory_world", []),
            "nearby_agents": context.get("nearby_agents", []),
        }
        section_interaction_text = str(context.get("ego_adv_interaction_analysis", "")).strip()
        section_stage_plan = {
            "stage_plan": context.get("stage_plan", None),
        }
        section_remaining_waypoint_plan = {
            "remaining_waypoint_plan": context.get("remaining_waypoint_plan", None),
            "executed_steps": context.get("executed_steps", execution_horizon),
        }
        section_failure_feedback = {
            "previous_failure_feedback": context.get("previous_failure_feedback", None),
        }
        nearby_agents_desc = (
            "- nearby_agents: Under the current configuration, information about nearby vehicles within the ADV radius is sent.\n"
            if self.send_nearby_agents
            else "- nearby_agents: Under the current configuration, the content of this field is not sent, so this array will be empty.\n"
        )
        nearby_future_traj_desc = (
            "- future_trajectory_world: the future trajectory position of a nearby agent.\n"
            if self.send_nearby_future_trajectory
            else "- future_trajectory_world: Under the current configuration, this field is not sent.\n"
        )
        route_information_block = (
            "route information\n"
            f"{json.dumps(section_route, ensure_ascii=False, indent=2)}\n\n"
            if send_route_information
            else "route information\n- Under the current configuration, route information is not sent, so this part is empty.\n\n"
        )
        route_field_meanings = (
            f"- route information: This provides the currently referenceable route information, used to help you judge the general road directions along which the vehicle can advance.\n- current_routes: A list of several currently referenceable routes; do not assume there is a primary/secondary relationship among them, they are all just currently available route clues.\n- route_world: Local world-coordinate sample points of a current route; starting near the vehicle's current position, along the route's forward direction at intervals of about {route_spacing_m:g} meters per sample, total {route_num_points} points.\n- route_yaw: the tangent yaw corresponding to each sample point in route_world, in radians; it can be regarded as the direction change information of this route.\n- route_id: the identifier of a current route.\n- route usage note: the role of route is to help you judge the road direction and a reasonable direction of travel. You may choose any one or several routes from current_routes that best match the current action as reference. Route is not an absolute hard constraint; you may deviate from route as long as you do not deviate from scenario_goal. Route may sometimes also be missing, for example, scenario_goal clearly requires a left-turn attack, but the current route information does not provide a suitable left-turn route; in this case, do not give up scenario_goal because route is missing, but continue planning according to the interaction mode required by scenario_goal.\n"
            if send_route_information
            else "- route information: Under the current configuration, route information is not sent, so you cannot rely on the current route; in this case, you can only judge a reasonable direction of travel yourself by combining the map, vehicle state, EGO future trajectory, and interaction relationships.\n"
        )
        route_thinking_item = (
            "4. When planning waypoints, you should refer to a reasonable current route. You should first judge from scenario_goal whether you are more suitable for going straight, turning left, turning right, merging, or cutting in, and then choose from the currently given routes the direction most consistent with that action to proceed; you may also refer to EGO's current route.\n5. When judging the turning trend of a current route, combine it with yaw changes: a left-turn route's yaw generally tends to increase, and a right-turn route's yaw generally tends to decrease. Usually, waypoints should roughly follow the road geometry of the current route you selected; but if deviating from route is necessary to achieve scenario_goal, you may also deviate reasonably. If you find that the current route information appears to be missing and cannot provide a route clearly matching scenario_goal, do not be mechanically limited by route either, but continue planning according to scenario_goal.\n"
            if send_route_information
            else "4. No route information is currently sent, so you should mainly judge how to advance this round based on map geometry, the vehicle's current yaw/speed, the EGO's future trajectory, and scenario_goal, rather than assuming that some directly referenceable route exists.\n5. Even without route information, you should still judge based on the heading change trend of the waypoints themselves whether this trajectory is closer to going straight, turning left, turning right, merging, or cutting in.\n"
        )
        route_output_requirements = (
            "1. route_direction_summary: combined with the currently given route information, explain which direction or directions of the current route the ADV generally chose to advance along this time. If you judge that the route information is missing, that is, the route you want to take does not exist, just give an explanation, for example, according to the scenario goal you currently need to act along a left-turn route, but that route information is not given.\n"
            if send_route_information
            else "1. route_direction_summary: route information is currently not sent; explicitly state \"no route information was provided this round\" and explain which map geometry, vehicle state, or EGO trajectory cues you mainly relied on to determine the direction of travel.\n"
        )
        route_caution_note = (
            "8. route is an important reference to help you judge the road direction and a reasonable direction of travel, not an absolute hard constraint. All routes currently given are only the currently referenceable routes; you should generally try to make the waypoints roughly consistent with the geometric direction of the current route you selected, but if deviating from the route is necessary to achieve the scenario_goal, you may deviate reasonably.\n"
            if send_route_information
            else "8. Route information is currently not sent, so do not assume there is a current route that can be directly referenced; you should mainly rely on map geometry, vehicle states, and interaction relationships to ensure the trajectory is reasonable.\n"
        )
        output_example = {
            "scenario_goal_restatement": "...",
            "stage_plan": [
                {
                    "stage_id": 1,
                    "stage_name": "stage_name_here",
                    "goal_description": "...",
                    "completion_condition": "...",
                }
            ],
            "waypoint_plan": [
                {"step": 5, "stage_id": 1, "position_world": [0.0, 0.0]},
                {"step": 10, "stage_id": 1, "position_world": [0.0, 0.0]},
                {"step": 15, "stage_id": 1, "position_world": [0.0, 0.0]},
                {"step": 20, "stage_id": 2, "position_world": [0.0, 0.0]},
            ],
            "segment_dynamics": [
                {
                    "from_step": 0,
                    "to_step": 5,
                    "estimated_heading_rad": 0.0,
                    "estimated_yaw_rate_rps": 0.1,
                    "estimated_accel_mps2": 1.2,
                    "estimated_end_speed_mps": 3.5,
                }
            ],
            "plan_analysis": {
                "route_direction_summary": "...",
                "waypoint_yaw_analysis": "...",
                "trajectory_action_summary": "...",
                "collision_expectation": {
                    "will_collide_with_ego": True,
                    "collision_location": "...",
                    "collision_reason": "...",
                    "why_match_scenario_goal": "...",
                },
            },
        }
        update_example = {
            "scenario_goal_restatement": "...",
            "adjustment_reason": "...",
            "updated_waypoint_plan": [
                {"step": 5, "stage_id": 2, "position_world": [0.0, 0.0]},
                {"step": 10, "stage_id": 2, "position_world": [0.0, 0.0]},
                {"step": 15, "stage_id": 2, "position_world": [0.0, 0.0]},
                {"step": 20, "stage_id": 3, "position_world": [0.0, 0.0]},
            ],
            "segment_dynamics": [
                {
                    "from_step": 0,
                    "to_step": 5,
                    "estimated_heading_rad": 0.0,
                    "estimated_yaw_rate_rps": 0.05,
                    "estimated_accel_mps2": 0.8,
                    "estimated_end_speed_mps": 3.0,
                }
            ],
            "plan_analysis": {
                "route_direction_summary": "...",
                "waypoint_yaw_analysis": "...",
                "trajectory_action_summary": "...",
                "collision_expectation": {
                    "will_collide_with_ego": False,
                    "collision_location": "...",
                    "collision_reason": "...",
                    "why_match_scenario_goal": "...",
                },
            },
        }

        system_prompt = ""

        user_prompt = (
            f"""【System Role】\nYou are an expert in dangerous scenario generation and closed-loop simulation planning.\nYou currently control only one ADV. The EGO is controlled by the planner, and background vehicles move along fixed trajectories.\n\n【Current Scenario Introduction】\nThe scenario information required for this round of planning is given below, including the map, route, vehicle states, interaction analysis between the EGO and ADV, the previous round's stage plan, the remaining waypoint plan, and failure feedback.\n\nmap information\n- {map_info_description}\n{json.dumps(section_map, ensure_ascii=False, indent=2)}\n\n{route_information_block}vehicle information\n{json.dumps(section_vehicle, ensure_ascii=False, indent=2)}\n\nego adv interaction analysis\n{section_interaction_text}\n\nstage plan from previous round\n{json.dumps(section_stage_plan, ensure_ascii=False, indent=2)}\n\nremaining waypoint plan\n{json.dumps(section_remaining_waypoint_plan, ensure_ascii=False, indent=2)}\n\nprevious failure feedback\n{json.dumps(section_failure_feedback, ensure_ascii=False, indent=2)}\n\nfield meanings\n- All *_world fields use the original world coordinate system, in meters; +x is to the right, +y is up.\n- step_dt_s: duration of each step, in seconds; currently 1 step = {step_dt_s:g} s each round.\n- yaw_world: the vehicle's heading angle in the world coordinate system, in radians, range [-pi, pi). yaw_world = 0 means facing the +x direction; increasing yaw_world means turning left, decreasing means turning right.\n- speed_mps: speed, in m/s.\n- velocity_world: velocity vector [vx, vy], in m/s.\n- extent_m: vehicle dimensions [length, width], in meters.\n- current_lane_id: the currently associated lane id.\n- ego_adv_relative_state: the position and velocity relationship of ADV relative to EGO, for auxiliary analysis only; your core task is still to make ADV act as described by scenario_goal.\n- centerline_world: lane centerline sample points, sorted in the lane's forward direction.\n- centerline_yaw: the tangent yaw corresponding to the centerline sample points.\n- lane_width_m: lane width, in meters.\n{route_field_meanings}- ego_planned_trajectory_steps: the future step corresponding to each point in ego_planned_trajectory_world; currently according to llm_waypoint_interval_steps = {waypoint_interval_steps} sampling, therefore usually {waypoint_steps}.\n- ego_planned_trajectory_world: EGO in the future {horizon} step sparse planning trajectory. It is used to help you understand the EGO's future motion trend, but your task is not to mechanically chase these points, but to make the ADV advance the interaction according to scenario_goal.\n- If a BEV image is attached this round: the red ADV indicates the vehicle you currently control, the blue EGO indicates the attacked vehicle, and the yellow EGO plan indicates the EGO's future planned trajectory. The planning stage will not mark the ADV's to-be-planned trajectory on the image, because that is exactly what you need to output.\n{nearby_agents_desc}{nearby_future_traj_desc}- waypoint_plan / updated_waypoint_plan: sparse waypoint plan starting from the current moment. step indicates the number of future steps relative to the current moment and must be strictly increasing; the step interval between adjacent waypoints is fixed at {waypoint_interval_steps}; the last step must be >= {execution_horizon}.\n- waypoint: each waypoint must contain step, stage_id, position_world, where position_world = [x, y].\n- segment_dynamics: after outputting waypoint_plan / updated_waypoint_plan, you must also output a segment_dynamics list, giving an approximate dynamics estimate for each segment from the current state to the first waypoint and between adjacent waypoints. Each segment must contain at least from_step, to_step, estimated_heading_rad, estimated_yaw_rate_rps, estimated_accel_mps2, estimated_end_speed_mps. \n- segment_dynamics is only an approximate estimate based on sparse waypoints, used to self-check whether the trajectory is roughly reasonable; it is not required to be completely consistent with the underlying interpolated trajectory.\n- Before local execution, dynamics statistics are computed based on sparse waypoint segments, rather than using frame-level spikes of the dense interpolated trajectory as the main criterion: segment-level |accel| exceeding {self.validation_max_abs_accel_mps2:g} m/s^2 or segment-level |yaw_rate| exceeds {self.validation_max_abs_yaw_rate_rps:g} rad/s will trigger local correction. The frame-level accel/yaw_rate of the dense rollout are only for diagnostic reference. Your primary goal is still to promote the EGO-ADV collision in scenario_goal, but you cannot create a collision by suddenly changing speed or suddenly twisting the head between segments.\n- When planning waypoints, please prioritize self-checking with a more conservative reachability estimate: |accel| should as much as possible not exceed {self.reachability_max_abs_accel_mps2:g} m/s^2, |yaw_rate| should not exceed {self.reachability_max_abs_yaw_rate_rps:g} rad/s. In particular, when starting from a standstill or at low speed, the first waypoint of step=5 can usually only be very close to the current position, and the heading of the first segment must be close to current_yaw_world; you must not place the first segment to the side-rear of the vehicle or in the opposite direction just to chase EGO.\n- stage_plan: list of stage plans. Each stage must include stage_id, stage_name, goal_description, completion_condition.\n- remaining_waypoint_plan: the waypoint queue from the previous round that has not yet been fully executed and can continue to be corrected and extended in this round; step has been realigned to the current moment.\n- previous_failure_feedback: the reason for the previous round's failure or the key correction points; if provided, you should absorb it first to avoid repeating the same type of problem.\n\n[Task Introduction]\nYour overall goal is: make ADV collide with EGO in the manner described by scenario_goal. The collision determination depends on whether the two vehicles' two-dimensional rotated boxes overlap.\nEach round, the system executes only the first {execution_horizon} steps, so this is rolling closed-loop planning, not one-time open-loop planning. You do not necessarily complete all attacks in a single round.\nBefore starting formal planning, you must first restate scenario_goal in your thinking, confirm what target interaction relationship you understand, and then continue outputting stage divisions, waypoints, segment_dynamics, and analysis.\n\nDuring the first round of planning:\n1. First decompose scenario_goal into several stages, and each stage must include stage_id, stage_name, goal_description, completion_condition.\n2. Then output waypoint_plan, and annotate each waypoint with its stage_id.\n3. The last step of waypoint_plan must be >= {execution_horizon}; if the current stage is insufficient to cover the previous {execution_horizon} steps, continue appending subsequent stages.\n4. The step interval between adjacent waypoints must be fixed at {waypoint_interval_steps}.\n5. Once you start outputting a stage, you must complete that stage fully, rather than only filling in just enough for the current execution scope.\n\nFor subsequent corrections:\n1. You will see the remaining_waypoint_plan from the previous round that has not yet been fully executed.\n2. Your task is not to start over, but to prioritize correction and continuation based on remaining_waypoint_plan.\n3. You may add or remove waypoints in it to better match the current state and scenario_goal.\n4. The last step of updated_waypoint_plan must also be >= {execution_horizon}; if the corrected remaining plan is not long enough, continue appending the next stage.\n5. The step interval between adjacent waypoints must still be fixed at {waypoint_interval_steps}.\n6. Once you start appending a new stage, you must complete that stage in full.\n\nRegarding remaining_waypoint_plan, be sure to understand it as follows:\n1. It represents the remaining plan from the previous round that has not yet been fully executed and needs to be continued in this round.\n2. If it is non-empty, then the last waypoint in it represents the endpoint of the last stage covered by the previous round's planning.\n3. Therefore, there is no situation where "the previous round only gave half of a stage, and this round needs to continue completing that stage."\n4. Your correction to remaining_waypoint_plan is mainly to make the existing plan more reasonable, not to endlessly continue writing the current last stage.\n5. If the corrected remaining_waypoint_plan is still insufficient to cover the previous {execution_horizon} step, you should append the next stage rather than always staying in the current stage.\n6. If remaining_waypoint_plan is empty, it means there is currently no remaining waypoint that can be directly continued, and you should continue planning from the next stage.\n\n[Thinking Method]\nBefore planning, please think clearly about the following questions:\n1. The stages you divide may be multiple stages or a single stage; overall, it is enough to conform to the scenario goal.\n2. What kind of interaction relationship does scenario_goal ultimately require to be formed, and which stage of achieving that goal is currently in progress.\n3. In this round, should remaining_waypoint_plan be prioritized for correction, or should the next stage be entered; how should the waypoint push the current stage forward, or how should it naturally transition to the next stage.\n4. What counts as a collision: this requires the waypoint and ego_planned_trajectory_world to have spatial interaction with EGO at some time. For example, if EGO is at point A at t=15, then if the ADV waypoint you give is also at point A at t=15 or within 0.5 meters around it, it can be regarded as the two vehicles having collided. Therefore, for the trajectory you give, if you think a collision should occur at a certain moment, then at that moment the positions of EGO and ADV are approximately the same.\n5. When giving waypoints, consider their temporal relationship with EGO's position, for example, a waypoint at t=5 corresponds to EGO's position also at t=5.\n{route_thinking_item}6. Before outputting the final result, you must first perform a segment_dynamics self-check based on waypoint_plan / updated_waypoint_plan. You need to estimate the heading, yaw rate, acceleration, and terminal speed for each segment from the current state to the first waypoint, and between adjacent waypoints.\n7. During the self-check, pay attention to trajectory realism: locally, acceleration/yaw_rate will be computed from sparse waypoint segments; if the absolute value of estimated_yaw_rate_rps for a segment is greater than {self.validation_max_abs_yaw_rate_rps:g}, indicating a rather sharp turn; if the absolute value of estimated_accel_mps2 for a certain segment is greater than {self.validation_max_abs_accel_mps2:g}, indicating that acceleration and deceleration are rather abrupt. You should try to make the waypoints smoother, but do not give up the scenario_goal due to being overly conservative; as long as the trajectory can advance the collision goal and there is no teleportation/non-finite value/extreme jump, it can be output.\n8. The estimated_yaw_rate_rps of the first segment should use the change from current_yaw_world to the first segment's heading divided by the duration of that segment; subsequent segments should use the change between adjacent segment headings divided by the duration of that segment. estimated_accel_mps2 can be approximated by constant acceleration, estimated using the segment's displacement length, starting speed, and segment duration.\n\nDo not just stare at a single point in EGO's future trajectory and chase it. You should first determine the current stage's task, then make waypoint_plan / updated_waypoint_plan serve the advancement of that stage.\n\n[Notes]\n1. The route may sometimes be missing. For example, the scenario goal requires a left turn and then a collision, but you find that no left-turn route exists. In that case, the route may be missing, and you should still act according to the requirements of the scenario goal.\n2. Without deviating from the scenario_goal, try to ensure that the waypoint plan is kinematically reasonable, geometrically continuous, and consistent with the motion characteristics of a real vehicle.\n3. Waypoints must be planned in combination with ADV's current yaw, speed, and executable time range, and must not seriously contradict the current heading or dynamic state. For example, if the vehicle is moving forward, then the next waypoint appearing behind the current position is absolutely not allowed.\n4. If the vehicle continues to accelerate, the spacing between adjacent waypoints should usually gradually increase; if it continues to decelerate, it should usually gradually decrease.\n5. The given waypoint spacing should usually roughly match the vehicle's current speed. For example, when the vehicle is starting from rest or at low speed, the given waypoints should reflect the acceleration process, gradually changing from short to long, and you should not directly give waypoints with excessively large spacing. In short, the given waypoints should conform to the vehicle's dynamic constraints and must not be set arbitrarily detached from the current motion state.\n6. The waypoint plan must reflect an interaction process consistent with the scenario_goal, rather than merely being geometrically close to EGO.\n6.1 The key action semantics in the scenario_goal must not be executed in reverse: for example, if a left turn is required, it must not be planned as a right turn or a lateral jump; if cutting in from the right is required, it must not detour from the left; if starting and then turning left is required, it must not remain stationary waiting or directly teleport to the conflict point.\n7. The waypoint plan may be slightly aggressive in order to achieve the scenario_goal, but it must never create a collision by instantaneous U-turn, lateral jump, sudden stop, reversing insertion, or excessive acceleration; if it is dynamically impossible, reduce the attack intensity or switch to normal driving.\n8. If the scenario_goal requires starting from rest and turning left/right, the first stage should first establish speed along the current heading and the corresponding steering direction; do not directly place the waypoint near the target collision point. First perform the small start, turn, and acceleration that a real vehicle would do, then gradually approach the conflict area.\n9. When feasible, try to avoid collisions with irrelevant background vehicles.\n10. After the collision occurs, the waypoints should roughly follow the route for normal driving, and you should not continue forcibly chasing EGO.\n11. Although your goal is to achieve the scenario_goal, during the closed loop EGO may evade in time according to your attack behavior, so the collision may not actually occur. Your core task is to make ADV advance the attack in the manner described by the scenario_goal; if it truly cannot collide, the collision opportunity has already been missed, or EGO has successfully evaded and a collision is basically no longer going to occur afterward, then continue driving according to a reasonable route, and do not output remedial trajectories that violate dynamics.\n{route_caution_note}\nrelation semantics\n- left / right / front / rear: Unless otherwise specified, "left / right / front / rear" are by default defined relative to the vehicle's own current forward direction, rather than judged simply by x / y magnitude in world coordinates.\n- same direction / opposite direction: "same direction" means the two vehicles' yaw_world are relatively close and their overall motion trends are basically consistent; "opposite direction" means the difference between the two vehicles' yaw_world is close to pi and their overall motion trends are generally opposite.\n- directly ahead / directly behind: "A is directly ahead of B" means that A is not only located in the front half-plane corresponding to B's current heading, but is also closer to being on the forward extension path along which B would continue moving at its current yaw_world; "A is directly behind B" is analogous.\n- following relation: "same-direction front-rear vehicle relation" means the two vehicles' yaw_world are relatively close and their overall motion directions are basically consistent, while one vehicle is closer to being directly ahead of the other, and the other vehicle is closer to being directly behind the front vehicle.\n- clarification: If the two vehicles' yaw_world are relatively close, or the difference is close to pi, but there is no obvious directly-ahead / directly-behind relation, then they are more likely driving side by side, and a collision usually will not occur.\n\nHow to determine whether a waypoint trajectory is turning left or right:\nDo not look only at whether the endpoint is on the left or right; instead look at how the heading of this trajectory changes step by step with the waypoints.\nFor adjacent waypoints, compute the heading of each segment in order: theta_i = atan2(dy, dx).\nThen compute the change in heading between adjacent segments: d_i = wrap_to_pi(theta_{{i+1}} - theta_i).\nIf these changes are overall predominantly negative, it is closer to a right turn; if overall predominantly positive, it is closer to a left turn; if overall close to 0, it is closer to going straight.\nWhen judging left or right turn, focus on the overall heading change trend of the entire trajectory, not the position of a single waypoint.\n\nRegarding collision, please note:\nYour goal is not to verbally "approach" or "cut into" EGO, but to truly push ADV and EGO in the spatiotemporal relation into an interaction geometry where box overlap may occur.\n\n[Output requirements]\nYou must output only JSON, without explanations, markdown, or code blocks.\n- You must first output a top-level field scenario_goal_restatement, restating your understanding of scenario_goal in one concise natural-language sentence.\n- The output order must be: for the first round, scenario_goal_restatement -> stage_plan -> waypoint_plan -> segment_dynamics -> plan_analysis; for subsequent rounds, scenario_goal_restatement -> adjustment_reason -> updated_waypoint_plan -> segment_dynamics -> plan_analysis.\n- segment_dynamics is a required field, and it must be computed based on the preceding waypoint_plan / updated_waypoint_plan, and cannot be written independently of the trajectory.\n- You must additionally output a top-level field plan_analysis, which contains route_direction_summary, waypoint_yaw_analysis, trajectory_action_summary, collision_expectation.\n- collision_expectation must contain will_collide_with_ego, collision_location, collision_reason, why_match_scenario_goal.\n- If this is the first-round plan, the top-level fields retain scenario_goal_restatement, stage_plan, waypoint_plan, segment_dynamics, plan_analysis.\n- If this is a subsequent correction, the top-level fields retain scenario_goal_restatement, adjustment_reason, updated_waypoint_plan, segment_dynamics, plan_analysis.\n\nplan_analysis must contain at least the following content:\n{route_output_requirements}2. waypoint_yaw_analysis: must be written based on the preceding waypoint_plan / updated_waypoint_plan and segment_dynamics, analyze the trend of heading / yaw changes, and determine whether this trajectory is broadly going straight, turning left, turning right, merging, cutting in, occupying the lane, forcing a stop, cutting in diagonally, or intruding head-on, while explaining the basis.\n3. trajectory_action_summary: must be based on the preceding waypoints and segment_dynamics, using concise natural language to summarize the core action of this trajectory at the current stage and how it serves scenario_goal.\n4. collision_expectation: if you believe a collision with EGO will occur this round, explain roughly where the expected collision will occur, why it will collide, and why it conforms to scenario_goal; if you believe it will not collide this round, you must explicitly write will_collide_with_ego=false, and explain why it temporarily does not collide in the current round and which stage it is serving. This analysis must also be consistent with the preceding waypoints and segment_dynamics.\n\nFirst-round output format example\n{json.dumps(output_example, ensure_ascii=False, indent=2)}\n\nExample output format for subsequent rounds\n{json.dumps(update_example, ensure_ascii=False, indent=2)}"""
        )
        return system_prompt, user_prompt

    def _validate_world_points(
        self,
        world_points: torch.Tensor,
        ctrl_idx: int,
        obs: Dict,
        target_horizon: int,
        dynamics_validation_mode: str = "dense",
        enforce_limits: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        online_local_guard_disabled = bool(
            getattr(self, "_llmscenario_prompt_mode", False)
            and self.llmscenario_disable_online_local_guard
        )
        dynamics_validation_mode = str(dynamics_validation_mode or "dense").strip().lower()
        validate_dense_dynamics = dynamics_validation_mode == "dense"
        limits_enforced = bool(enforce_limits and not online_local_guard_disabled)
        current_world = obs["centroid"][ctrl_idx : ctrl_idx + 1]
        world_points = self._trim_leading_stationary_points(world_points, current_world)
        stats: Dict[str, Any] = {
            "validation_status": "not_run",
            "validation_failure_reason": None,
            "online_local_guard_disabled": online_local_guard_disabled,
            "validation_limits_enforced": bool(limits_enforced),
            "dense_dynamics_validation_mode": (
                "enforced" if validate_dense_dynamics else "diagnostic_only"
            ),
            "max_step_distance_limit_m": float(self.max_step_distance),
            "validation_max_speed_limit_mps": float(self.validation_max_speed_mps),
            "validation_max_abs_accel_limit_mps2": float(self.validation_max_abs_accel_mps2),
            "validation_max_abs_yaw_rate_limit_rps": float(self.validation_max_abs_yaw_rate_rps),
            "validation_reject_dynamics_exceeded": bool(self.validation_reject_dynamics_exceeded),
        }
        if world_points.shape[0] < 2:
            stats["validation_status"] = "failed"
            stats["validation_failure_reason"] = "insufficient_world_points"
            return None, None, stats

        if world_points.shape[0] < target_horizon:
            pad = world_points[-1:].repeat(target_horizon - world_points.shape[0], 1)
            world_points = torch.cat([world_points, pad], dim=0)
        world_points = world_points[:target_horizon]

        step_norm = torch.norm(torch.diff(torch.cat([current_world, world_points], dim=0), dim=0), dim=-1)
        if not torch.isfinite(world_points).all():
            stats["validation_status"] = "failed"
            stats["validation_failure_reason"] = "non_finite_world_points"
            return None, None, stats
        stats.update(
            {
                "first_step_distance_m": float(step_norm[0].item()) if step_norm.numel() > 0 else 0.0,
                "mean_step_distance_m": float(step_norm.mean().item()) if step_norm.numel() > 0 else 0.0,
                "max_step_distance_m": float(step_norm.max().item()) if step_norm.numel() > 0 else 0.0,
                "max_step_distance_exceeded": bool(torch.any(step_norm > self.max_step_distance).item()),
            }
        )
        if stats["max_step_distance_exceeded"] and limits_enforced:
            stats["validation_status"] = "failed"
            stats["validation_failure_reason"] = "max_step_distance_exceeded"
            return None, None, stats

        local_positions = self._to_local(world_points, obs["agent_from_world"][ctrl_idx])
        local_yaws = self._compute_local_yaws(
            world_points,
            obs["yaw"][ctrl_idx],
            min_heading_update_distance=self.min_heading_update_distance,
        ).unsqueeze(-1)
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        dt = max(float(dt), 1e-6)
        yaws_world = _wrap_to_pi(local_yaws.squeeze(-1) + obs["yaw"][ctrl_idx].to(self.device, dtype=torch.float32))
        prev_speed = torch.tensor(
            float(obs["curr_speed"][ctrl_idx].item()) if "curr_speed" in obs else 0.0,
            dtype=torch.float32,
            device=self.device,
        )
        speeds = step_norm / dt
        speed_for_accel = torch.cat([prev_speed.reshape(1), speeds], dim=0)
        accel = (speed_for_accel[1:] - speed_for_accel[:-1]) / dt
        yaw_ref = torch.cat([
            obs["yaw"][ctrl_idx].to(self.device, dtype=torch.float32).reshape(1),
            yaws_world,
        ])
        yaw_rate = _wrap_to_pi(yaw_ref[1:] - yaw_ref[:-1]) / dt
        max_speed = float(speeds.max().item()) if speeds.numel() > 0 else 0.0
        max_abs_accel = float(torch.abs(accel).max().item()) if accel.numel() > 0 else 0.0
        max_abs_yaw_rate = float(torch.abs(yaw_rate).max().item()) if yaw_rate.numel() > 0 else 0.0
        dense_accel_exceeded = bool(max_abs_accel > self.validation_max_abs_accel_mps2)
        dense_yaw_rate_exceeded = bool(max_abs_yaw_rate > self.validation_max_abs_yaw_rate_rps)
        stats.update(
            {
                "max_speed_mps": max_speed,
                "validation_speed_exceeded": bool(max_speed > self.validation_max_speed_mps),
                "dense_max_abs_accel_mps2": max_abs_accel,
                "dense_max_abs_yaw_rate_rps": max_abs_yaw_rate,
                "dense_validation_accel_exceeded": dense_accel_exceeded,
                "dense_validation_yaw_rate_exceeded": dense_yaw_rate_exceeded,
            }
        )
        if validate_dense_dynamics:
            stats.update(
                {
                    "max_abs_accel_mps2": max_abs_accel,
                    "max_abs_yaw_rate_rps": max_abs_yaw_rate,
                    "validation_accel_exceeded": dense_accel_exceeded,
                    "validation_yaw_rate_exceeded": dense_yaw_rate_exceeded,
                }
            )
        if stats["validation_speed_exceeded"] and limits_enforced:
            stats["validation_status"] = "failed"
            stats["validation_failure_reason"] = "max_speed_exceeded"
            return None, None, stats
        if (
            limits_enforced
            and validate_dense_dynamics
            and self.validation_reject_dynamics_exceeded
            and (
                stats["validation_accel_exceeded"] or stats["validation_yaw_rate_exceeded"]
            )
        ):
            stats["validation_status"] = "failed"
            stats["validation_failure_reason"] = "dynamics_limit_exceeded"
            return None, None, stats
        if (
            (online_local_guard_disabled or not enforce_limits)
            and (
                stats["max_step_distance_exceeded"]
                or stats["validation_speed_exceeded"]
                or dense_accel_exceeded
                or dense_yaw_rate_exceeded
            )
        ):
            stats["validation_status"] = (
                "ok_with_local_reality_warnings"
                if online_local_guard_disabled
                else "ok_with_validation_warnings"
            )
            stats["validation_failure_reason"] = None
        else:
            stats["validation_status"] = "ok"
        return local_positions, local_yaws, stats

    def _force_last_llm_candidate_trajectory(
        self,
        candidate_meta: Optional[Dict[str, Any]],
        ctrl_idx: int,
        obs: Dict,
        target_horizon: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        if not isinstance(candidate_meta, dict):
            return None, None, None
        parse_meta = candidate_meta.get("parse_meta")
        if not isinstance(parse_meta, dict):
            return None, None, None
        trajectory_world_used = parse_meta.get("trajectory_world_used")
        if not isinstance(trajectory_world_used, list) or len(trajectory_world_used) == 0:
            return None, None, None

        world_points = torch.as_tensor(
            trajectory_world_used,
            dtype=torch.float32,
            device=self.device,
        )
        if (
            world_points.ndim != 2
            or world_points.shape[-1] != 2
            or world_points.shape[0] == 0
            or not torch.isfinite(world_points).all()
        ):
            return None, None, None

        local_positions, local_yaws, stats = self._validate_world_points(
            world_points=world_points,
            ctrl_idx=ctrl_idx,
            obs=obs,
            target_horizon=target_horizon,
            dynamics_validation_mode="diagnostic_only",
            enforce_limits=False,
        )
        if local_positions is None or local_yaws is None:
            return None, None, None

        yaws_world_raw = parse_meta.get("yaws_world_used")
        if isinstance(yaws_world_raw, list) and len(yaws_world_raw) >= local_positions.shape[0]:
            yaws_world = torch.as_tensor(
                yaws_world_raw[: local_positions.shape[0]],
                dtype=torch.float32,
                device=self.device,
            )
            if yaws_world.ndim == 1 and torch.isfinite(yaws_world).all():
                local_yaws = _wrap_to_pi(
                    yaws_world - obs["yaw"][ctrl_idx].to(self.device, dtype=torch.float32)
                ).unsqueeze(-1)

        forced_parse_meta = copy.deepcopy(parse_meta)
        original_status = forced_parse_meta.get("status")
        original_validation_status = forced_parse_meta.get("validation_status")
        original_validation_failure_reason = forced_parse_meta.get("validation_failure_reason")
        forced_parse_meta.update(stats)
        forced_parse_meta.update(
            {
                "status": "ok_last_llm_plan_forced_after_validation_failure",
                "action_source": "target_plan_final_attempt_forced",
                "final_attempt_forced_llm_plan": True,
                "forced_original_status": original_status,
                "forced_original_validation_status": original_validation_status,
                "forced_original_validation_failure_reason": original_validation_failure_reason,
                "validation_status": "forced_after_validation_failure",
                "validation_failure_reason": None,
                "validation_limits_enforced": False,
            }
        )
        return local_positions, local_yaws, forced_parse_meta

    def _build_local_validation_failure_feedback(self, parse_meta: Dict[str, Any]) -> Dict[str, Any]:
        reason = str(parse_meta.get("validation_failure_reason") or parse_meta.get("status") or "unknown")
        max_accel = parse_meta.get("max_abs_accel_mps2")
        max_yaw_rate = parse_meta.get("max_abs_yaw_rate_rps")
        accel_limit = parse_meta.get("validation_max_abs_accel_limit_mps2", self.validation_max_abs_accel_mps2)
        yaw_limit = parse_meta.get("validation_max_abs_yaw_rate_limit_rps", self.validation_max_abs_yaw_rate_rps)
        max_speed_jump = parse_meta.get("max_waypoint_speed_jump_mps")
        max_heading_jump = parse_meta.get("max_waypoint_heading_jump_rad")
        speed_jump_limit = parse_meta.get(
            "validation_max_waypoint_speed_jump_limit_mps",
            self.validation_max_waypoint_speed_jump_mps,
        )
        heading_jump_limit = parse_meta.get(
            "validation_max_waypoint_heading_jump_limit_rad",
            self.validation_max_waypoint_heading_jump_rad,
        )
        details: List[str] = []
        if max_accel is not None:
            details.append(f"max_abs_accel_mps2={float(max_accel):.3f}, limit={float(accel_limit):.3f}")
        if max_yaw_rate is not None:
            details.append(f"max_abs_yaw_rate_rps={float(max_yaw_rate):.3f}, limit={float(yaw_limit):.3f}")
        if max_speed_jump is not None:
            details.append(
                f"max_waypoint_speed_jump_mps={float(max_speed_jump):.3f}, "
                f"limit={float(speed_jump_limit):.3f}"
            )
        if max_heading_jump is not None:
            details.append(
                f"max_waypoint_heading_jump_rad={float(max_heading_jump):.3f}, "
                f"limit={float(heading_jump_limit):.3f}"
            )
        detail_text = "; ".join(details) if details else "no reliable dynamics stats"
        dynamics_source = str(parse_meta.get("dynamics_measurement_source") or "trajectory_validation")
        return {
            "failure_type": "local_trajectory_validation",
            "failure_feedback": (
                f"The previous round's waypoint trajectory did not pass the local check because {reason}. The current dynamics statistics source is {dynamics_source}, check result: {detail_text}. Do not create collisions using instantaneous U-turns, instantaneous lateral jumps, sudden stops, or excessive acceleration."
            ),
            "revision_advice": (
                "Next round, please replan smoother waypoints: the first segment must respect the current yaw_world and speed_mps, and the acceleration and yaw rate between adjacent waypoint segments must both be continuous; when starting from rest or at low speed, first give a short-distance acceleration point, then gradually increase the spacing; if the collision opportunity has already been missed, drive normally along a reasonable route, and do not output a trajectory that violates vehicle dynamics just to pursue a collision."
            ),
            "parse_status": parse_meta.get("status"),
            "validation_failure_reason": parse_meta.get("validation_failure_reason"),
            "max_abs_accel_mps2": max_accel,
            "max_abs_yaw_rate_rps": max_yaw_rate,
            "max_waypoint_speed_jump_mps": max_speed_jump,
            "max_waypoint_heading_jump_rad": max_heading_jump,
            "validation_max_abs_accel_limit_mps2": accel_limit,
            "validation_max_abs_yaw_rate_limit_rps": yaw_limit,
            "validation_max_waypoint_speed_jump_limit_mps": speed_jump_limit,
            "validation_max_waypoint_heading_jump_limit_rad": heading_jump_limit,
        }

    def _parse_llm_target_plan(
        self,
        response_text: str,
        ctrl_idx: int,
        ego_idx: int,
        obs: Dict,
        target_horizon: int,
        step_index: int,
        prior_stage_plan: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        parse_meta: Dict[str, Any] = {
            "status": "not_parsed",
            "parsed_response": None,
            "scenario_goal_restatement": None,
            "stage_plan": None,
            "anchor_plan_reason": None,
            "adv_anchor_plan": None,
            "segment_dynamics": None,
            "plan_analysis": None,
            "adjustment_reason": None,
            "waypoint_plan": None,
            "waypoints_world": None,
            "plan_queue_source": None,
            "trajectory_world_used": None,
            "yaws_world_used": None,
            "first_step_distance_m": None,
            "mean_step_distance_m": None,
            "action_source": None,
        }
        parsed = _extract_json_object(response_text)
        parse_meta["parsed_response"] = parsed
        if parsed is None:
            parse_meta["status"] = "json_parse_failed"
            return None, None, parse_meta
        scenario_goal_restatement = str(parsed.get("scenario_goal_restatement", "") or "").strip()
        stage_plan_raw = parsed.get("stage_plan")
        segment_dynamics_raw = parsed.get("segment_dynamics")
        plan_analysis_raw = parsed.get("plan_analysis")
        waypoint_plan_raw = parsed.get("waypoint_plan")
        updated_waypoint_plan_raw = parsed.get("updated_waypoint_plan")
        adv_anchor_plan_raw = parsed.get("adv_anchor_plan")
        anchor_plan_reason = str(parsed.get("anchor_plan_reason", "") or "").strip()
        adjustment_reason = str(parsed.get("adjustment_reason", "") or "").strip()
        if scenario_goal_restatement:
            parse_meta["scenario_goal_restatement"] = scenario_goal_restatement
        if anchor_plan_reason:
            parse_meta["anchor_plan_reason"] = anchor_plan_reason
        if isinstance(segment_dynamics_raw, list):
            parse_meta["segment_dynamics"] = copy.deepcopy(segment_dynamics_raw)
        parse_meta["adjustment_reason"] = adjustment_reason
        if isinstance(plan_analysis_raw, dict):
            parse_meta["plan_analysis"] = copy.deepcopy(plan_analysis_raw)
        is_initial_plan = isinstance(stage_plan_raw, list)
        active_queue_raw = waypoint_plan_raw if isinstance(waypoint_plan_raw, list) else updated_waypoint_plan_raw
        active_queue_source = "waypoint_plan" if isinstance(waypoint_plan_raw, list) else "updated_waypoint_plan"
        if active_queue_raw is None and isinstance(adv_anchor_plan_raw, list):
            active_queue_raw = adv_anchor_plan_raw
            active_queue_source = "adv_anchor_plan"
            # Anchor-style output intentionally omits a heavy stage plan. Keep a
            # synthetic stage so the downstream waypoint machinery stays unchanged.
            if not is_initial_plan:
                stage_plan_raw = [
                    {
                        "stage_id": 1,
                        "stage_name": "anchor_guided_stage",
                        "goal_description": str(anchor_plan_reason or self.scenario_goal),
                        "completion_condition": (
                            "advance when the sparse anchors have created the intended "
                            "scenario-goal interaction or when this rolling window ends"
                        ),
                    }
                ]
                is_initial_plan = True
        if active_queue_raw is None:
            parse_meta["status"] = "missing_waypoint_plan"
            parse_meta["action_source"] = "target_plan"
            return None, None, parse_meta
        parse_meta["plan_queue_source"] = str(active_queue_source)

        normalized_stage_plan: Optional[List[Dict[str, Any]]] = None
        if is_initial_plan:
            normalized_stage_plan = []
            for stage_item in stage_plan_raw:
                if not isinstance(stage_item, dict):
                    parse_meta["status"] = "invalid_stage_plan_item"
                    parse_meta["action_source"] = "target_plan"
                    return None, None, parse_meta
                stage_id = _safe_int(stage_item.get("stage_id"), None)
                stage_name = str(stage_item.get("stage_name", "") or "").strip()
                goal_description = str(stage_item.get("goal_description", "") or "").strip()
                completion_condition = str(stage_item.get("completion_condition", "") or "").strip()
                if stage_id is None:
                    parse_meta["status"] = "invalid_stage_plan_item"
                    parse_meta["action_source"] = "target_plan"
                    return None, None, parse_meta
                if not stage_name:
                    stage_name = f"stage_{int(stage_id)}"
                if not goal_description:
                    goal_description = stage_name
                if not completion_condition:
                    completion_condition = "advance to the next stage when this behavior is complete"
                normalized_stage_plan.append(
                    {
                        "stage_id": int(stage_id),
                        "stage_name": stage_name,
                        "goal_description": goal_description,
                        "completion_condition": completion_condition,
                    }
                )
            parse_meta["stage_plan"] = copy.deepcopy(normalized_stage_plan)
        else:
            if not adjustment_reason and active_queue_source != "adv_anchor_plan":
                parse_meta["status"] = "missing_adjustment_reason"
                parse_meta["action_source"] = "target_plan"
                return None, None, parse_meta
            if isinstance(prior_stage_plan, list):
                normalized_stage_plan = copy.deepcopy(prior_stage_plan)
                parse_meta["stage_plan"] = copy.deepcopy(normalized_stage_plan)

        if not isinstance(active_queue_raw, list) or len(active_queue_raw) == 0:
            parse_meta["status"] = "empty_waypoint_plan"
            parse_meta["action_source"] = "target_plan"
            return None, None, parse_meta

        normalized_waypoint_plan: List[Dict[str, Any]] = []
        normalized_anchor_plan: Optional[List[Dict[str, Any]]] = [] if active_queue_source == "adv_anchor_plan" else None
        valid_stage_ids = {
            int(stage["stage_id"]) for stage in normalized_stage_plan
        } if isinstance(normalized_stage_plan, list) else None
        default_stage_id = 1
        prev_step_value = 0
        for waypoint in active_queue_raw:
            step_value = None
            position_world = None
            stage_id_value = None
            waypoint_reason = ""
            if isinstance(waypoint, dict):
                step_value = _safe_int(waypoint.get("step"), None)
                position_world = waypoint.get("position_world")
                stage_id_value = _safe_int(waypoint.get("stage_id"), None)
                waypoint_reason = str(waypoint.get("reason", "") or "").strip()
            if step_value is None or step_value <= prev_step_value:
                parse_meta["status"] = "invalid_waypoint_step_sequence"
                parse_meta["action_source"] = "target_plan"
                return None, None, parse_meta
            if stage_id_value is None:
                if active_queue_source == "adv_anchor_plan":
                    stage_id_value = int(default_stage_id or 1)
                else:
                    parse_meta["status"] = "invalid_waypoint_stage_id"
                    parse_meta["action_source"] = "target_plan"
                    return None, None, parse_meta
            if valid_stage_ids is not None and int(stage_id_value) not in valid_stage_ids:
                parse_meta["status"] = "unknown_waypoint_stage_id"
                parse_meta["action_source"] = "target_plan"
                return None, None, parse_meta
            if (
                not isinstance(position_world, (list, tuple))
                or len(position_world) != 2
                or _safe_float(position_world[0], None) is None
                or _safe_float(position_world[1], None) is None
            ):
                parse_meta["status"] = "invalid_waypoint_position_world"
                parse_meta["action_source"] = "target_plan"
                return None, None, parse_meta
            normalized_waypoint_plan.append(
                {
                    "step": int(step_value),
                    "stage_id": int(stage_id_value),
                    "position_world": [
                        float(position_world[0]),
                        float(position_world[1]),
                    ],
                }
            )
            if normalized_anchor_plan is not None:
                normalized_anchor_plan.append(
                    {
                        "step": int(step_value),
                        "position_world": [
                            float(position_world[0]),
                            float(position_world[1]),
                        ],
                        "reason": waypoint_reason,
                    }
                )
            prev_step_value = int(step_value)

        interval_steps = max(1, int(self.waypoint_interval_steps))
        for waypoint_prev, waypoint_curr in zip(
            normalized_waypoint_plan[:-1],
            normalized_waypoint_plan[1:],
        ):
            if int(waypoint_curr["step"]) - int(waypoint_prev["step"]) != interval_steps:
                parse_meta["status"] = "invalid_waypoint_interval_steps"
                parse_meta["action_source"] = "target_plan"
                return None, None, parse_meta
        if int(normalized_waypoint_plan[0]["step"]) != interval_steps:
            parse_meta["status"] = "invalid_first_waypoint_step"
            parse_meta["action_source"] = "target_plan"
            return None, None, parse_meta

        if normalized_waypoint_plan[-1]["step"] < int(self.executed_steps_per_call):
            parse_meta["status"] = "waypoint_plan_too_short"
            parse_meta["action_source"] = "target_plan"
            return None, None, parse_meta

        parse_meta["waypoint_plan"] = copy.deepcopy(normalized_waypoint_plan)
        parse_meta["waypoints_world"] = copy.deepcopy(normalized_waypoint_plan)
        if normalized_anchor_plan is not None:
            parse_meta["adv_anchor_plan"] = copy.deepcopy(normalized_anchor_plan)
        waypoint_segment_stats = self._validate_waypoint_segment_dynamics(
            ctrl_idx=ctrl_idx,
            obs=obs,
            waypoints_world=normalized_waypoint_plan,
        )
        parse_meta.update(waypoint_segment_stats)
        waypoint_segment_validation_failed = bool(
            self.validation_reject_waypoint_segment_jumps
            and waypoint_segment_stats.get("waypoint_segment_validation_status") == "failed"
        )

        dense_rollout_horizon = max(target_horizon, int(normalized_waypoint_plan[-1]["step"]))
        world_points, yaws_world, velocity_sequence, rollout_meta = self._rollout_world_points_from_waypoints(
            ctrl_idx=ctrl_idx,
            obs=obs,
            waypoints_world=normalized_waypoint_plan,
            rollout_horizon_steps=dense_rollout_horizon,
        )
        parse_meta["trajectory_world_used"] = world_points.detach().cpu().tolist()
        parse_meta["yaws_world_used"] = yaws_world.detach().cpu().tolist()
        parse_meta["velocity_world_used"] = velocity_sequence.detach().cpu().tolist()
        if velocity_sequence.shape[0] > 0:
            parse_meta["terminal_velocity_world_used"] = velocity_sequence[-1].detach().cpu().tolist()
            executed_steps = min(self.executed_steps_per_call, int(velocity_sequence.shape[0]))
            parse_meta["executed_terminal_velocity_world_used"] = (
                velocity_sequence[executed_steps - 1].detach().cpu().tolist()
            )
        parse_meta["target_rollout_meta"] = copy.deepcopy(rollout_meta)
        local_positions, local_yaws, stats = self._validate_world_points(
            world_points=world_points,
            ctrl_idx=ctrl_idx,
            obs=obs,
            target_horizon=target_horizon,
            dynamics_validation_mode="diagnostic_only",
        )
        parse_meta.update(stats)
        if waypoint_segment_validation_failed:
            parse_meta["status"] = "invalid_waypoint_segment_dynamics"
            parse_meta["action_source"] = "target_plan"
            parse_meta["validation_status"] = "failed"
            parse_meta["validation_failure_reason"] = str(
                waypoint_segment_stats.get(
                    "waypoint_segment_validation_failure_reason",
                    "waypoint_segment_dynamics_exceeded",
                )
            )
            return None, None, parse_meta
        if local_positions is None or local_yaws is None:
            parse_meta["status"] = "target_rollout_invalid_after_guard"
            parse_meta["action_source"] = "target_plan"
            return None, None, parse_meta

        yaws_world = yaws_world[: local_positions.shape[0]]
        local_yaws = _wrap_to_pi(
            yaws_world - obs["yaw"][ctrl_idx].to(self.device, dtype=torch.float32)
        ).unsqueeze(-1)

        parse_meta["status"] = "ok_target_plan"
        parse_meta["action_source"] = "target_plan"
        return local_positions, local_yaws, parse_meta

    def _build_next_planning_state_record(
        self,
        parse_meta: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(parse_meta, dict):
            return None
        waypoint_plan = parse_meta.get("waypoint_plan")
        if not isinstance(waypoint_plan, list):
            return None
        remaining_waypoint_plan = self._shift_remaining_waypoint_plan(
            waypoint_queue=waypoint_plan,
            executed_steps=self.executed_steps_per_call,
        )
        return {
            "scenario_goal_restatement": copy.deepcopy(parse_meta.get("scenario_goal_restatement")),
            "stage_plan": copy.deepcopy(parse_meta.get("stage_plan")),
            "anchor_plan_reason": copy.deepcopy(parse_meta.get("anchor_plan_reason")),
            "adv_anchor_plan": copy.deepcopy(parse_meta.get("adv_anchor_plan")),
            "segment_dynamics": copy.deepcopy(parse_meta.get("segment_dynamics")),
            "plan_analysis": copy.deepcopy(parse_meta.get("plan_analysis")),
            "remaining_waypoint_plan": copy.deepcopy(remaining_waypoint_plan),
        }

    def _update_cached_plan(
        self,
        cache_key: Tuple[int, int],
        step_index: int,
        parse_meta: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(parse_meta, dict):
            return
        trajectory_world_used = parse_meta.get("trajectory_world_used")
        if not isinstance(trajectory_world_used, list) or len(trajectory_world_used) == 0:
            return
        yaws_world_used = parse_meta.get("yaws_world_used")
        velocity_world_used = parse_meta.get("velocity_world_used")
        self._last_plan_cache_by_agent[cache_key] = {
            "step_index": int(step_index),
            "scenario_goal_restatement": copy.deepcopy(parse_meta.get("scenario_goal_restatement")),
            "stage_plan": copy.deepcopy(parse_meta.get("stage_plan")),
            "anchor_plan_reason": copy.deepcopy(parse_meta.get("anchor_plan_reason")),
            "adv_anchor_plan": copy.deepcopy(parse_meta.get("adv_anchor_plan")),
            "segment_dynamics": copy.deepcopy(parse_meta.get("segment_dynamics")),
            "plan_analysis": copy.deepcopy(parse_meta.get("plan_analysis")),
            "waypoint_plan": copy.deepcopy(parse_meta.get("waypoint_plan")),
            "waypoints_world": copy.deepcopy(parse_meta.get("waypoints_world")),
            "trajectory_world_used": copy.deepcopy(trajectory_world_used),
            "yaws_world_used": copy.deepcopy(yaws_world_used),
            "velocity_world_used": copy.deepcopy(velocity_world_used),
        }

    def _reuse_cached_plan(
        self,
        cache_key: Tuple[int, int],
        step_index: int,
        ctrl_idx: int,
        obs: Dict,
        target_horizon: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        cached_plan = self._last_plan_cache_by_agent.get(cache_key)
        if cached_plan is None:
            return None, None, None

        last_step_index = int(cached_plan.get("step_index", -1))
        elapsed_steps = int(step_index) - last_step_index
        if elapsed_steps <= 0 or elapsed_steps >= self.call_interval_steps:
            return None, None, None

        trajectory_world_used = cached_plan.get("trajectory_world_used")
        if not isinstance(trajectory_world_used, list) or len(trajectory_world_used) <= elapsed_steps:
            return None, None, None

        world_points = torch.as_tensor(
            trajectory_world_used[elapsed_steps:],
            dtype=torch.float32,
            device=self.device,
        )
        if world_points.ndim != 2 or world_points.shape[-1] != 2:
            return None, None, None

        local_positions, local_yaws, stats = self._validate_world_points(
            world_points=world_points,
            ctrl_idx=ctrl_idx,
            obs=obs,
            target_horizon=target_horizon,
        )
        if local_positions is None or local_yaws is None:
            return None, None, None

        cached_yaws_world = cached_plan.get("yaws_world_used")
        if isinstance(cached_yaws_world, list) and len(cached_yaws_world) > elapsed_steps:
            cached_yaws_world = copy.deepcopy(cached_yaws_world[elapsed_steps:])
        else:
            cached_yaws_world = None
        cached_velocity_world = cached_plan.get("velocity_world_used")
        if isinstance(cached_velocity_world, list) and len(cached_velocity_world) > elapsed_steps:
            cached_velocity_world = copy.deepcopy(cached_velocity_world[elapsed_steps:])
        else:
            cached_velocity_world = None

        parse_meta = {
            "status": "ok_cached_plan_reuse",
            "parsed_response": None,
            "scenario_goal_restatement": copy.deepcopy(cached_plan.get("scenario_goal_restatement")),
            "stage_plan": copy.deepcopy(cached_plan.get("stage_plan")),
            "anchor_plan_reason": copy.deepcopy(cached_plan.get("anchor_plan_reason")),
            "adv_anchor_plan": copy.deepcopy(cached_plan.get("adv_anchor_plan")),
            "segment_dynamics": copy.deepcopy(cached_plan.get("segment_dynamics")),
            "plan_analysis": copy.deepcopy(cached_plan.get("plan_analysis")),
            "adjustment_reason": None,
            "waypoint_plan": copy.deepcopy(cached_plan.get("waypoint_plan")),
            "waypoints_world": copy.deepcopy(cached_plan.get("waypoints_world")),
            "trajectory_world_used": world_points.detach().cpu().tolist(),
            "yaws_world_used": cached_yaws_world,
            "velocity_world_used": cached_velocity_world,
            "action_source": "cached_plan_reuse",
            "cached_plan_meta": {
                "original_step_index": last_step_index,
                "current_step_index": int(step_index),
                "elapsed_steps": elapsed_steps,
                "call_interval_steps": int(self.call_interval_steps),
            },
        }
        parse_meta.update(stats)
        return local_positions, local_yaws, parse_meta

    def _fallback_from_cached_terminal_motion(
        self,
        cache_key: Tuple[int, int],
        step_index: int,
        ctrl_idx: int,
        obs: Dict,
        target_horizon: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        cached_plan = self._last_plan_cache_by_agent.get(cache_key)
        if cached_plan is None:
            return None, None, None

        cached_velocity_world = cached_plan.get("velocity_world_used")
        if not isinstance(cached_velocity_world, list) or len(cached_velocity_world) == 0:
            return None, None, None
        terminal_velocity = cached_velocity_world[-1]
        if not isinstance(terminal_velocity, (list, tuple)) or len(terminal_velocity) != 2:
            return None, None, None

        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        current_pos = obs["centroid"][ctrl_idx].detach().to(self.device, dtype=torch.float32)
        velocity_world = torch.tensor(terminal_velocity, dtype=torch.float32, device=self.device)
        speed_mps = float(torch.linalg.norm(velocity_world).item())
        if speed_mps <= 0.25:
            return None, None, None
        if speed_mps > self.max_speed_mps:
            velocity_world = velocity_world * (self.max_speed_mps / max(speed_mps, 1e-6))
            speed_mps = float(torch.linalg.norm(velocity_world).item())

        rollout_steps = max(2, int(target_horizon))
        world_points = torch.zeros((rollout_steps, 2), dtype=torch.float32, device=self.device)
        yaws_world = torch.zeros((rollout_steps,), dtype=torch.float32, device=self.device)
        velocity_world_used: List[List[float]] = []
        yaws_world_used: List[float] = []

        curr_pos = current_pos.clone()
        terminal_yaw = math.atan2(float(velocity_world[1].item()), float(velocity_world[0].item()))
        for step_idx in range(rollout_steps):
            curr_pos = curr_pos + velocity_world * dt
            world_points[step_idx] = curr_pos
            yaws_world[step_idx] = float(terminal_yaw)
            velocity_world_used.append([float(velocity_world[0].item()), float(velocity_world[1].item())])
            yaws_world_used.append(float(terminal_yaw))

        local_positions, local_yaws, stats = self._validate_world_points(
            world_points=world_points,
            ctrl_idx=ctrl_idx,
            obs=obs,
            target_horizon=target_horizon,
        )
        if local_positions is None or local_yaws is None:
            return None, None, None

        parse_meta = {
            "status": "ok_cached_terminal_motion_fallback",
            "parsed_response": None,
            "scenario_goal_restatement": copy.deepcopy(cached_plan.get("scenario_goal_restatement")),
            "stage_plan": copy.deepcopy(cached_plan.get("stage_plan")),
            "anchor_plan_reason": copy.deepcopy(cached_plan.get("anchor_plan_reason")),
            "adv_anchor_plan": copy.deepcopy(cached_plan.get("adv_anchor_plan")),
            "segment_dynamics": copy.deepcopy(cached_plan.get("segment_dynamics")),
            "plan_analysis": copy.deepcopy(cached_plan.get("plan_analysis")),
            "adjustment_reason": None,
            "waypoint_plan": copy.deepcopy(cached_plan.get("waypoint_plan")),
            "waypoints_world": copy.deepcopy(cached_plan.get("waypoints_world")),
            "trajectory_world_used": world_points.detach().cpu().tolist(),
            "yaws_world_used": yaws_world_used,
            "velocity_world_used": velocity_world_used,
            "action_source": "cached_terminal_motion_fallback",
            "cached_plan_meta": {
                "original_step_index": int(cached_plan.get("step_index", -1)),
                "current_step_index": int(step_index),
                "call_interval_steps": int(self.call_interval_steps),
                "fallback_reason": "planner_failed_or_invalid",
                "fallback_speed_mps": float(speed_mps),
            },
        }
        parse_meta.update(stats)
        return local_positions, local_yaws, parse_meta

    def _query_llm_for_ctrl(
        self,
        obs: Dict,
        ctrl_idx: int,
        ego_idx: int,
        step_index: int,
        planning_state: Optional[Dict[str, Any]] = None,
        failure_feedback: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        context = self._build_prompt_context(
            obs,
            ctrl_idx,
            ego_idx,
            step_index,
            planning_state=planning_state,
            failure_feedback=failure_feedback,
        )
        target_horizon = int(context.get("execution_horizon", self.executed_steps_per_call))
        system_prompt, user_prompt = self._build_prompt(context)
        self._maybe_write_prompt_sample(
            step_index=step_index,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        bev_data_url, bev_meta = self._build_bev_data_url(
            obs=obs,
            ctrl_idx=ctrl_idx,
            step_index=step_index,
            scene_index=int(obs["scene_index"][ctrl_idx].item()),
            ego_idx=ego_idx,
            overlay_mode="planner",
        )
        context["send_bev_to_llm"] = bool(self.send_bev_to_llm)
        context["bev_attached"] = bool(bev_meta.get("bev_attached", False))
        context["bev_status"] = str(bev_meta.get("bev_status", "unknown"))

        manual_response_text = self._load_manual_response_text(step_index)
        if manual_response_text is not None:
            print(
                f"[AnchorDriveLLM] step={step_index} ctrl_track_id={int(obs['track_id'][ctrl_idx].item())} "
                f"manual_response_loaded path={self.manual_response_path}",
                flush=True,
            )
            local_positions, local_yaws, parse_meta = self._parse_llm_target_plan(
                response_text=manual_response_text,
                ctrl_idx=ctrl_idx,
                ego_idx=ego_idx,
                obs=obs,
                target_horizon=target_horizon,
                step_index=step_index,
                prior_stage_plan=planning_state.get("stage_plan") if isinstance(planning_state, dict) else None,
            )
            parsed_output_for_history = None
            if local_positions is not None and local_yaws is not None:
                parsed_output_for_history = self._build_next_planning_state_record(parse_meta)
            return local_positions, local_yaws, {
                "status": str(parse_meta.get("status", "manual_parse_unknown")),
                "retry_idx": -1,
                "context": context,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_text": manual_response_text,
                "parse_meta": parse_meta,
                "bev_meta": bev_meta,
                "parsed_output_for_history": parsed_output_for_history,
            }

        last_status = "not_called"
        last_response = ""
        last_parse_meta: Dict[str, Any] = {
            "status": "not_called",
            "parsed_response": None,
            "stage_plan": None,
            "anchor_plan_reason": None,
            "adv_anchor_plan": None,
            "plan_analysis": None,
            "adjustment_reason": None,
            "waypoint_plan": None,
            "waypoints_world": None,
            "plan_queue_source": None,
            "trajectory_world_used": None,
            "yaws_world_used": None,
            "first_step_distance_m": None,
            "mean_step_distance_m": None,
        }
        last_parsed_output_for_history: Optional[Dict[str, Any]] = None
        for retry_idx in range(self.max_retries):
            request_start_t = time.time()
            print(
                f"[AnchorDriveLLM] step={step_index} ctrl_track_id={int(obs['track_id'][ctrl_idx].item())} "
                f"retry={retry_idx} llm_request_start model={self.llm_cfg.model}",
                flush=True,
            )
            try:
                response_text = self._get_client().complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    user_image_data_url=bev_data_url,
                )
            except Exception as exc:  # pragma: no cover
                last_status = f"llm_error:{type(exc).__name__}"
                last_response = str(exc)
                elapsed_s = time.time() - request_start_t
                print(
                    f"[AnchorDriveLLM] step={step_index} ctrl_track_id={int(obs['track_id'][ctrl_idx].item())} "
                    f"retry={retry_idx} llm_request_error elapsed_s={elapsed_s:.2f} error={last_status}",
                    flush=True,
                )
                continue
            elapsed_s = time.time() - request_start_t
            print(
                f"[AnchorDriveLLM] step={step_index} ctrl_track_id={int(obs['track_id'][ctrl_idx].item())} "
                f"retry={retry_idx} llm_request_done elapsed_s={elapsed_s:.2f}",
                flush=True,
            )

            last_response = response_text
            local_positions, local_yaws, parse_meta = self._parse_llm_target_plan(
                response_text=response_text,
                ctrl_idx=ctrl_idx,
                ego_idx=ego_idx,
                obs=obs,
                target_horizon=target_horizon,
                step_index=step_index,
                prior_stage_plan=planning_state.get("stage_plan") if isinstance(planning_state, dict) else None,
            )
            if (
                isinstance(planning_state, dict)
                and parse_meta.get("stage_plan") is None
                and isinstance(planning_state.get("stage_plan"), list)
            ):
                parse_meta["stage_plan"] = copy.deepcopy(planning_state.get("stage_plan"))
            if (
                isinstance(planning_state, dict)
                and parse_meta.get("plan_analysis") is None
                and isinstance(planning_state.get("plan_analysis"), dict)
            ):
                parse_meta["plan_analysis"] = copy.deepcopy(planning_state.get("plan_analysis"))
            last_status = str(parse_meta.get("status", "parse_unknown"))
            last_parse_meta = parse_meta
            planning_state_record = self._build_next_planning_state_record(parse_meta)
            if isinstance(planning_state_record, dict):
                last_parsed_output_for_history = copy.deepcopy(planning_state_record)
            if local_positions is not None and local_yaws is not None:
                return local_positions, local_yaws, {
                    "status": last_status,
                    "retry_idx": retry_idx,
                    "context": context,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response_text": response_text,
                    "parse_meta": parse_meta,
                    "bev_meta": bev_meta,
                    "parsed_output_for_history": last_parsed_output_for_history,
                }
            if parse_meta.get("validation_failure_reason") in {
                "dynamics_limit_exceeded",
                "waypoint_segment_jump_exceeded",
                "waypoint_segment_dynamics_exceeded",
            }:
                break

        return None, None, {
            "status": last_status,
            "retry_idx": self.max_retries - 1,
            "context": context,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": last_response,
            "parse_meta": last_parse_meta,
            "bev_meta": bev_meta,
            "parsed_output_for_history": last_parsed_output_for_history,
        }

    def _compute_world_yaws_from_points(
        self,
        points_world: torch.Tensor,
        current_yaw: float,
    ) -> torch.Tensor:
        current_yaw_t = torch.tensor(float(current_yaw), dtype=torch.float32, device=self.device)
        local_yaws = self._compute_local_yaws(
            points_world=points_world,
            current_yaw=current_yaw_t,
            min_heading_update_distance=self.min_heading_update_distance,
        )
        return _wrap_to_pi(local_yaws + current_yaw_t)

    def _build_rollout_steps_summary(
        self,
        positions_world: torch.Tensor,
        yaws_world: torch.Tensor,
        steps: int,
    ) -> List[Dict[str, Any]]:
        max_steps = min(int(steps), int(positions_world.shape[0]), int(yaws_world.shape[0]))
        summary: List[Dict[str, Any]] = []
        for step_idx in range(max_steps):
            summary.append(
                {
                    "step": int(step_idx),
                    "position_world": positions_world[step_idx].detach().cpu().tolist(),
                    "yaw_world": float(yaws_world[step_idx].item()),
                }
            )
        return summary

    def _extract_executed_rollout_summary(
        self,
        obs: Dict,
        ctrl_idx: int,
        ego_idx: int,
        parse_meta: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        trajectory_world_used = parse_meta.get("trajectory_world_used")
        if not isinstance(trajectory_world_used, list):
            return None

        adv_positions_world = torch.as_tensor(
            trajectory_world_used,
            dtype=torch.float32,
            device=self.device,
        )
        if adv_positions_world.ndim != 2 or adv_positions_world.shape[-1] != 2 or adv_positions_world.shape[0] == 0:
            return None

        ego_plan_world_raw = obs.get("ego_plan")
        if ego_plan_world_raw is None:
            return None
        dt = float(obs["dt"][ctrl_idx].item()) if "dt" in obs else 0.1
        target_horizon = min(
            int(self.executed_steps_per_call),
            int(obs["target_positions"].shape[1]),
            int(adv_positions_world.shape[0]),
        )
        ego_plan = ego_plan_world_raw.to(self.device)
        if ego_plan.ndim == 2:
            ego_plan = ego_plan.unsqueeze(0)
        ego_indices = obs.get("ego_idx")
        if ego_indices is None:
            ego_plan = ego_plan[:1]
        else:
            if isinstance(ego_indices, torch.Tensor):
                ego_indices = ego_indices.detach().cpu().tolist()
            else:
                ego_indices = list(ego_indices)
            ego_plan_row = ego_indices.index(ego_idx)
            ego_plan = ego_plan[ego_plan_row : ego_plan_row + 1]
        if ego_plan.shape[1] < target_horizon:
            ego_plan = extend_ego_plan(ego_plan, target_horizon, dt=dt)
        ego_plan_world = self._to_world(
            ego_plan[0, :target_horizon],
            obs["world_from_agent"][ego_idx].to(self.device),
        )

        executed_steps = min(
            self.executed_steps_per_call,
            adv_positions_world.shape[0],
            ego_plan_world.shape[0],
        )
        if executed_steps <= 0:
            return None

        adv_positions_world = adv_positions_world[:executed_steps]
        ego_positions_world = ego_plan_world[:executed_steps]
        adv_yaws_world_raw = parse_meta.get("yaws_world_used")
        if isinstance(adv_yaws_world_raw, list) and len(adv_yaws_world_raw) >= executed_steps:
            adv_yaws_world = torch.as_tensor(
                adv_yaws_world_raw[:executed_steps],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            adv_yaws_world = self._compute_world_yaws_from_points(
                adv_positions_world,
                current_yaw=float(obs["yaw"][ctrl_idx].item()),
            )
        ego_yaws_world = self._compute_world_yaws_from_points(
            ego_positions_world,
            current_yaw=float(obs["yaw"][ego_idx].item()),
        )
        distances_m = torch.norm(ego_positions_world - adv_positions_world, dim=-1)
        post_idx = executed_steps - 1

        return {
            "executed_steps": int(executed_steps),
            "step_dt_s": float(dt),
            "full_planner_response": copy.deepcopy(parse_meta.get("parsed_response")),
            "llm_reply_executed_segment": {
                "scenario_goal_restatement": copy.deepcopy(parse_meta.get("scenario_goal_restatement")),
                "stage_plan": copy.deepcopy(parse_meta.get("stage_plan")),
                "anchor_plan_reason": copy.deepcopy(parse_meta.get("anchor_plan_reason")),
                "adv_anchor_plan": copy.deepcopy(parse_meta.get("adv_anchor_plan")),
                "segment_dynamics": copy.deepcopy(parse_meta.get("segment_dynamics")),
                "plan_analysis": copy.deepcopy(parse_meta.get("plan_analysis")),
                "adjustment_reason": parse_meta.get("adjustment_reason"),
                "waypoint_plan": copy.deepcopy(parse_meta.get("waypoint_plan")),
                "waypoints_world": copy.deepcopy(parse_meta.get("waypoints_world")),
            },
            "rollout_dynamics_summary": {
                "dynamics_measurement_source": parse_meta.get("dynamics_measurement_source"),
                "waypoint_segment_validation_mode": parse_meta.get("waypoint_segment_validation_mode"),
                "first_step_distance_m": parse_meta.get("first_step_distance_m"),
                "mean_step_distance_m": parse_meta.get("mean_step_distance_m"),
                "max_step_distance_m": parse_meta.get("max_step_distance_m"),
                "max_speed_mps": parse_meta.get("max_speed_mps"),
                "max_abs_accel_mps2": parse_meta.get("max_abs_accel_mps2"),
                "max_abs_yaw_rate_rps": parse_meta.get("max_abs_yaw_rate_rps"),
                "max_waypoint_segment_abs_accel_mps2": parse_meta.get("max_waypoint_segment_abs_accel_mps2"),
                "max_waypoint_segment_abs_yaw_rate_rps": parse_meta.get("max_waypoint_segment_abs_yaw_rate_rps"),
                "dense_max_abs_accel_mps2": parse_meta.get("dense_max_abs_accel_mps2"),
                "dense_max_abs_yaw_rate_rps": parse_meta.get("dense_max_abs_yaw_rate_rps"),
                "max_waypoint_speed_jump_mps": parse_meta.get("max_waypoint_speed_jump_mps"),
                "max_waypoint_heading_jump_rad": parse_meta.get("max_waypoint_heading_jump_rad"),
                "validation_accel_exceeded": parse_meta.get("validation_accel_exceeded"),
                "validation_yaw_rate_exceeded": parse_meta.get("validation_yaw_rate_exceeded"),
                "waypoint_segment_accel_exceeded": parse_meta.get("waypoint_segment_accel_exceeded"),
                "waypoint_segment_yaw_rate_exceeded": parse_meta.get("waypoint_segment_yaw_rate_exceeded"),
                "dense_validation_accel_exceeded": parse_meta.get("dense_validation_accel_exceeded"),
                "dense_validation_yaw_rate_exceeded": parse_meta.get("dense_validation_yaw_rate_exceeded"),
                "waypoint_speed_jump_exceeded": parse_meta.get("waypoint_speed_jump_exceeded"),
                "waypoint_heading_jump_exceeded": parse_meta.get("waypoint_heading_jump_exceeded"),
                "validation_max_abs_accel_limit_mps2": parse_meta.get(
                    "validation_max_abs_accel_limit_mps2",
                    self.validation_max_abs_accel_mps2,
                ),
                "validation_max_abs_yaw_rate_limit_rps": parse_meta.get(
                    "validation_max_abs_yaw_rate_limit_rps",
                    self.validation_max_abs_yaw_rate_rps,
                ),
                "validation_max_speed_limit_mps": parse_meta.get(
                    "validation_max_speed_limit_mps",
                    self.validation_max_speed_mps,
                ),
                "validation_max_waypoint_speed_jump_limit_mps": parse_meta.get(
                    "validation_max_waypoint_speed_jump_limit_mps",
                    self.validation_max_waypoint_speed_jump_mps,
                ),
                "validation_max_waypoint_heading_jump_limit_rad": parse_meta.get(
                    "validation_max_waypoint_heading_jump_limit_rad",
                    self.validation_max_waypoint_heading_jump_rad,
                ),
                "waypoint_segment_summaries": copy.deepcopy(parse_meta.get("waypoint_segment_summaries")),
            },
            "adv_rollout": {
                "trajectory_world": self._build_rollout_steps_summary(
                    adv_positions_world,
                    adv_yaws_world,
                    executed_steps,
                )
            },
            "ego_rollout": {
                "trajectory_world": self._build_rollout_steps_summary(
                    ego_positions_world,
                    ego_yaws_world,
                    executed_steps,
                )
            },
            "ego_adv_distance_per_step_m": [
                {
                    "step": int(step_idx),
                    "distance_m": float(distances_m[step_idx].item()),
                }
                for step_idx in range(executed_steps)
            ],
            "post_rollout_interaction_analysis": self._build_ego_adv_interaction_analysis_text(
                adv_pos=adv_positions_world[post_idx],
                adv_yaw=float(adv_yaws_world[post_idx].item()),
                ego_pos=ego_positions_world[post_idx],
                ego_yaw=float(ego_yaws_world[post_idx].item()),
            ),
            "adv_positions_world_tensor": adv_positions_world,
            "adv_yaws_world_tensor": adv_yaws_world,
            "ego_positions_world_tensor": ego_positions_world,
            "ego_yaws_world_tensor": ego_yaws_world,
        }

    def _check_local_ego_adv_collision(
        self,
        obs: Dict,
        ctrl_idx: int,
        ego_idx: int,
        rollout_summary: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        adv_positions_world = rollout_summary.get("adv_positions_world_tensor")
        adv_yaws_world = rollout_summary.get("adv_yaws_world_tensor")
        ego_positions_world = rollout_summary.get("ego_positions_world_tensor")
        ego_yaws_world = rollout_summary.get("ego_yaws_world_tensor")
        if any(item is None for item in [adv_positions_world, adv_yaws_world, ego_positions_world, ego_yaws_world]):
            return None

        adv_extent = obs["extent"][ctrl_idx, :2].detach().cpu().numpy()
        ego_extent = obs["extent"][ego_idx, :2].detach().cpu().numpy()
        steps = min(
            int(rollout_summary.get("executed_steps", 0)),
            int(adv_positions_world.shape[0]),
            int(ego_positions_world.shape[0]),
        )
        for step_idx in range(steps):
            collision = detect_collision(
                ego_pos=ego_positions_world[step_idx].detach().cpu().numpy(),
                ego_yaw=float(ego_yaws_world[step_idx].item()),
                ego_extent=ego_extent,
                other_pos=adv_positions_world[step_idx : step_idx + 1].detach().cpu().numpy(),
                other_yaw=np.asarray([float(adv_yaws_world[step_idx].item())], dtype=np.float32),
                other_extent=np.asarray([adv_extent], dtype=np.float32),
            )
            if collision is not None:
                return {
                    "collision_step": int(step_idx),
                    "executed_steps_until_collision": int(step_idx + 1),
                    "ego_position_world": ego_positions_world[step_idx].detach().cpu().tolist(),
                    "adv_position_world": adv_positions_world[step_idx].detach().cpu().tolist(),
                    "ego_yaw_world": float(ego_yaws_world[step_idx].item()),
                    "adv_yaw_world": float(adv_yaws_world[step_idx].item()),
                    "collision_type": str(collision[0]),
                }
        return None

    def _build_local_reality_assessor_meta(
        self,
        obs: Dict,
        ctrl_idx: int,
        ego_idx: int,
        parse_meta: Dict[str, Any],
        rollout_summary: Dict[str, Any],
        local_collision_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        findings: List[Dict[str, Any]] = []

        def _add_finding(name: str, value: Any, limit: Any = None, severity: str = "warning") -> None:
            findings.append(
                {
                    "name": name,
                    "value": value,
                    "limit": limit,
                    "severity": severity,
                }
            )

        if bool(parse_meta.get("max_step_distance_exceeded", False)):
            _add_finding(
                "max_step_distance_exceeded",
                parse_meta.get("max_step_distance_m"),
                parse_meta.get("max_step_distance_limit_m", self.max_step_distance),
                severity="severe",
            )
        if bool(parse_meta.get("validation_speed_exceeded", False)):
            _add_finding(
                "speed_limit_exceeded",
                parse_meta.get("max_speed_mps"),
                parse_meta.get("validation_max_speed_limit_mps", self.validation_max_speed_mps),
                severity="severe",
            )
        dynamics_source = str(parse_meta.get("dynamics_measurement_source") or "")
        if bool(parse_meta.get("validation_accel_exceeded", False)):
            _add_finding(
                (
                    "waypoint_segment_accel_limit_exceeded"
                    if dynamics_source == "sparse_waypoint_segments"
                    else "accel_limit_exceeded"
                ),
                parse_meta.get("max_abs_accel_mps2"),
                parse_meta.get(
                    "validation_max_abs_accel_limit_mps2",
                    self.validation_max_abs_accel_mps2,
                ),
                severity="warning",
            )
        if bool(parse_meta.get("validation_yaw_rate_exceeded", False)):
            _add_finding(
                (
                    "waypoint_segment_yaw_rate_limit_exceeded"
                    if dynamics_source == "sparse_waypoint_segments"
                    else "yaw_rate_limit_exceeded"
                ),
                parse_meta.get("max_abs_yaw_rate_rps"),
                parse_meta.get(
                    "validation_max_abs_yaw_rate_limit_rps",
                    self.validation_max_abs_yaw_rate_rps,
                ),
                severity="warning",
            )
        if bool(parse_meta.get("waypoint_speed_jump_exceeded", False)):
            _add_finding(
                "waypoint_speed_jump_exceeded",
                parse_meta.get("max_waypoint_speed_jump_mps"),
                parse_meta.get(
                    "validation_max_waypoint_speed_jump_limit_mps",
                    self.validation_max_waypoint_speed_jump_mps,
                ),
                severity="warning",
            )
        if bool(parse_meta.get("waypoint_heading_jump_exceeded", False)):
            _add_finding(
                "waypoint_heading_jump_exceeded",
                parse_meta.get("max_waypoint_heading_jump_rad"),
                parse_meta.get(
                    "validation_max_waypoint_heading_jump_limit_rad",
                    self.validation_max_waypoint_heading_jump_rad,
                ),
                severity="warning",
            )

        lane_deviation_meta: Dict[str, Any] = {"status": "not_available"}
        adv_positions_world = rollout_summary.get("adv_positions_world_tensor")
        if isinstance(adv_positions_world, torch.Tensor):
            try:
                map_info = self._get_map_information(obs, adv_idx=ctrl_idx, ego_idx=ego_idx)
                lanes = map_info.get("candidate_lanes", []) if isinstance(map_info, dict) else []
                lane_points: List[torch.Tensor] = []
                lane_widths: List[float] = []
                for lane in lanes:
                    if not isinstance(lane, dict):
                        continue
                    centerline = lane.get("centerline_world")
                    if not isinstance(centerline, list) or len(centerline) < 2:
                        continue
                    center_tensor = torch.as_tensor(
                        centerline,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if center_tensor.ndim != 2 or center_tensor.shape[-1] != 2:
                        continue
                    lane_points.append(center_tensor)
                    lane_widths.append(float(lane.get("lane_width_m", self.default_lane_width_m)))
                if lane_points:
                    per_lane_dist = [
                        torch.cdist(adv_positions_world.to(self.device, dtype=torch.float32), lane_xy).min(dim=1).values
                        for lane_xy in lane_points
                    ]
                    min_dist = torch.stack(per_lane_dist, dim=0).min(dim=0).values
                    max_deviation = float(min_dist.max().item())
                    mean_deviation = float(min_dist.mean().item())
                    lane_width = max(lane_widths) if lane_widths else float(self.default_lane_width_m)
                    deviation_limit = max(4.5, 0.5 * float(lane_width) + 2.5)
                    lane_deviation_meta = {
                        "status": "ok" if max_deviation <= deviation_limit else "warning",
                        "max_candidate_lane_deviation_m": max_deviation,
                        "mean_candidate_lane_deviation_m": mean_deviation,
                        "candidate_lane_deviation_limit_m": float(deviation_limit),
                        "num_candidate_lanes": int(len(lane_points)),
                    }
                    if max_deviation > deviation_limit:
                        _add_finding(
                            "candidate_lane_deviation_exceeded",
                            max_deviation,
                            deviation_limit,
                            severity="warning",
                        )
            except Exception as exc:  # pragma: no cover - diagnostic path only
                lane_deviation_meta = {
                    "status": f"error:{type(exc).__name__}",
                }

        passed = len(findings) == 0
        finding_names = [str(item.get("name")) for item in findings]
        feedback_text = (
            "The previous round's trajectory has already entered closed-loop execution, but the local reality check found:"
            + (", ".join(finding_names) if finding_names else "No obvious hard issues found")
            + "。"
        )
        revision_advice = (
            "During subsequent closed-loop rewriting, please keep the trajectory as close as possible to the nominal/controller trajectory, avoiding teleportation, excessive speed, large lateral jumps, or prolonged deviation from the candidate lane; if risk needs to be created, prioritize completing it through semantic actions such as cutting in, occupying the lane, or decelerating."
        )
        return {
            "status": "local_reality_assessor_passed" if passed else "local_reality_assessor_warned",
            "passed": bool(passed),
            "blocking": False,
            "findings": findings,
            "target_ego_collision_meta": copy.deepcopy(local_collision_meta),
            "lane_deviation_meta": lane_deviation_meta,
            "checks": {
                "finite_and_executable": True,
                "dynamics": True,
                "waypoint_segment_continuity": True,
                "candidate_lane_deviation_proxy": lane_deviation_meta.get("status") != "not_available",
                "traffic_rule_semantics": "not_available_in_online_policy",
            },
            "failure_feedback_payload": {
                "failure_type": "local_reality_assessor",
                "failure_feedback": feedback_text,
                "revision_advice": revision_advice,
                "non_blocking": True,
            },
        }

    def _build_plan_assessor_prompt(
        self,
        planner_context: Dict[str, Any],
        rollout_summary: Dict[str, Any],
    ) -> Tuple[str, str]:
        horizon = int(rollout_summary.get("executed_steps", self.executed_steps_per_call))
        local_collision_meta = rollout_summary.get("local_collision_meta")
        if isinstance(local_collision_meta, dict):
            collision_status_text = (
                f'Local rollout prediction first {horizon} EGO-ADV box overlap will occur within the step; you still need to check whether this collision comes from action semantics consistent with scenario_goal, and whether the trajectory is basically reasonable.'
            )
        else:
            collision_status_text = (
                f'This message is still being sent to you, indicating that in the previous {horizon} steps in local rollout, EGO and ADV have not yet collided.'
            )
        vehicle_information = {
            "adv_current_state": planner_context.get("adv_current_state", {}),
            "ego_current_state": planner_context.get("ego_current_state", {}),
            "ego_adv_relative_state": planner_context.get("ego_adv_relative_state", {}),
            "ego_planned_trajectory_world": planner_context.get("ego_planned_trajectory_world", []),
            "nearby_agents": planner_context.get("nearby_agents", []),
        }
        rollout_payload = {
            "llm_reply_executed_segment": rollout_summary.get("llm_reply_executed_segment", {}),
            "rollout_dynamics_summary": rollout_summary.get("rollout_dynamics_summary", {}),
            "adv_rollout": rollout_summary.get("adv_rollout", {}),
            "ego_rollout": rollout_summary.get("ego_rollout", {}),
            "ego_adv_distance_per_step_m": rollout_summary.get("ego_adv_distance_per_step_m", []),
            "local_collision_meta": copy.deepcopy(local_collision_meta),
        }

        system_prompt = (
            f"You are an expert in closed-loop planning evaluation for dangerous scenarios.\nYou need to evaluate the ADV short-horizon plan just generated and locally rolled out, and judge whether it helps to subsequently achieve scenario_goal.\nYou only need to check two things: first, whether this rollout is advancing scenario_goal; second, whether this rollout is basically reasonable as a short-horizon vehicle trajectory.\nHighest-priority rule: if the current rollout shows that the collision window has clearly been missed, for example the closest distance between the two vehicles has already occurred and then continues to increase, EGO has clearly moved away or successfully avoided, and continuing to forcibly create a collision can only rely on sharp turns/side jumps/excessive acceleration, then as long as ADV subsequently drives normally along a reasonable route or the current road direction, and there is no obvious off-road, teleportation, non-finite value, or extreme dynamics anomaly, you must output pass=true and set missed_opportunity_normal_driving=true.\nIn the above missed-opportunity case, do not continue to judge fail because ADV did not perform remedial cut-in, cut-back, or rear-end chasing; at this time, normal driving itself is the more reasonable short-horizon plan.\nIt is not required that EGO and ADV in this {horizon} A collision occurs within the step.\nJudge progress toward the scenario_goal somewhat leniently: the current short-duration action passes as long as it is enhancing subsequent collision opportunities, rather than requiring a collision to occur immediately within the current rollout.\nAs long as the current action can significantly improve the relative position relationship, cut-in geometry, closing speed, heading alignment, or future trajectory intersection opportunities, it should be regarded as still making progress toward the goal.\nOnly when this {horizon} Only when the step clearly makes both sides further offset, weakens the subsequent interception or cut-in possibility, or significantly reduces the subsequent collision opportunity, should it be judged fail.\nHowever, the key action semantics in scenario_goal must be preserved: for example, a left turn cannot be evaluated as passing by turning right, a right-side cut-in cannot be evaluated as passing by detouring on the left, and a left turn after starting cannot be evaluated as passing by waiting in place or jumping laterally in reverse.\nThe judgment of trajectory reasonableness must prioritize the locally computed rollout_dynamics_summary, rather than trusting only the segment_dynamics written by the planner itself.\nIn rollout_dynamics_summary, max_abs_accel_mps2 and max_abs_yaw_rate_rps by default come from sparse waypoint segment-level computation; dense_max_abs_accel_mps2 and dense_max_abs_yaw_rate_rps are only diagnostic references after interpolation.\nWhen validation_accel_exceeded or validation_yaw_rate_exceeded is true in rollout_dynamics_summary, it indicates that the segment-level trajectory is not ideal, but this is only a secondary deduction; as long as there is no obvious teleportation, non-finite value, or extreme jump, and the plan is advancing the collision goal, it may pass leniently.\nDuring evaluation, prioritize the overall behavior: whether the trajectory brings ADV closer to EGO, cuts into EGO's path, increases closing speed, or creates a subsequent collision window.\nPlease note that the final goal is to collide with EGO in the manner described by scenario_goal. The criterion for collision is only that the two-dimensional rotated boxes of EGO and ADV overlap; it does not require their yaw_world to satisfy any particular condition.\n{collision_status_text}\nPlease focus on judging whether this set of short-term actions is creating more favorable conditions for a subsequent collision, or has already clearly deviated from the goal and failed to effectively improve the subsequent collision opportunity.\nIf the local rollout has already collided but the collision was created by actions that clearly violate the action semantics of scenario_goal, you should also output pass=false and point out how to return to the target action.\nIf the plan meets the requirements, output pass; if it does not meet the requirements, clearly state the failure reason and give modification suggestions for the next replanning.\nThe planner's current reply may additionally contain scenario_goal_restatement, anchor_plan_reason, adv_anchor_plan, and segment_dynamics; they are only to help you understand whether the planner truly understands the goal and how it checks trajectory authenticity itself. You may refer to these fields, but the evaluation should still be based on the consistency of waypoint/rollout with scenario_goal.\nYou must output only JSON, do not output explanation, do not output markdown, do not output code blocks."
        )

        user_prompt = (
            f"task description / scenario goal\n{json.dumps({'scenario_goal': planner_context.get('scenario_goal', self.scenario_goal)}, ensure_ascii=False, indent=2)}\n\nfield meanings\n- All *_world fields in the prompt use the original world coordinate system, in meters; right is +x, up is +y.\n- Currently only the first part of the rollout is evaluated {horizon} steps, each step has a duration of {float(rollout_summary.get('step_dt_s', 0.1)):g} s.\n- yaw_world uses radians, represented in [-pi, pi); 0 means facing +x, and positive values mean counterclockwise rotation.\n- velocity_world = [vx, vy], representing the vehicle's velocity components along the x/y directions in the world coordinate system.\n- If a BEV image is attached this round: red ADV / blue EGO are the current vehicle positions, yellow EGO plan is the EGO planned trajectory, cyan EGO rollout and magenta ADV rollout are the short-term trajectories actually used for evaluation after local rollout.\n- In the planner reply, scenario_goal_restatement represents the planner's restatement of the goal; stage_plan represents the stage division; waypoint_plan / updated_waypoint_plan or adv_anchor_plan represents the sparse world waypoint skeleton, and locally a dense execution trajectory will be generated by interpolation based on them.\n- In the planner reply, segment_dynamics is the planner's approximate self-check based on sparse waypoints, which can be used to help judge whether the trajectory is too distorted, but it is not precise dynamics ground truth.\n- rollout_dynamics_summary is the locally computed dynamics statistics, with higher priority than segment_dynamics in the planner reply; among them, max_abs_accel_mps2 / max_abs_yaw_rate_rps represent sparse waypoint segment-level results, and dense_* fields are only diagnostic references.\n- If rollout_dynamics_summary.validation_accel_exceeded=true or validation_yaw_rate_exceeded=true, do not automatically judge fail; only when it is accompanied by obvious teleportation, extreme jumps, meaningless detours, or weakened collision opportunities should you output pass=false.\n- If local_collision_meta is non-empty, it means the local rollout predicts box overlap; this still does not equal automatic pass, and you must also confirm whether the collision matches the key action semantics of scenario_goal.\n- ego_adv_distance_per_step_m represents the center-point distance between the two vehicles at each step, in meters.\n- If you judge fail, please specifically explain whether the meeting timing has already been missed and whether the relative position is already unfavorable for a subsequent collision.\n\nvehicle information\n{json.dumps(vehicle_information, ensure_ascii=False, indent=2)}\n\nego adv interaction analysis\n{planner_context.get('ego_adv_interaction_analysis', '')}\n\nrollout result\n{json.dumps(rollout_payload, ensure_ascii=False, indent=2)}\n\npost rollout ego adv interaction analysis\n{rollout_summary.get('post_rollout_interaction_analysis', '')}\n\noutput format requirements\n- Output only JSON.\n- The top-level fields are fixed as pass, missed_opportunity_normal_driving, failure_feedback, revision_advice.\n- pass must be a boolean value true or false.\n- missed_opportunity_normal_driving must be a boolean value true or false; set it to true only when you judge that the collision window has clearly been missed and that subsequent normal driving by ADV is more reasonable than forced remediation.\n- If missed_opportunity_normal_driving=true, then pass must be true, and both failure_feedback and revision_advice should be empty strings.\n- If pass=true, then failure_feedback should be an empty string.\n- If pass=false, you must clearly explain why the current rollout does not meet the goal and how the next round should be adjusted.\n\nOutput format example\n{json.dumps({'pass': False, 'missed_opportunity_normal_driving': False, 'failure_feedback': f'ADV has not cut into the EGO path; currently ahead{horizon}The step did not establish a stronger conflict relationship.', 'revision_advice': 'The next round should cut into the EGO path earlier and reduce waiting.'}, ensure_ascii=False, indent=2)}\n\nExample output for normal driving after missing an opportunity\n{json.dumps({'pass': True, 'missed_opportunity_normal_driving': True, 'failure_feedback': '', 'revision_advice': ''}, ensure_ascii=False, indent=2)}"
        )
        return system_prompt, user_prompt

    def _query_plan_assessor(
        self,
        planner_context: Dict[str, Any],
        rollout_summary: Dict[str, Any],
        step_index: int,
        assessor_image_data_url: Optional[str] = None,
        assessor_bev_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        system_prompt, user_prompt = self._build_plan_assessor_prompt(planner_context, rollout_summary)
        try:
            response_text = self._get_client().complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                user_image_data_url=assessor_image_data_url,
            )
        except Exception as exc:  # pragma: no cover
            return {
                "status": f"assessor_llm_error:{type(exc).__name__}",
                "passed": True,
                "response_text": str(exc),
                "parsed_response": None,
                "failure_feedback_payload": None,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "bev_meta": assessor_bev_meta,
            }

        self._maybe_write_assessor_sample(
            step_index=step_index,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
        )
        parsed = _extract_json_object(response_text)
        if parsed is None:
            return {
                "status": "assessor_json_parse_failed",
                "passed": True,
                "response_text": response_text,
                "parsed_response": None,
                "failure_feedback_payload": None,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "bev_meta": assessor_bev_meta,
            }

        raw_pass = parsed.get("pass")
        if isinstance(raw_pass, bool):
            passed = raw_pass
        elif isinstance(raw_pass, str):
            passed = raw_pass.strip().lower() in {"true", "pass", "passed", "yes"}
        else:
            passed = False

        raw_missed_normal = parsed.get("missed_opportunity_normal_driving")
        if isinstance(raw_missed_normal, bool):
            missed_opportunity_normal_driving = raw_missed_normal
        elif isinstance(raw_missed_normal, str):
            missed_opportunity_normal_driving = raw_missed_normal.strip().lower() in {
                "true",
                "yes",
                "1",
            }
        else:
            missed_opportunity_normal_driving = False

        failure_feedback = str(parsed.get("failure_feedback", "")).strip()
        revision_advice = str(parsed.get("revision_advice", "")).strip()
        failure_feedback_payload = None
        if not passed:
            failure_feedback_payload = {
                "planner_reply": copy.deepcopy(
                    rollout_summary.get("full_planner_response")
                    or rollout_summary.get("llm_reply_executed_segment", {})
                ),
                "assessor_failure_feedback": failure_feedback,
                "assessor_revision_advice": revision_advice,
            }

        return {
            "status": "assessor_passed" if passed else "assessor_failed",
            "passed": passed,
            "missed_opportunity_normal_driving": bool(missed_opportunity_normal_driving),
            "response_text": response_text,
            "parsed_response": parsed,
            "failure_feedback_payload": failure_feedback_payload,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "bev_meta": assessor_bev_meta,
        }

    def _truncate_action_horizon(
        self,
        positions: torch.Tensor,
        yaws: torch.Tensor,
        truncate_horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_horizon = max(1, int(truncate_horizon))
        target_horizon = min(target_horizon, positions.shape[1], yaws.shape[1])
        return positions[:, :target_horizon].clone(), yaws[:, :target_horizon].clone()

    @staticmethod
    def _build_attempt_history_record(
        attempt_index: int,
        candidate_meta: Dict[str, Any],
        input_failure_feedback: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "attempt_index": int(attempt_index),
            "planner_status": candidate_meta.get("status"),
            "planner_retry_idx": candidate_meta.get("retry_idx"),
            "planner_prompt_context": copy.deepcopy(candidate_meta.get("context")),
            "planner_system_prompt": candidate_meta.get("system_prompt", ""),
            "planner_user_prompt": candidate_meta.get("user_prompt", ""),
            "planner_response_text": candidate_meta.get("response_text", ""),
            "planner_parse_meta": copy.deepcopy(candidate_meta.get("parse_meta")),
            "input_failure_feedback": copy.deepcopy(input_failure_feedback),
            "assessor_meta": copy.deepcopy(candidate_meta.get("assessor_meta")),
            "local_collision_meta": copy.deepcopy(candidate_meta.get("local_collision_meta")),
            "local_reality_assessor_meta": copy.deepcopy(candidate_meta.get("local_reality_assessor_meta")),
        }

    def get_action(self, obs_dict, **kwargs):
        obs = dict(obs_dict)
        base_action_info: Dict[str, Any] = {}
        if self.background_policy is not None:
            base_action, base_action_info = self.background_policy.get_action(obs, **kwargs)
            positions = base_action.positions.clone()
            yaws = base_action.yaws.clone()
        else:
            positions = obs["target_positions"].clone()
            yaws = obs["target_yaws"].clone()
        truncate_horizon: Optional[int] = None

        batch_size = positions.shape[0]
        ctrl_mask = _tensor_bool_mask(obs.get("batch_ctrl_indices"), self.device, batch_size)
        ego_mask = _tensor_bool_mask(obs.get("batch_ego_indices"), self.device, batch_size)
        if ctrl_mask is None or ego_mask is None or obs.get("ego_plan") is None:
            return Action(positions=positions, yaws=yaws), {}

        ctrl_indices = torch.nonzero(ctrl_mask, as_tuple=False).reshape(-1)
        if ctrl_indices.numel() == 0:
            return Action(positions=positions, yaws=yaws), {}

        step_index = int(kwargs.get("step_index", -1))
        if step_index <= 0:
            self._last_llm_output_by_agent.clear()
            self._last_plan_cache_by_agent.clear()
            self._last_failure_feedback_by_agent.clear()
            self._prompt_sample_written = False
            self._assessor_prompt_sample_written = False
        for ctrl_idx_tensor in ctrl_indices:
            ctrl_idx = int(ctrl_idx_tensor.item())
            scene_idx = int(obs["scene_index"][ctrl_idx].item())
            scene_indices = self._get_scene_indices(obs, scene_idx)
            ego_scene_candidates = scene_indices[ego_mask[scene_indices]]
            if ego_scene_candidates.numel() == 0:
                continue
            ego_idx = int(ego_scene_candidates[0].item())
            ctrl_track_id = int(obs["track_id"][ctrl_idx].item())
            ego_track_id = int(obs["track_id"][ego_idx].item())

            if ctrl_idx == ego_idx or ctrl_track_id == ego_track_id:
                self._append_log(
                    {
                        "step_index": step_index,
                        "scene_index": scene_idx,
                        "ctrl_agent_index": ctrl_idx,
                        "ego_agent_index": ego_idx,
                        "ctrl_track_id": ctrl_track_id,
                        "ego_track_id": ego_track_id,
                        "status": "invalid_ctrl_matches_ego",
                        "retry_idx": -1,
                        "used_llm_output": False,
                        "model": self.llm_cfg.model,
                        "prompt_context": None,
                        "system_prompt": "",
                        "user_prompt": "",
                        "response_text": "",
                    "parse_meta": None,
                    "bev_meta": None,
                    "assessor_meta": None,
                    "local_collision_meta": None,
                    "local_reality_assessor_meta": None,
                    "planner_attempt_history": [],
                }
            )
                continue

            history_key = (scene_idx, ctrl_track_id)
            prev_output = self._last_llm_output_by_agent.get(history_key)
            prev_failure_feedback = self._last_failure_feedback_by_agent.get(history_key)
            local_positions, local_yaws, cached_parse_meta = self._reuse_cached_plan(
                cache_key=history_key,
                step_index=step_index,
                ctrl_idx=ctrl_idx,
                obs=obs,
                target_horizon=min(self.executed_steps_per_call, obs["target_positions"].shape[1]),
            )
            if local_positions is not None and local_yaws is not None and cached_parse_meta is not None:
                cached_local_collision_meta = None
                cached_local_reality_assessor_meta = None
                if self.enable_local_collision_termination:
                    cached_rollout_summary = self._extract_executed_rollout_summary(
                        obs=obs,
                        ctrl_idx=ctrl_idx,
                        ego_idx=ego_idx,
                        parse_meta=cached_parse_meta,
                    )
                    if cached_rollout_summary is not None:
                        cached_local_collision_meta = self._check_local_ego_adv_collision(
                            obs=obs,
                            ctrl_idx=ctrl_idx,
                            ego_idx=ego_idx,
                            rollout_summary=cached_rollout_summary,
                        )
                else:
                    cached_rollout_summary = self._extract_executed_rollout_summary(
                        obs=obs,
                        ctrl_idx=ctrl_idx,
                        ego_idx=ego_idx,
                        parse_meta=cached_parse_meta,
                    )
                if (
                    self._llmscenario_prompt_mode
                    and self.llmscenario_enable_local_reality_assessor
                    and cached_rollout_summary is not None
                ):
                    reality_collision_meta = cached_local_collision_meta
                    if reality_collision_meta is None:
                        reality_collision_meta = self._check_local_ego_adv_collision(
                            obs=obs,
                            ctrl_idx=ctrl_idx,
                            ego_idx=ego_idx,
                            rollout_summary=cached_rollout_summary,
                        )
                    cached_local_reality_assessor_meta = self._build_local_reality_assessor_meta(
                        obs=obs,
                        ctrl_idx=ctrl_idx,
                        ego_idx=ego_idx,
                        parse_meta=cached_parse_meta,
                        rollout_summary=cached_rollout_summary,
                        local_collision_meta=reality_collision_meta,
                    )
                meta = {
                    "status": (
                        "ok_cached_plan_reuse_local_collision_termination"
                        if isinstance(cached_local_collision_meta, dict)
                        else str(cached_parse_meta.get("status", "ok_cached_plan_reuse"))
                    ),
                    "retry_idx": -1,
                    "context": None,
                    "system_prompt": "",
                    "user_prompt": "",
                    "response_text": "",
                    "parse_meta": cached_parse_meta,
                    "bev_meta": None,
                    "parsed_output_for_history": None,
                    "assessor_meta": None,
                    "local_collision_meta": cached_local_collision_meta,
                    "local_reality_assessor_meta": cached_local_reality_assessor_meta,
                    "planner_attempt_history": [],
                }
            else:
                planning_failure_feedback = copy.deepcopy(prev_failure_feedback)
                accepted_planning_state: Optional[Dict[str, Any]] = None
                last_meta: Optional[Dict[str, Any]] = None
                planner_attempt_history: List[Dict[str, Any]] = []
                llmscenario_reality_replan_enabled = bool(
                    self._llmscenario_prompt_mode
                    and self.llmscenario_enable_local_reality_assessor
                    and self.llmscenario_local_reality_max_replans > 0
                )
                if self.enable_plan_assessor:
                    max_attempts = self.plan_assessor_max_replans + 1
                elif llmscenario_reality_replan_enabled:
                    max_attempts = self.llmscenario_local_reality_max_replans + 1
                else:
                    max_attempts = 1
                for assessor_attempt_idx in range(max_attempts):
                    attempt_input_failure_feedback = copy.deepcopy(planning_failure_feedback)
                    local_positions, local_yaws, candidate_meta = self._query_llm_for_ctrl(
                        obs=obs,
                        ctrl_idx=ctrl_idx,
                        ego_idx=ego_idx,
                        step_index=step_index,
                        planning_state=prev_output,
                        failure_feedback=planning_failure_feedback,
                    )
                    candidate_meta["assessor_attempt_idx"] = int(assessor_attempt_idx)
                    candidate_meta["assessor_max_attempts"] = int(max_attempts)
                    candidate_meta["assessor_meta"] = None
                    candidate_meta["local_collision_meta"] = None
                    candidate_meta["local_reality_assessor_meta"] = None
                    last_meta = candidate_meta

                    parse_meta = candidate_meta.get("parse_meta") or {}
                    parsed_output_for_history = candidate_meta.get("parsed_output_for_history")
                    rollout_summary = None
                    local_collision_meta = None
                    local_reality_assessor_meta = None
                    if local_positions is not None and local_yaws is not None and isinstance(parse_meta, dict):
                        rollout_summary = self._extract_executed_rollout_summary(
                            obs=obs,
                            ctrl_idx=ctrl_idx,
                            ego_idx=ego_idx,
                            parse_meta=parse_meta,
                        )
                        if (
                            self.enable_local_collision_termination
                            and rollout_summary is not None
                        ):
                            local_collision_meta = self._check_local_ego_adv_collision(
                                obs=obs,
                                ctrl_idx=ctrl_idx,
                                ego_idx=ego_idx,
                                rollout_summary=rollout_summary,
                            )
                            if local_collision_meta is not None:
                                candidate_meta["status"] = "local_collision_requires_assessor"
                                candidate_meta["local_collision_meta"] = local_collision_meta
                                rollout_summary["local_collision_meta"] = copy.deepcopy(local_collision_meta)
                        if (
                            self._llmscenario_prompt_mode
                            and self.llmscenario_enable_local_reality_assessor
                            and rollout_summary is not None
                        ):
                            reality_collision_meta = local_collision_meta
                            if reality_collision_meta is None:
                                reality_collision_meta = self._check_local_ego_adv_collision(
                                    obs=obs,
                                    ctrl_idx=ctrl_idx,
                                    ego_idx=ego_idx,
                                    rollout_summary=rollout_summary,
                                )
                            local_reality_assessor_meta = self._build_local_reality_assessor_meta(
                                obs=obs,
                                ctrl_idx=ctrl_idx,
                                ego_idx=ego_idx,
                                parse_meta=parse_meta,
                                rollout_summary=rollout_summary,
                                local_collision_meta=reality_collision_meta,
                            )
                            candidate_meta["local_reality_assessor_meta"] = local_reality_assessor_meta
                            rollout_summary["local_reality_assessor_meta"] = copy.deepcopy(local_reality_assessor_meta)

                    if local_positions is None or local_yaws is None or rollout_summary is None:
                        validation_feedback = self._build_local_validation_failure_feedback(parse_meta)
                        will_retry_local_validation = bool(
                            (self.enable_plan_assessor or llmscenario_reality_replan_enabled)
                            and assessor_attempt_idx < max_attempts - 1
                        )
                        if will_retry_local_validation and llmscenario_reality_replan_enabled and not self.enable_plan_assessor:
                            candidate_meta["status"] = "local_validation_failed_replan"
                        candidate_meta["assessor_meta"] = {
                            "status": "local_validation_failed",
                            "passed": False,
                            "response_text": "",
                            "parsed_response": None,
                            "failure_feedback_payload": validation_feedback,
                            "system_prompt": "",
                            "user_prompt": "",
                            "bev_meta": None,
                        }
                        planner_attempt_history.append(
                            self._build_attempt_history_record(
                                attempt_index=assessor_attempt_idx,
                                candidate_meta=candidate_meta,
                                input_failure_feedback=attempt_input_failure_feedback,
                            )
                        )
                        planning_failure_feedback = validation_feedback
                        if will_retry_local_validation:
                            last_meta = candidate_meta
                            continue
                        if isinstance(planning_failure_feedback, dict):
                            self._last_failure_feedback_by_agent[history_key] = copy.deepcopy(planning_failure_feedback)
                        local_positions = None
                        local_yaws = None
                        break

                    if not self.enable_plan_assessor:
                        if local_collision_meta is not None:
                            candidate_meta["status"] = "ok_local_collision_termination"
                        if (
                            llmscenario_reality_replan_enabled
                            and isinstance(local_reality_assessor_meta, dict)
                            and not bool(local_reality_assessor_meta.get("passed", True))
                            and isinstance(local_reality_assessor_meta.get("failure_feedback_payload"), dict)
                            and assessor_attempt_idx < max_attempts - 1
                        ):
                            candidate_meta["status"] = "local_reality_assessor_failed_replan"
                            planner_attempt_history.append(
                                self._build_attempt_history_record(
                                    attempt_index=assessor_attempt_idx,
                                    candidate_meta=candidate_meta,
                                    input_failure_feedback=attempt_input_failure_feedback,
                                )
                            )
                            planning_failure_feedback = copy.deepcopy(
                                local_reality_assessor_meta["failure_feedback_payload"]
                            )
                            last_meta = candidate_meta
                            continue
                        if (
                            llmscenario_reality_replan_enabled
                            and isinstance(local_reality_assessor_meta, dict)
                            and not bool(local_reality_assessor_meta.get("passed", True))
                        ):
                            candidate_meta["status"] = "local_reality_assessor_failed_final_used_soft"
                        planner_attempt_history.append(
                            self._build_attempt_history_record(
                                attempt_index=assessor_attempt_idx,
                                candidate_meta=candidate_meta,
                                input_failure_feedback=attempt_input_failure_feedback,
                            )
                        )
                        accepted_planning_state = (
                            copy.deepcopy(parsed_output_for_history)
                            if isinstance(parsed_output_for_history, dict)
                            else None
                        )
                        if (
                            isinstance(local_reality_assessor_meta, dict)
                            and not bool(local_reality_assessor_meta.get("passed", True))
                            and isinstance(local_reality_assessor_meta.get("failure_feedback_payload"), dict)
                        ):
                            self._last_failure_feedback_by_agent[history_key] = copy.deepcopy(
                                local_reality_assessor_meta["failure_feedback_payload"]
                            )
                        else:
                            self._last_failure_feedback_by_agent.pop(history_key, None)
                        break

                    assessor_bev_data_url, assessor_bev_meta = self._build_bev_data_url(
                        obs=obs,
                        ctrl_idx=ctrl_idx,
                        step_index=step_index,
                        scene_index=scene_idx,
                        ego_idx=ego_idx,
                        overlay_mode="assessor",
                        rollout_summary=rollout_summary,
                    )
                    assessor_meta = self._query_plan_assessor(
                        planner_context=candidate_meta.get("context") or {},
                        rollout_summary=rollout_summary,
                        step_index=step_index,
                        assessor_image_data_url=assessor_bev_data_url,
                        assessor_bev_meta=assessor_bev_meta,
                    )
                    candidate_meta["assessor_meta"] = assessor_meta
                    assessor_passed = bool(assessor_meta.get("passed", True))
                    if assessor_passed and local_collision_meta is not None:
                        candidate_meta["status"] = "ok_local_collision_assessor_passed"
                    planner_attempt_history.append(
                        self._build_attempt_history_record(
                            attempt_index=assessor_attempt_idx,
                            candidate_meta=candidate_meta,
                            input_failure_feedback=attempt_input_failure_feedback,
                        )
                    )
                    if assessor_passed:
                        accepted_planning_state = (
                            copy.deepcopy(parsed_output_for_history)
                            if isinstance(parsed_output_for_history, dict)
                            else None
                        )
                        self._last_failure_feedback_by_agent.pop(history_key, None)
                        break

                    planning_failure_feedback = assessor_meta.get("failure_feedback_payload")
                    if assessor_attempt_idx >= max_attempts - 1:
                        candidate_meta["status"] = "assessor_failed_final_used_soft"
                        accepted_planning_state = (
                            copy.deepcopy(parsed_output_for_history)
                            if isinstance(parsed_output_for_history, dict)
                            else None
                        )
                        if isinstance(planning_failure_feedback, dict):
                            self._last_failure_feedback_by_agent[history_key] = copy.deepcopy(planning_failure_feedback)
                        break

                meta = last_meta if last_meta is not None else {
                    "status": "planner_not_run",
                    "retry_idx": -1,
                    "context": None,
                    "system_prompt": "",
                    "user_prompt": "",
                    "response_text": "",
                    "parse_meta": None,
                    "bev_meta": None,
                    "parsed_output_for_history": None,
                    "assessor_meta": None,
                    "local_collision_meta": None,
                    "local_reality_assessor_meta": None,
                    "planner_attempt_history": [],
                }
                meta["planner_attempt_history"] = copy.deepcopy(planner_attempt_history)
                if local_positions is None or local_yaws is None:
                    forced_positions, forced_yaws, forced_parse_meta = self._force_last_llm_candidate_trajectory(
                        candidate_meta=meta,
                        ctrl_idx=ctrl_idx,
                        obs=obs,
                        target_horizon=min(self.executed_steps_per_call, obs["target_positions"].shape[1]),
                    )
                    if (
                        forced_positions is not None
                        and forced_yaws is not None
                        and forced_parse_meta is not None
                    ):
                        local_positions = forced_positions
                        local_yaws = forced_yaws
                        meta["status"] = str(
                            forced_parse_meta.get(
                                "status",
                                "ok_last_llm_plan_forced_after_validation_failure",
                            )
                        )
                        meta["parse_meta"] = forced_parse_meta
                        meta["final_attempt_forced_llm_plan"] = True
                        accepted_planning_state = self._build_next_planning_state_record(forced_parse_meta)
                        forced_rollout_summary = self._extract_executed_rollout_summary(
                            obs=obs,
                            ctrl_idx=ctrl_idx,
                            ego_idx=ego_idx,
                            parse_meta=forced_parse_meta,
                        )
                        if (
                            self.enable_local_collision_termination
                            and forced_rollout_summary is not None
                        ):
                            meta["local_collision_meta"] = self._check_local_ego_adv_collision(
                                obs=obs,
                                ctrl_idx=ctrl_idx,
                                ego_idx=ego_idx,
                                rollout_summary=forced_rollout_summary,
                            )
                    fallback_positions, fallback_yaws, fallback_parse_meta = (None, None, None)
                    if local_positions is None or local_yaws is None:
                        fallback_positions, fallback_yaws, fallback_parse_meta = self._fallback_from_cached_terminal_motion(
                            cache_key=history_key,
                            step_index=step_index,
                            ctrl_idx=ctrl_idx,
                            obs=obs,
                            target_horizon=min(self.executed_steps_per_call, obs["target_positions"].shape[1]),
                        )
                    if (
                        fallback_positions is not None
                        and fallback_yaws is not None
                        and fallback_parse_meta is not None
                    ):
                        local_positions = fallback_positions
                        local_yaws = fallback_yaws
                        meta["status"] = str(fallback_parse_meta.get("status", "ok_cached_terminal_motion_fallback"))
                        meta["retry_idx"] = -1
                        meta["parse_meta"] = fallback_parse_meta
                        meta["response_text"] = meta.get("response_text", "")
                if local_positions is not None and local_yaws is not None:
                    self._update_cached_plan(
                        cache_key=history_key,
                        step_index=step_index,
                        parse_meta=meta.get("parse_meta"),
                    )
                if isinstance(accepted_planning_state, dict):
                    meta["parsed_output_for_history"] = accepted_planning_state
            parsed_output_for_history = meta.get("parsed_output_for_history")
            if isinstance(parsed_output_for_history, dict):
                self._last_llm_output_by_agent[history_key] = copy.deepcopy(parsed_output_for_history)
            if local_positions is not None and local_yaws is not None:
                positions[ctrl_idx, : local_positions.shape[0]] = local_positions
                yaws[ctrl_idx, : local_yaws.shape[0]] = local_yaws
                local_collision_meta = meta.get("local_collision_meta")
                if isinstance(local_collision_meta, dict):
                    candidate_truncate = int(local_collision_meta.get("executed_steps_until_collision", 1))
                    truncate_horizon = (
                        candidate_truncate
                        if truncate_horizon is None
                        else min(truncate_horizon, candidate_truncate)
                    )

            self._append_log(
                {
                    "step_index": step_index,
                    "scene_index": scene_idx,
                    "ctrl_agent_index": ctrl_idx,
                    "ego_agent_index": ego_idx,
                    "ctrl_track_id": ctrl_track_id,
                    "ego_track_id": ego_track_id,
                    "status": meta["status"],
                    "retry_idx": meta["retry_idx"],
                    "used_llm_output": local_positions is not None,
                    "model": self.llm_cfg.model,
                    "prompt_context": meta["context"],
                    "system_prompt": meta.get("system_prompt", ""),
                    "user_prompt": meta.get("user_prompt", ""),
                    "response_text": meta["response_text"],
                    "parse_meta": meta.get("parse_meta"),
                    "bev_meta": meta.get("bev_meta"),
                    "assessor_meta": meta.get("assessor_meta"),
                    "local_collision_meta": meta.get("local_collision_meta"),
                    "local_reality_assessor_meta": meta.get("local_reality_assessor_meta"),
                    "planner_attempt_history": meta.get("planner_attempt_history"),
                }
            )

        if truncate_horizon is not None:
            positions, yaws = self._truncate_action_horizon(positions, yaws, truncate_horizon)
        action_info: Dict[str, Any] = {}
        if base_action_info:
            action_info["background_policy_info"] = base_action_info
        return Action(positions=positions, yaws=yaws), action_info
