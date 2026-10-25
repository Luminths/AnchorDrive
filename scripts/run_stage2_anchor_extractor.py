import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tbsim.configs.eval_config import EvaluationConfig

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None

try:
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    httpx = None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_api_key(
    api_keys_path: str,
    base_url: str,
    api_key_env: Optional[str],
    api_key_json_field: Optional[str],
) -> str:
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

    for env_name in env_candidates:
        if not env_name:
            continue
        api_key = os.getenv(env_name)
        if api_key:
            return api_key

    with open(api_keys_path, "r", encoding="utf-8") as f:
        key_config = json.load(f)
    if not isinstance(key_config, dict):
        raise ValueError(f"Expected JSON object in api_keys file: {api_keys_path}")
    for field_name in json_field_candidates:
        if not field_name:
            continue
        api_key = key_config.get(field_name)
        if api_key:
            return str(api_key)

    raise ValueError(
        f"Missing {provider_name} API key. Checked env vars {env_candidates} and json fields {json_field_candidates}."
    )


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


class ChatClient:
    def __init__(
        self,
        api_keys_path: str,
        base_url: str,
        model: str,
        temperature: float,
        timeout_s: float,
        api_key_env: Optional[str] = None,
        api_key_json_field: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
    ) -> None:
        if OpenAI is None:
            raise ModuleNotFoundError("openai is not installed (pip install openai)")

        api_key = _resolve_api_key(
            api_keys_path=api_keys_path,
            base_url=base_url,
            api_key_env=api_key_env,
            api_key_json_field=api_key_json_field,
        )
        extra_headers: Dict[str, str] = {}
        if http_referer:
            extra_headers["HTTP-Referer"] = str(http_referer)
        if app_title:
            extra_headers["X-Title"] = str(app_title)

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": _make_openai_timeout(timeout_s),
            "max_retries": 0,
        }
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        base_url_lower = str(base_url).strip().lower()
        if "openrouter.ai" in base_url_lower:
            self.provider_name = "openrouter"
        elif "deepseek.com" in base_url_lower:
            self.provider_name = "deepseek"
        else:
            self.provider_name = "dashscope"
        self.enable_thinking = bool(enable_thinking)
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None and int(thinking_budget) > 0 else None
        self.timeout_s = float(timeout_s)
        self.reasoning_effort = (
            str(reasoning_effort).strip().lower() if reasoning_effort is not None else ""
        )
        self.last_reasoning_content = ""

    @staticmethod
    def _response_to_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if isinstance(response.get("choices"), list) and response["choices"]:
                message = response["choices"][0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    return content
            for key in ("content", "text", "response"):
                value = response.get(key)
                if isinstance(value, str):
                    return value

        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        chunks.append(str(item.get("text", "")))
                    else:
                        text = getattr(item, "text", None)
                        if text is not None:
                            chunks.append(str(text))
                return "\n".join(chunk for chunk in chunks if chunk)

        return str(response)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
        # Keep reasoning_effort in the CLI/config for compatibility, but do not
        # forward it to generic OpenAI-compatible endpoints.
        return self._response_to_text(response)


def _build_payload(case_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_data.get("case_id"),
        "scene_id": case_data.get("scene_id"),
        "ctrl_track_id": case_data.get("ctrl_track_id"),
        "scenario_goal": case_data.get("scenario_goal"),
        "stage_plan": case_data.get("stage_plan"),
        "executed_waypoint_plan": case_data.get("executed_waypoint_plan"),
        "collision_events": case_data.get("collision_events"),
        "adv_rollout_traj": case_data.get("adv_rollout_traj"),
    }


def _build_prompts(case_data: Dict[str, Any]) -> Dict[str, str]:
    system_prompt = (
        "You are the Stage 2 ADV anchor extractor. Your goal is not to replicate Stage 1's sparse plan, but to provide constraints for subsequent diffusion to generate a more reasonable, smoother, and more realistic ADV trajectory. You are only allowed to extract anchors for the ADV; do not extract any anchors for background vehicles or the EGO. The output must be a JSON object containing only the fields anchor_extraction_reason and adv_anchor_plan."
    )

    user_payload = _build_payload(case_data)
    user_prompt = (
        f'Based on the Stage 2 case below, extract a moderate set of anchors for ADV.\nThe requirements are as follows:\n1. The purpose of the anchors is to help diffusion generate more reasonable, smoother, and more realistic driving trajectories, rather than retaining only a few event nodes.\n2. The anchors do not need to strictly copy exact points on the Stage 1 trajectory; small adjustments are allowed, as long as they are more conducive to obtaining a reasonable trajectory while not breaking the overall attack intent.\n3. The number of anchors must not be too small, otherwise diffusion will have too much freedom and easily lose the trajectory shape; nor too many, otherwise it will lock in unreasonable details from Stage 1.\n4. If a certain turn is completed too abruptly in Stage 1, do not mention only two points, "turn start" and "turn end." You may add anchors before the turn starts, after the turn ends, or both, as needed, to help diffusion generate a more natural steering process.\n5. Prioritize the ADV trajectory from the real rollout; use the waypoint plan only as auxiliary reference, and do not trust only the planning text.\n6. If collision_events is non-empty, it means Stage 1 locally detected that the EGO/ADV boxes already overlap; please ensure that adv_anchor_plan covers the vicinity of the collision event, and usually it should directly include the step and position_world of the corresponding collision event, to avoid the anchors stopping early before the collision frame.\n7. Each element in adv_anchor_plan must contain step, position_world, and reason; step must be strictly increasing, and must use step numbers from adv_rollout_traj or collision_events.\n8. position_world must be a world-coordinate array of length 2.\n9. Do not output any extra fields, do not output Markdown, and do not output code blocks.\n\nThe output format must be strictly:\n{{\n  "anchor_extraction_reason": "...",\n  "adv_anchor_plan": [\n    {{"step": 6, "position_world": [x, y], "reason": "..."}},\n    {{"step": 12, "position_world": [x, y], "reason": "..."}}\n  ]\n}}\n\nStage 2 case:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}'
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def _validate_anchor_response(case_data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    anchor_extraction_reason = payload.get("anchor_extraction_reason")
    if not isinstance(anchor_extraction_reason, str) or not anchor_extraction_reason.strip():
        raise ValueError("anchor_extraction_reason must be a non-empty string.")

    adv_anchor_plan = payload.get("adv_anchor_plan")
    if not isinstance(adv_anchor_plan, list) or not adv_anchor_plan:
        raise ValueError("adv_anchor_plan must be a non-empty list.")

    adv_rollout_traj = case_data.get("adv_rollout_traj")
    if not isinstance(adv_rollout_traj, list) or not adv_rollout_traj:
        raise ValueError("case_data.adv_rollout_traj must be a non-empty list.")
    valid_steps = {
        _as_int(item.get("step"))
        for item in adv_rollout_traj
        if isinstance(item, dict) and _as_int(item.get("step")) is not None
    }
    collision_events = case_data.get("collision_events")
    if isinstance(collision_events, list):
        valid_steps.update(
            _as_int(item.get("step"))
            for item in collision_events
            if isinstance(item, dict) and _as_int(item.get("step")) is not None
        )
    valid_steps.discard(None)
    if not valid_steps:
        raise ValueError("adv_rollout_traj/collision_events do not contain valid step indices.")
    sorted_valid_steps = sorted(int(step) for step in valid_steps)

    normalized_plan: List[Dict[str, Any]] = []
    normalization_warnings: List[str] = []
    prev_step: Optional[int] = None
    for item in adv_anchor_plan:
        if not isinstance(item, dict):
            raise ValueError("Each anchor must be a JSON object.")
        step = _as_int(item.get("step"))
        if step is None:
            raise ValueError("Each anchor step must be an integer.")
        raw_step = int(step)
        if raw_step not in valid_steps:
            candidates = [
                valid_step
                for valid_step in sorted_valid_steps
                if prev_step is None or valid_step > prev_step
            ]
            if not candidates:
                normalization_warnings.append(
                    f"Skipped anchor at invalid step {raw_step}; no later valid step was available."
                )
                continue
            step = min(candidates, key=lambda valid_step: (abs(valid_step - raw_step), valid_step))
            normalization_warnings.append(
                f"Snapped anchor step {raw_step} to nearest valid step {step}."
            )
        if prev_step is not None and step <= prev_step:
            candidates = [valid_step for valid_step in sorted_valid_steps if valid_step > prev_step]
            if not candidates:
                normalization_warnings.append(
                    f"Skipped anchor at step {step}; strict ordering after {prev_step} was impossible."
                )
                continue
            new_step = min(candidates, key=lambda valid_step: (abs(valid_step - raw_step), valid_step))
            normalization_warnings.append(
                f"Moved anchor step {step} to {new_step} to keep steps strictly increasing."
            )
            step = new_step
        prev_step = step

        position_world = item.get("position_world")
        if not isinstance(position_world, (list, tuple)) or len(position_world) != 2:
            raise ValueError("position_world must be a length-2 array.")
        x = _as_float(position_world[0])
        y = _as_float(position_world[1])
        if x is None or y is None:
            raise ValueError("position_world must contain numeric values.")

        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Each anchor must include a non-empty reason.")

        normalized_plan.append(
            {
                "step": step,
                "position_world": [x, y],
                "reason": reason.strip(),
            }
        )

    result = {
        "anchor_extraction_reason": anchor_extraction_reason.strip(),
        "adv_anchor_plan": normalized_plan,
    }
    if normalization_warnings:
        result["normalization_warnings"] = normalization_warnings
    return result


def main() -> None:
    default_cfg = EvaluationConfig()

    parser = argparse.ArgumentParser(description="Run the Stage 2 ADV anchor extractor on a single case JSON.")
    parser.add_argument("--case_path", type=str, required=True, help="Path to a Stage 2 case JSON.")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the extracted adv_anchor_plan JSON. Defaults to <case_path stem>.adv_anchor_plan.json",
    )
    parser.add_argument(
        "--dump_prompt_path",
        type=str,
        default=None,
        help="Optional path to save the exact prompts before sending to the LLM.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only dump the prompts and exit without calling the LLM API.",
    )
    parser.add_argument("--llm_api_keys_path", type=str, default=default_cfg.llm.api_keys_path)
    parser.add_argument("--llm_base_url", type=str, default=default_cfg.llm.base_url)
    parser.add_argument("--llm_api_key_env", type=str, default=default_cfg.llm.api_key_env)
    parser.add_argument("--llm_api_key_json_field", type=str, default=default_cfg.llm.api_key_json_field)
    parser.add_argument("--llm_http_referer", type=str, default=default_cfg.llm.http_referer)
    parser.add_argument("--llm_app_title", type=str, default=default_cfg.llm.app_title)
    parser.add_argument("--llm_model", type=str, default=default_cfg.llm.model)
    parser.add_argument("--llm_temperature", type=float, default=0.1)
    parser.add_argument(
        "--llm_reasoning_effort",
        type=str,
        default="",
        help="Optional reasoning_effort for providers that explicitly support it. Leave empty for generic OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--llm_enable_thinking",
        action="store_true",
        help="Forward DashScope-compatible enable_thinking=true and stream the response.",
    )
    parser.add_argument(
        "--llm_thinking_budget",
        type=int,
        default=None,
        help="Optional DashScope-compatible thinking token budget.",
    )
    parser.add_argument("--llm_timeout_s", type=float, default=default_cfg.llm.timeout_s)
    args = parser.parse_args()

    case_path = Path(args.case_path).expanduser().resolve()
    case_data = _load_json(case_path)
    prompts = _build_prompts(case_data)

    if args.dump_prompt_path:
        dump_prompt_path = Path(args.dump_prompt_path).expanduser().resolve()
        dump_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        dump_prompt_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved prompts to {dump_prompt_path}")

    if args.dry_run:
        print("Dry run enabled; skipped LLM call.")
        return

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else case_path.with_name(f"{case_path.stem}.adv_anchor_plan.json")
    )
    raw_output_path = output_path.with_suffix(f"{output_path.suffix}.raw.txt")

    client = ChatClient(
        api_keys_path=args.llm_api_keys_path,
        base_url=args.llm_base_url,
        model=args.llm_model,
        temperature=args.llm_temperature,
        timeout_s=args.llm_timeout_s,
        api_key_env=args.llm_api_key_env,
        api_key_json_field=args.llm_api_key_json_field,
        reasoning_effort=args.llm_reasoning_effort,
        enable_thinking=bool(args.llm_enable_thinking),
        thinking_budget=args.llm_thinking_budget,
        http_referer=args.llm_http_referer,
        app_title=args.llm_app_title,
    )
    response_text = client.complete(
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
    )
    raw_output_path.write_text(response_text, encoding="utf-8")
    print(f"Saved raw response to {raw_output_path}")

    parsed = _extract_json_object(response_text)
    if parsed is None:
        raise ValueError("Failed to parse a JSON object from the LLM response.")
    normalized = _validate_anchor_response(case_data=case_data, payload=parsed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved anchor plan to {output_path}")


if __name__ == "__main__":
    main()
