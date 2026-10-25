import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from tbsim.policies.base import Policy
from tbsim.utils.agent_rel_classify import find_conflict_point
from tbsim.utils.geometry_utils import transform_points_tensor


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


class Stage2DiffusionPolicy(Policy):
    """
    Lightweight wrapper that injects Stage 2 anchor targets into the diffusion policy input.

    The wrapped policy remains the actual diffusion model. This wrapper only augments the
    observation dict with per-agent anchor tensors, allowing the guidance code to stay inside
    the existing diffusion/guidance framework.
    """

    def __init__(self, policy: Policy, eval_config, case_dir: Optional[str] = None, require_anchor: bool = True):
        super().__init__(policy.device)
        self.policy = policy
        self.eval_config = eval_config
        self.nets = getattr(policy, "nets", None)
        self.stage2_cfg = getattr(eval_config, "stage2", None)
        self.mode = str(getattr(self.stage2_cfg, "mode", "case_anchor")).strip().lower()
        self.case_dir = Path(case_dir).expanduser().resolve() if case_dir else None
        self.require_anchor = bool(require_anchor)
        self._horizon = self._infer_horizon()
        self._inner_policy = self._infer_inner_policy()
        self._step_time = self._infer_step_time()
        self._dyn = self._infer_dynamics()
        self._guidance_names = self._infer_guidance_names()
        self._guidance_configs = self._infer_guidance_configs()
        self._trajectory_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._base_weights, self._base_ctrl_weights = self._infer_loss_weights()
        self._partial_enabled = self._infer_partial_enabled()
        self._anchor_registry: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if self.mode == "case_anchor":
            if self.case_dir is None:
                raise ValueError("stage2.mode=case_anchor requires stage2.case_dir.")
            self._anchor_registry = self._load_anchor_registry(self.case_dir, self.require_anchor)
        else:
            raise ValueError(f"Unsupported stage2.mode: {self.mode}")

    def eval(self):
        self.policy.eval()

    def unwrap(self):
        return self.policy.unwrap() if hasattr(self.policy, "unwrap") else self.policy

    def get_action(self, obs_dict, **kwargs):
        obs_aug = dict(obs_dict)
        step_index = int(kwargs.get("step_index", -1))
        obs_aug["rollout_step_index"] = step_index
        obs_aug.update(self._build_stage2_tensors(obs_aug, step_index))
        action, info = self.policy.get_action(obs_aug, **kwargs)
        self._update_cache(obs_aug, step_index, info)
        return action, info

    def update_guide_config(self, *args, **kwargs):
        if hasattr(self.policy, "update_guide_config"):
            return self.policy.update_guide_config(*args, **kwargs)
        return None

    def _infer_horizon(self) -> int:
        inner_policy = self._infer_inner_policy()
        if inner_policy is not None:
            diffuse_args = getattr(inner_policy, "diffuse_args", None)
            if diffuse_args is not None:
                num_points = diffuse_args.get("num_points")
                if num_points is not None:
                    return int(num_points)
        raise ValueError("Unable to infer diffusion horizon from wrapped policy.")

    def _infer_inner_policy(self):
        nets = getattr(self.policy, "nets", None)
        if nets is not None and "policy" in nets:
            return nets["policy"]
        raise ValueError("Unable to locate inner diffusion policy from wrapped policy.")

    def _infer_step_time(self) -> float:
        diffusion = getattr(self._inner_policy, "diffusion", None)
        net = getattr(diffusion, "net", None)
        step_time = getattr(net, "step_time", None)
        if step_time is None:
            raise ValueError("Unable to infer diffusion step_time from wrapped policy.")
        return float(step_time)

    def _infer_dynamics(self):
        diffusion = getattr(self._inner_policy, "diffusion", None)
        net = getattr(diffusion, "net", None)
        dyn = getattr(net, "dyn", None)
        if dyn is None:
            raise ValueError("Unable to infer diffusion dynamics model from wrapped policy.")
        return dyn

    def _infer_guidance_names(self) -> List[str]:
        guide_config = getattr(self._inner_policy, "guide_config", None)
        if guide_config is None:
            return []
        return list(getattr(guide_config, "guidance_fn", []))

    def _infer_guidance_configs(self) -> Dict[str, Dict[str, Any]]:
        guide_config = getattr(self._inner_policy, "guide_config", None)
        configs = getattr(guide_config, "configs", None)
        if isinstance(configs, dict):
            return configs
        guidance_configs = getattr(guide_config, "guidance_configs", None)
        if isinstance(guidance_configs, dict):
            return guidance_configs
        return {}

    def _infer_loss_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        guide_config = getattr(self._inner_policy, "guide_config", None)
        if guide_config is None:
            return None, None
        combine_cfg = getattr(guide_config, "combineloss_config", None)
        if combine_cfg is None:
            return None, None
        weights = combine_cfg.get("weights")
        ctrl_weights = combine_cfg.get("ctrl_weights")
        weights_tensor = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        ctrl_weights_tensor = torch.tensor(ctrl_weights, dtype=torch.float32) if ctrl_weights is not None else None
        return weights_tensor, ctrl_weights_tensor

    def _infer_partial_enabled(self) -> bool:
        guide_config = getattr(self._inner_policy, "guide_config", None)
        if guide_config is None:
            return False
        params = getattr(guide_config, "params", None)
        partial_t = params.get("partial_t") if params is not None and hasattr(params, "get") else None
        return partial_t is not None

    @staticmethod
    def _load_anchor_registry(case_dir: Path, require_anchor: bool) -> Dict[Tuple[int, int], Dict[str, Any]]:
        if not case_dir.exists():
            raise FileNotFoundError(f"Stage 2 case directory does not exist: {case_dir}")

        registry: Dict[Tuple[int, int], Dict[str, Any]] = {}
        case_paths = sorted(case_dir.glob("*.json"))
        for case_path in case_paths:
            if case_path.name == "index.json" or case_path.name.endswith(".adv_anchor_plan.json") or case_path.name.endswith(".anchor_prompt.json"):
                continue
            case_payload = _load_json(case_path)
            if case_payload.get("schema_version") != "stage2_case_v1":
                continue

            anchor_path = case_path.with_name(f"{case_path.stem}.adv_anchor_plan.json")
            if not anchor_path.exists():
                if require_anchor:
                    raise FileNotFoundError(
                        f"Missing anchor file for Stage 2 case {case_path.name}: expected {anchor_path.name}"
                    )
                continue

            anchor_payload = _load_json(anchor_path)
            adv_anchor_plan = anchor_payload.get("adv_anchor_plan")
            if not isinstance(adv_anchor_plan, list):
                raise ValueError(f"Invalid adv_anchor_plan in {anchor_path}")

            scene_index = _as_int(case_payload.get("scene_index"))
            ctrl_track_id = _as_int(case_payload.get("ctrl_track_id"))
            if scene_index is None or ctrl_track_id is None:
                raise ValueError(f"Missing scene_index/ctrl_track_id in {case_path}")

            registry[(scene_index, ctrl_track_id)] = {
                "case_id": case_payload.get("case_id"),
                "scene_id": case_payload.get("scene_id"),
                "case_path": str(case_path),
                "anchor_path": str(anchor_path),
                "adv_anchor_plan": adv_anchor_plan,
            }
        return registry

    def _build_stage2_tensors(self, obs: Dict[str, Any], step_index: int) -> Dict[str, Any]:
        batch_size = int(obs["track_id"].shape[0])
        horizon = self._horizon
        device = obs["centroid"].device
        dtype = obs["centroid"].dtype

        anchor_positions_world = torch.zeros(batch_size, horizon, 2, device=device, dtype=dtype)
        anchor_valid_mask = torch.zeros(batch_size, horizon, device=device, dtype=torch.bool)
        dynamic_batch_weights = None
        adv_proposals_dict = None

        ctrl_mask_raw = obs.get("batch_ctrl_indices")
        if ctrl_mask_raw is None:
            return {
                "adv_anchor_positions_world": anchor_positions_world,
                "adv_anchor_valid_mask": anchor_valid_mask,
            }
        ctrl_mask = torch.as_tensor(ctrl_mask_raw, device=device).bool().flatten()
        if ctrl_mask.numel() != batch_size:
            return {
                "adv_anchor_positions_world": anchor_positions_world,
                "adv_anchor_valid_mask": anchor_valid_mask,
            }

        if self._should_use_dynamic_weights() and self._base_weights is not None:
            dynamic_batch_weights = self._base_weights.to(device).repeat(batch_size, 1)
            if self._base_ctrl_weights is not None:
                dynamic_batch_weights[ctrl_mask] = self._base_ctrl_weights.to(device)

        proposal_actions: List[torch.Tensor] = []
        proposal_states: List[torch.Tensor] = []
        proposal_ctrl_indices: List[int] = []

        ctrl_indices = torch.nonzero(ctrl_mask, as_tuple=False).reshape(-1)
        for ctrl_idx_tensor in ctrl_indices:
            ctrl_idx = int(ctrl_idx_tensor.item())
            entry = self._resolve_stage2_entry(obs=obs, ctrl_idx=ctrl_idx, step_index=step_index)
            if entry is None:
                continue

            future_anchors = []
            for anchor_item in entry["adv_anchor_plan"]:
                if not isinstance(anchor_item, dict):
                    continue
                anchor_step = _as_int(anchor_item.get("step"))
                position_world = anchor_item.get("position_world")
                if anchor_step is None or not isinstance(position_world, (list, tuple)) or len(position_world) != 2:
                    continue
                rel_step = int(anchor_step) - int(step_index) - 1
                future_anchors.append(
                    {
                        "step": int(anchor_step),
                        "position_world": [float(position_world[0]), float(position_world[1])],
                        "rel_step": rel_step,
                    }
                )
                if rel_step < 0 or rel_step >= horizon:
                    continue

                anchor_positions_world[ctrl_idx, rel_step, 0] = float(position_world[0])
                anchor_positions_world[ctrl_idx, rel_step, 1] = float(position_world[1])
                anchor_valid_mask[ctrl_idx, rel_step] = True

            if dynamic_batch_weights is not None:
                dynamic_batch_weights[ctrl_idx] = self._compute_dynamic_weights(
                    step_index=step_index,
                    future_anchors=future_anchors,
                    obs=obs,
                    ctrl_idx=ctrl_idx,
                    anchor_valid_mask=anchor_valid_mask[ctrl_idx],
                    device=device,
                )

            if self._partial_enabled:
                proposal = self._build_partial_proposal(
                    obs=obs,
                    ctrl_idx=ctrl_idx,
                    step_index=step_index,
                    future_anchors=future_anchors,
                )
                if proposal is not None:
                    proposal_actions.append(proposal["actions"])
                    proposal_states.append(proposal["states"])
                    proposal_ctrl_indices.append(ctrl_idx)

        output = {
            "adv_anchor_positions_world": anchor_positions_world,
            "adv_anchor_valid_mask": anchor_valid_mask,
        }
        if dynamic_batch_weights is not None:
            output["dynamic_batch_weights"] = dynamic_batch_weights
        if proposal_actions:
            proposal_actions_tensor = torch.stack(proposal_actions, dim=0).unsqueeze(1)
            proposal_states_tensor = torch.stack(proposal_states, dim=0).unsqueeze(1)
            adv_idx_tensor = torch.as_tensor(proposal_ctrl_indices, device=device, dtype=torch.long)
            padded_actions = self._create_padded_tensor(batch_size, adv_idx_tensor, proposal_actions_tensor)
            padded_positions = self._create_padded_tensor(batch_size, adv_idx_tensor, proposal_states_tensor[..., :2])
            adv_proposals_dict = {
                "adv_proposals": proposal_actions_tensor,
                "adv_proposals_states": proposal_states_tensor,
                "adv_idx": adv_idx_tensor,
                "padded_adv_collision_actions": padded_actions,
                "padded_adv_proposals_pos": padded_positions,
            }
            output["adv_proposals_dict"] = adv_proposals_dict
        return output



    def _resolve_stage2_entry(
        self,
        obs: Dict[str, Any],
        ctrl_idx: int,
        step_index: int,
    ) -> Optional[Dict[str, Any]]:
        scene_index = int(obs["scene_index"][ctrl_idx].item())
        ctrl_track_id = int(obs["track_id"][ctrl_idx].item())
        return self._anchor_registry.get((scene_index, ctrl_track_id))






    @staticmethod
    def _create_padded_tensor(batch_size: int, adv_idx: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        padded_shape = (batch_size, *proposals.shape[1:])
        padded_tensor = torch.zeros(padded_shape, dtype=proposals.dtype, device=proposals.device)
        padded_tensor[adv_idx] = proposals
        return padded_tensor

    def _should_use_dynamic_weights(self) -> bool:
        return True

    def _compute_dynamic_weights(
        self,
        step_index: int,
        future_anchors: List[Dict[str, Any]],
        obs: Dict[str, Any],
        ctrl_idx: int,
        anchor_valid_mask: torch.Tensor,
        device,
    ) -> torch.Tensor:
        if self._base_weights is None:
            return None

        weights = self._base_ctrl_weights.clone() if self._base_ctrl_weights is not None else self._base_weights.clone()
        if "advanchor" not in self._guidance_names or "causecollision" not in self._guidance_names:
            return weights.to(device)

        anchor_idx = self._guidance_names.index("advanchor")
        collision_idx = self._guidance_names.index("causecollision")
        route_idx = self._guidance_names.index("route") if "route" in self._guidance_names else None
        all_anchor_steps = [item["step"] for item in future_anchors]
        anchor_active = bool(anchor_valid_mask.bool().any().item())
        cause_active = self._is_causecollision_active(obs=obs, ctrl_idx=ctrl_idx)

        # While either task-specific loss still has signal, keep ADV guidance on
        # advanchor/causecollision only. If both are exhausted, transfer that mass
        # to route so the controlled vehicle continues with a sane drivable path.
        allowed_indices = {anchor_idx, collision_idx}
        if route_idx is not None:
            allowed_indices.add(route_idx)
        for idx in range(weights.numel()):
            if idx not in allowed_indices:
                weights[idx] = 0.0

        transfer_mass = float(weights[anchor_idx].item() + weights[collision_idx].item())
        if transfer_mass <= 0.0 and self._base_ctrl_weights is not None:
            transfer_mass = float(
                self._base_ctrl_weights[anchor_idx].item()
                + self._base_ctrl_weights[collision_idx].item()
            )

        weights[anchor_idx] = 0.0
        weights[collision_idx] = 0.0
        if route_idx is not None:
            weights[route_idx] = 0.0

        if not anchor_active and not cause_active:
            if route_idx is not None:
                route_base = (
                    float(self._base_ctrl_weights[route_idx].item())
                    if self._base_ctrl_weights is not None
                    else float(self._base_weights[route_idx].item())
                )
                weights[route_idx] = max(route_base, transfer_mass)
            return weights.to(device)

        if not all_anchor_steps:
            if cause_active:
                weights[collision_idx] = transfer_mass
            elif anchor_active:
                weights[anchor_idx] = transfer_mass
            return weights.to(device)

        progress = self._anchor_schedule_progress(step_index, all_anchor_steps)

        if anchor_active and cause_active:
            weights[anchor_idx] = transfer_mass * (1.0 - progress)
            weights[collision_idx] = transfer_mass * progress
        elif anchor_active:
            weights[anchor_idx] = transfer_mass
        elif cause_active:
            weights[collision_idx] = transfer_mass
        return weights.to(device)

    @staticmethod
    def _anchor_schedule_progress(step_index: int, anchor_steps: List[int]) -> float:
        if not anchor_steps:
            return 1.0
        last_anchor_step = max(anchor_steps)
        valid_anchor_steps = [step for step in anchor_steps if step > step_index]
        if len(valid_anchor_steps) == 0:
            return 1.0
        next_anchor_step = min(valid_anchor_steps)
        progress = float(next_anchor_step) / float(max(last_anchor_step, 1))
        return float(max(0.0, min(1.0, progress)))

    @staticmethod
    def _nearest_polyline_idx(polyline: torch.Tensor, point: torch.Tensor) -> Optional[int]:
        if polyline is None or polyline.ndim != 2 or polyline.shape[0] == 0:
            return None
        distances = torch.norm(polyline - point.unsqueeze(0), dim=-1)
        if not torch.isfinite(distances).any():
            return None
        return int(torch.argmin(distances).item())

    def _is_causecollision_active(self, obs: Dict[str, Any], ctrl_idx: int) -> bool:
        if "causecollision" not in self._guidance_names:
            return False

        cfg = self._guidance_configs.get("causecollision", {})
        interact_mode = str(cfg.get("interact_mode", "distance"))
        interact_dist_thresh = float(cfg.get("interact_dist_thresh", 100.0))
        scene_index = int(obs["scene_index"][ctrl_idx].item())
        ctrl_track_id = int(obs["track_id"][ctrl_idx].item())

        ego_mask_raw = obs.get("batch_ego_indices")
        if ego_mask_raw is None:
            return True
        ego_mask = torch.as_tensor(ego_mask_raw, device=obs["centroid"].device).bool().flatten()
        same_scene = obs["scene_index"] == scene_index
        ego_candidates = torch.nonzero(ego_mask & same_scene, as_tuple=False).reshape(-1)
        if ego_candidates.numel() == 0:
            return True
        ego_idx = int(ego_candidates[0].item())
        if ctrl_idx == ego_idx or int(obs["track_id"][ego_idx].item()) == ctrl_track_id:
            return False

        extras = obs.get("extras", {})
        lane_avail = extras.get("has_lane")
        centerline = extras.get("centerline_xy")
        if (
            interact_mode == "conflict"
            and lane_avail is not None
            and centerline is not None
            and bool(lane_avail[ego_idx].item() if isinstance(lane_avail[ego_idx], torch.Tensor) else lane_avail[ego_idx])
            and bool(lane_avail[ctrl_idx].item() if isinstance(lane_avail[ctrl_idx], torch.Tensor) else lane_avail[ctrl_idx])
        ):
            try:
                world_from_agent = obs["world_from_agent"].reshape(-1, 3, 3)
                ego_centerline = transform_points_tensor(centerline[ego_idx], world_from_agent[ego_idx])
                ctrl_centerline = transform_points_tensor(centerline[ctrl_idx], world_from_agent[ctrl_idx])
                conflict_point, ego_conflict_idx, ctrl_conflict_idx = find_conflict_point(
                    ego_centerline,
                    ctrl_centerline,
                )
                if conflict_point is not None:
                    ego_lane_idx = self._nearest_polyline_idx(ego_centerline, obs["centroid"][ego_idx])
                    ctrl_lane_idx = self._nearest_polyline_idx(ctrl_centerline, obs["centroid"][ctrl_idx])
                    if ego_lane_idx is None or ctrl_lane_idx is None:
                        return True
                    ego_before = ego_lane_idx < int(ego_conflict_idx)
                    ctrl_before = ctrl_lane_idx < int(ctrl_conflict_idx)
                    if not (ego_before or ctrl_before):
                        return False
                    ego_dist_to_conflict = float(torch.norm(obs["centroid"][ego_idx] - conflict_point).item())
                    ctrl_dist_to_conflict = float(torch.norm(obs["centroid"][ctrl_idx] - conflict_point).item())
                    return min(ego_dist_to_conflict, ctrl_dist_to_conflict) < interact_dist_thresh
            except Exception:
                return True

        current_distance = float(torch.norm(obs["centroid"][ego_idx] - obs["centroid"][ctrl_idx]).item())
        if current_distance < interact_dist_thresh:
            return True

        ego_plan = obs.get("ego_plan")
        if ego_plan is None:
            return False
        try:
            ego_plan = ego_plan.to(obs["centroid"].device)
            if ego_plan.ndim == 2:
                ego_plan = ego_plan.unsqueeze(0)
            ego_indices = obs.get("ego_idx")
            if ego_indices is not None:
                ego_indices_list = ego_indices.detach().cpu().tolist() if isinstance(ego_indices, torch.Tensor) else list(ego_indices)
                if ego_idx in ego_indices_list:
                    row = ego_indices_list.index(ego_idx)
                    ego_plan = ego_plan[row : row + 1]
            ego_plan_world = transform_points_tensor(ego_plan[0], obs["world_from_agent"][ego_idx])
            min_future_distance = float(
                torch.norm(ego_plan_world - obs["centroid"][ctrl_idx].unsqueeze(0), dim=-1).min().item()
            )
            return min_future_distance < interact_dist_thresh
        except Exception:
            return False

    def _build_partial_proposal(
        self,
        obs: Dict[str, Any],
        ctrl_idx: int,
        step_index: int,
        future_anchors: List[Dict[str, Any]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        scene_index = int(obs["scene_index"][ctrl_idx].item())
        ctrl_track_id = int(obs["track_id"][ctrl_idx].item())
        cache_key = (scene_index, ctrl_track_id)
        cache_entry = self._trajectory_cache.get(cache_key)

        proposal_actions = torch.zeros(self._horizon, 2, device=self.device, dtype=torch.float32)
        suffix_len = 0
        if cache_entry is not None:
            prev_step_index = int(cache_entry["rollout_step_index"])
            prev_controls = cache_entry["selected_controls"].to(self.device)
            executed_steps = max(step_index - prev_step_index, 0)
            if executed_steps < prev_controls.shape[0]:
                suffix_controls = prev_controls[executed_steps:]
                suffix_len = min(int(suffix_controls.shape[0]), self._horizon)
                proposal_actions[:suffix_len] = suffix_controls[:suffix_len]

        current_speed = float(obs["curr_speed"][ctrl_idx].item())
        current_state = torch.tensor([0.0, 0.0, current_speed, 0.0], device=self.device, dtype=torch.float32)

        if suffix_len > 0:
            suffix_states = self._rollout_controls(current_state, proposal_actions[:suffix_len])
            current_state = suffix_states[-1]

        local_future_anchors = self._prepare_local_future_anchors(obs, ctrl_idx, future_anchors, step_index, suffix_len)
        if suffix_len < self._horizon:
            extension_actions, extension_states = self._extend_controls_to_targets(
                start_state=current_state,
                local_future_anchors=local_future_anchors,
                existing_prefix_len=suffix_len,
            )
            proposal_actions[suffix_len:] = extension_actions
            if suffix_len > 0:
                full_states = torch.cat([suffix_states, extension_states], dim=0)
            else:
                full_states = extension_states
        else:
            full_states = suffix_states

        if full_states.shape[0] < self._horizon:
            pad_actions, pad_states = self._build_constant_velocity_tail(
                start_state=full_states[-1] if full_states.shape[0] > 0 else current_state,
                tail_length=self._horizon - full_states.shape[0],
            )
            proposal_actions[full_states.shape[0] :] = pad_actions
            full_states = torch.cat([full_states, pad_states], dim=0) if full_states.shape[0] > 0 else pad_states

        return {
            "actions": proposal_actions,
            "states": full_states[: self._horizon],
        }

    def _prepare_local_future_anchors(
        self,
        obs: Dict[str, Any],
        ctrl_idx: int,
        future_anchors: List[Dict[str, Any]],
        step_index: int,
        suffix_len: int,
    ) -> List[Dict[str, Any]]:
        if not future_anchors:
            return []
        anchor_world = torch.tensor(
            [item["position_world"] for item in future_anchors],
            device=self.device,
            dtype=torch.float32,
        )
        anchor_local = self._world_to_agent_points(obs, ctrl_idx, anchor_world)
        local_future_anchors: List[Dict[str, Any]] = []
        for item, anchor_xy_local in zip(future_anchors, anchor_local):
            rel_step = int(item["step"]) - int(step_index) - 1
            if rel_step < suffix_len:
                continue
            local_future_anchors.append(
                {
                    "step": int(item["step"]),
                    "rel_step": int(rel_step),
                    "position_local": anchor_xy_local,
                }
            )
        return local_future_anchors

    def _world_to_agent_points(self, obs: Dict[str, Any], ctrl_idx: int, points_world: torch.Tensor) -> torch.Tensor:
        if points_world.numel() == 0:
            return torch.zeros(0, 2, device=self.device, dtype=torch.float32)
        agent_from_world = obs.get("agent_from_world")
        if agent_from_world is None:
            world_from_agent = obs["world_from_agent"][ctrl_idx]
            agent_from_world = torch.linalg.inv(world_from_agent)
        else:
            agent_from_world = agent_from_world[ctrl_idx]
        ones = torch.ones(points_world.shape[0], 1, device=points_world.device, dtype=points_world.dtype)
        homo = torch.cat([points_world, ones], dim=-1)
        local = homo @ agent_from_world.T
        return local[:, :2]

    def _rollout_controls(self, start_state: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if controls.shape[0] == 0:
            return torch.zeros(0, 4, device=self.device, dtype=torch.float32)
        states = []
        state = start_state.clone()
        for control in controls:
            state = self._unicycle_step(state, control)
            states.append(state)
        return torch.stack(states, dim=0)

    def _extend_controls_to_targets(
        self,
        start_state: torch.Tensor,
        local_future_anchors: List[Dict[str, Any]],
        existing_prefix_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        remaining = self._horizon - existing_prefix_len
        if remaining <= 0:
            return (
                torch.zeros(0, 2, device=self.device, dtype=torch.float32),
                torch.zeros(0, 4, device=self.device, dtype=torch.float32),
            )

        actions = []
        states = []
        state = start_state.clone()
        target_idx = 0
        for proposal_step in range(existing_prefix_len, self._horizon):
            while target_idx < len(local_future_anchors) and local_future_anchors[target_idx]["rel_step"] < proposal_step:
                target_idx += 1

            if target_idx >= len(local_future_anchors):
                tail_actions, tail_states = self._build_constant_velocity_tail(
                    start_state=state,
                    tail_length=self._horizon - proposal_step,
                )
                if tail_actions.shape[0] > 0:
                    actions.append(tail_actions)
                    states.append(tail_states)
                break

            target = local_future_anchors[target_idx]
            target_xy = target["position_local"]
            steps_to_target = max(int(target["rel_step"] - proposal_step + 1), 1)
            control = self._steer_toward_target(state, target_xy, steps_to_target)
            state = self._unicycle_step(state, control)
            actions.append(control.unsqueeze(0))
            states.append(state.unsqueeze(0))

        if actions:
            return torch.cat(actions, dim=0), torch.cat(states, dim=0)
        return (
            torch.zeros(0, 2, device=self.device, dtype=torch.float32),
            torch.zeros(0, 4, device=self.device, dtype=torch.float32),
        )

    def _build_constant_velocity_tail(self, start_state: torch.Tensor, tail_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if tail_length <= 0:
            return (
                torch.zeros(0, 2, device=self.device, dtype=torch.float32),
                torch.zeros(0, 4, device=self.device, dtype=torch.float32),
            )
        controls = torch.zeros(tail_length, 2, device=self.device, dtype=torch.float32)
        states = self._rollout_controls(start_state, controls)
        return controls, states

    def _steer_toward_target(self, state: torch.Tensor, target_xy: torch.Tensor, steps_to_target: int) -> torch.Tensor:
        current_xy = state[:2]
        current_v = float(state[2].item())
        current_yaw = float(state[3].item())
        diff = target_xy - current_xy
        distance = float(torch.linalg.norm(diff).item())
        desired_yaw = float(torch.atan2(diff[1], diff[0]).item()) if distance > 1e-4 else current_yaw
        yaw_error = self._wrap_angle(desired_yaw - current_yaw)
        desired_speed = distance / max(steps_to_target * self._step_time, self._step_time)
        max_speed = float(getattr(self._dyn, "vbound", [0.0, 40.0])[1])
        desired_speed = max(0.0, min(max_speed, desired_speed))
        desired_acc = (desired_speed - current_v) / max(steps_to_target * self._step_time, self._step_time)
        desired_yaw_rate = yaw_error / max(self._step_time, 1e-3)
        control = torch.tensor([desired_acc, desired_yaw_rate], device=self.device, dtype=torch.float32)
        return self._clip_control(state, control)

    def _clip_control(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        v = float(state[2].item())
        vclip = max(abs(v), 0.1)
        max_steer = float(getattr(self._dyn, "max_steer", 0.5))
        max_yawvel = float(getattr(self._dyn, "max_yawvel", 8.0))
        yaw_bound = min(max_steer * vclip, max_yawvel / vclip)
        yaw_bound = max(yaw_bound, 0.1)
        acce_bound = getattr(self._dyn, "acce_bound", (-6.0, 4.0))
        vbound = getattr(self._dyn, "vbound", (-10.0, 30.0))
        acc_lb = max(acce_bound[0], vbound[0] - v)
        acc_ub = min(acce_bound[1], vbound[1] - v)
        clipped = torch.empty_like(control)
        clipped[0] = torch.clamp(control[0], min=float(acc_lb), max=float(acc_ub))
        clipped[1] = torch.clamp(control[1], min=-float(yaw_bound), max=float(yaw_bound))
        return clipped

    def _unicycle_step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        control = self._clip_control(state, control)
        x, y, v, yaw = state
        acc, yaw_rate = control
        dt = self._step_time
        v_mid = v + 0.5 * acc * dt
        next_x = x + torch.cos(yaw) * v_mid * dt
        next_y = y + torch.sin(yaw) * v_mid * dt
        next_v = v + acc * dt
        vbound = getattr(self._dyn, "vbound", (-10.0, 30.0))
        next_v = torch.clamp(next_v, min=float(vbound[0]), max=float(vbound[1]))
        next_yaw = yaw + yaw_rate * dt
        return torch.stack([next_x, next_y, next_v, next_yaw], dim=0)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _update_cache(self, obs: Dict[str, Any], step_index: int, action_info: Dict[str, Any]) -> None:
        selected_controls = action_info.get("selected_controls") if action_info is not None else None
        if selected_controls is None:
            return
        ctrl_mask_raw = obs.get("batch_ctrl_indices")
        if ctrl_mask_raw is None:
            return
        ctrl_mask = torch.as_tensor(ctrl_mask_raw, device=selected_controls.device).bool().flatten()
        ctrl_indices = torch.nonzero(ctrl_mask, as_tuple=False).reshape(-1)
        for ctrl_idx_tensor in ctrl_indices:
            ctrl_idx = int(ctrl_idx_tensor.item())
            scene_index = int(obs["scene_index"][ctrl_idx].item())
            ctrl_track_id = int(obs["track_id"][ctrl_idx].item())
            self._trajectory_cache[(scene_index, ctrl_track_id)] = {
                "rollout_step_index": int(step_index),
                "selected_controls": selected_controls[ctrl_idx].detach().cpu(),
            }
