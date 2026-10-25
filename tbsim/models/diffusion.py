import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from contextlib import nullcontext
import torch
from torch.nn import Module, ModuleList, Parameter

from tbsim.models.common import *
from tbsim.utils.geometry_utils import batch_nd_transform_points
from tbsim.utils.guidance_utils import CombinedLossCalculator, Guidance
from tbsim.utils.tensor_utils import compare_tensors_ignore_nan
from tbsim.utils.trajdata_utils import enlarge_batch_samples
from tbsim.utils.control_projection import project_guided_controls, rollout_controls


class VarianceSchedule(Module):

    def __init__(self, num_steps, mode="cosine", beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ("linear", "cosine")
        print(f"diffusion mode: {mode}")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == "cosine":
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DiffusionTraj(Module):

    def __init__(self, net, var_sched: VarianceSchedule, predict_epsilon=False):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.predict_epsilon = predict_epsilon

    def sample(
        self,
        context,
        num_points,
        bestof,
        num_samples,
        guide_sample_fn=None,
        data_batch_for_guidance=None,
        forward_mode="train",
        current_states=None,
        point_dim=2,
        flexibility=0.0,
        ret_traj=False,
        sampling_mode="ddpm",
        sample_step=1,
        guide_config=None,
        adv_proposals_dict=None,
    ):
        """
        Args:
        context: Traffic context conditioning
        num_points: the number of points in the trajectory
        num_samples: the number of parallel samples for diffusion models
        starting_noisy_a_t: the starting noisy trajectory
        initial steps: the current diffusion steps of the starting_noisy_a_t for the truncated diffusion
        adv_indices: which noisy trajectory to replace
        """
        batch_size = context.size(0)
        if data_batch_for_guidance is not None and guide_config is not None:
            data_batch_for_guidance["trace_artifact_dir"] = self._guide_param(
                guide_config,
                "trace_artifact_dir",
                None,
            )
            data_batch_for_guidance["trace_dump_artifacts"] = bool(
                self._guide_param(guide_config, "trace_dump_artifacts", False)
            )
        ## ENLARGE context and states (batch,dim)->(batch*sample,dim)
        context = enlarge_batch_samples(context, batch_size, num_samples)
        if current_states is not None:
            current_states = enlarge_batch_samples(current_states, batch_size, num_samples)  # [:batch_size] are the same
            assert torch.all(torch.eq(current_states[: num_samples - 1, :], current_states[1:num_samples, :]))
        assert compare_tensors_ignore_nan(context[: num_samples - 1, :], context[1:num_samples, :])
        if bestof:
            a_T = torch.randn([batch_size * num_samples, num_points, point_dim]).to(context.device)
        else:
            a_T = torch.zeros([batch_size * num_samples, num_points, point_dim]).to(context.device)
        # 1) Prepare partial diffusion config if needed
        partial_diffusion_cfg = self._setup_partial_diffusion(guide_config, adv_proposals_dict, num_samples)
        
        # Main diffusion loop
        a_t = a_T
        stride = sample_step
        denoised_traj_list = []
        for t in range(self.var_sched.num_steps, 0, -stride):
            z = torch.randn_like(a_T) if t > 1 else torch.zeros_like(a_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t - stride]
            beta_val = self.var_sched.betas[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            # replace the adv a_t if passes the diffusion steps
            if partial_diffusion_cfg["enabled"] and t == partial_diffusion_cfg["partial_step"]:
                a_t = self._replace_with_partial_diffusion(
                    a_t, 
                    batch_size, 
                    num_samples, 
                    num_points, 
                    point_dim,
                    partial_diffusion_cfg["adv_proposals"],
                    partial_diffusion_cfg["adv_idx"]
                )
            c0 = torch.sqrt(alpha_bar_next) * beta_val / (1 - alpha_bar)
            c1 = torch.sqrt(alpha) * (1 - alpha_bar_next) / (1 - alpha_bar)
            with torch.enable_grad() if guide_sample_fn is not None else torch.no_grad():
                clean_control = guide_sample_fn is not None and guide_config.params.grad_wrt == "clean_control"
                if guide_sample_fn is not None and not clean_control:
                    a_t.requires_grad_(True)
                beta = self.var_sched.betas[[t] * batch_size * num_samples]  # beta means time here?

                ## forward dyanmics
                if self.net.dyn is not None:
                    traj_t = self.net.dyn.forward_dynamics(current_states, a_t, self.net.step_time)[0]
                    tau_t = torch.cat([a_t, traj_t], dim=-1)
                else:
                    tau_t = a_t
                if self.predict_epsilon:
                    raise NotImplementedError
                else:
                    with torch.no_grad() if clean_control else nullcontext():
                        a0_predict = self.net(
                            tau_t, context, time=beta, current_states=current_states, output_type="control"
                        )
                    a0_predict = project_guided_controls(a0_predict,current_states,data_batch_for_guidance,self.net.step_time,stage='denoised_clean')
                    if guide_sample_fn is not None and guide_config.params.grad_wrt in {"clean_guide", "clean_control"}:
                        if data_batch_for_guidance is not None:
                            data_batch_for_guidance["diffusion_step"] = int(t)
                            data_batch_for_guidance["loss_trace_stage"] = "guidance_grad"
                        a0_predict = self.n_step_guided_p_sample(
                            a_t,
                            a0_predict,
                            guide=guide_sample_fn,
                            beta=beta_val,
                            current_states=current_states,
                            data_batch_for_guidance=data_batch_for_guidance,
                            inner_lr = guide_config.params.inner_lr,
                            inner_beta = guide_config.params.inner_beta,
                            guidance_horizon = guide_config.params.guidance_horizon if guide_config.params.guidance_horizon is not None else 32,
                            n_guide_steps = guide_config.params.n_guide_steps,
                            scale_grad_by_std = guide_config.params.scale_grad_by_std,
                            multiple_guidance_strategy = guide_config.params.multiple_guidance_strategy,
                            grad_wrt = guide_config.params.grad_wrt,
                            channel_step_max = self._guide_param(guide_config,"channel_step_max",None),
                        )
                       
                    if sampling_mode == "ddpm":
                        a_next = c0 * a0_predict + c1 * a_t + sigma * z
                        if guide_sample_fn is not None and guide_config.params.grad_wrt == "noisy_guide":
                            a_next = self.n_step_guided_p_sample(
                                a_t,
                                a_next,
                                guide=guide_sample_fn,
                                beta=beta_val,
                                current_states=current_states,
                                **guide_config.params.to_dict(),
                            )

                    elif sampling_mode == "ddim":
                        e_theta = (a_t - alpha_bar.sqrt() * a0_predict) / (1 - alpha_bar).sqrt()
                        a_next = alpha_bar_next.sqrt() * a0_predict + (1 - alpha_bar_next).sqrt() * e_theta
                    else:
                        raise NotImplementedError
                a_t = a_next.detach()  # Stop gradient and save trajectory
                if ret_traj:
                    denoised_traj_list.append(a_next.detach())
                else:
                    pass
        if forward_mode == "train":
            return a_t

        if guide_sample_fn is None:
            return a_next.view(batch_size, -1, num_points, point_dim)

        ## do filtration after guidance
        if guide_sample_fn is not None:
            a_t = project_guided_controls(a_t,current_states,data_batch_for_guidance,self.net.step_time,stage='final_clean')
            if self.net.dyn is not None:
                traj_final = (rollout_controls(current_states,a_t,self.net.step_time)
                    if (data_batch_for_guidance or {}).get('control_projection') is not None else
                    self.net.dyn.forward_dynamics(current_states, a_t, self.net.step_time)[0])
            else:
                raise NotImplementedError

            if data_batch_for_guidance is not None:
                data_batch_for_guidance["diffusion_step"] = -1
                data_batch_for_guidance["guide_iteration"] = -1
                data_batch_for_guidance["loss_trace_stage"] = "sample_filter"
            a_traj_best_sample = guide_sample_fn.filter(a_t, traj_final, data_batch_for_guidance)  # input shape(B*sample,...)
            self._maybe_dump_trace_artifact(
                action=a_traj_best_sample[:, 0],
                traj=traj_final.view(batch_size, -1, num_points, traj_final.shape[-1])[:, 0],
                data_batch_for_guidance=data_batch_for_guidance,
                stage_name="sample_filter",
                sample_index=0,
            )
            a_traj_samples = a_t.view(batch_size, -1, num_points, point_dim)
            a_traj_samples[:, 0:1] = a_traj_best_sample
            if ret_traj:
                a_denoised = torch.stack(denoised_traj_list, dim=1).view(
                    batch_size, -1, num_points, point_dim
                )  # (B, sample*diffusionstep,T,2)
                # TODO add index to store?

            return {"sampled_trajectories": a_traj_samples, "denoised_trajectories": a_denoised if ret_traj else None}
        return a_t.view(batch_size, -1, num_points, point_dim)  # (B, sample, T,2)

    def n_step_guided_p_sample(
        self,
        a_t,  # The input action tau_k
        action,  # action is the noisy action or the cleaned action
        guide: Guidance,  # The guidance object
        beta,  # Covariance
        current_states=None,  # Current states, if applicable
        data_batch_for_guidance=None,
        scale: float = 0.001,  # Scaling factor
        guidance_horizon: int = 32,  # Threshold for stopping gradient computation
        n_guide_steps: int = 1,  # Number of guidance steps
        scale_grad_by_std: bool = False,  # Whether to scale gradient by standard deviation
        inner_lr: float = 1.0,  # Inner learning rate
        inner_beta: float = 0.5,  # Inner beta value
        multiple_guidance_strategy: str = "weight_guide",  # Strategy for multiple guidance
        grad_wrt: str = "noisy_guide",  # Gradient with respect to
        channel_step_max=None,
        **kwargs,
    ):
        step_bound=inner_beta
        if channel_step_max is not None:
            if len(channel_step_max)!=action.shape[-1] or any(not np.isfinite(v) or v<=0 for v in channel_step_max):
                raise ValueError('channel_step_max needs one positive finite bound per control channel')
            step_bound=action.new_tensor(channel_step_max).clamp(max=inner_beta)
        for i in range(n_guide_steps):
            if grad_wrt=="clean_control":
                action=action.detach().requires_grad_(True)
            action_original = action.clone()  # Keep a copy of original action

            # pass through the dynamics
            if self.net.dyn is not None:
                traj = (rollout_controls(current_states,action,self.net.step_time)
                    if (data_batch_for_guidance or {}).get('control_projection') is not None else
                    self.net.dyn.forward_dynamics(current_states, action, self.net.step_time, mode="parallel")[0])
            else:
                raise NotImplementedError
            if data_batch_for_guidance is not None:
                data_batch_for_guidance["guide_iteration"] = int(i)
            grad = guide.calculate_grad(action, traj, a_t, grad_wrt, data_batch_for_guidance=data_batch_for_guidance)  # detach action from its computation history

            if scale_grad_by_std:
                grad = beta * grad

            # Update action with gradient and clip changes
            action = action_original + torch.clamp(
                inner_lr * grad,
                -step_bound,
                step_bound
            )
            action = project_guided_controls(action,current_states,data_batch_for_guidance,self.net.step_time,stage='guided_clean')

        if self.net.dyn is not None and data_batch_for_guidance is not None:
            final_traj = (rollout_controls(current_states,action,self.net.step_time)
                if data_batch_for_guidance.get('control_projection') is not None else
                self.net.dyn.forward_dynamics(current_states,action,self.net.step_time,mode="parallel")[0])
            self._maybe_dump_trace_artifact(
                action=action,
                traj=final_traj,
                data_batch_for_guidance=data_batch_for_guidance,
                stage_name=data_batch_for_guidance.get("loss_trace_stage", "guidance_grad"),
                sample_index=0,
            )

        return action

    @staticmethod
    def _guide_param(guide_config, key, default=None):
        params = getattr(guide_config, "params", None)
        if params is None:
            return default
        if isinstance(params, dict):
            return params.get(key, default)
        return getattr(params, key, default)

    @staticmethod
    def _trace_scalar(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.detach().reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            return value[0] if len(value) > 0 else None
        return value

    def _maybe_dump_trace_artifact(
        self,
        action,
        traj,
        data_batch_for_guidance,
        stage_name,
        sample_index=0,
    ):
        if data_batch_for_guidance is None:
            return
        if not bool(data_batch_for_guidance.get("trace_dump_artifacts", False)):
            return
        root_dir = data_batch_for_guidance.get("trace_artifact_dir")
        if not root_dir:
            return

        ctrl_flat_idx = self._select_ctrl_flat_index(
            data_batch_for_guidance,
            sample_index,
            max_rows=int(action.shape[0]),
        )
        if ctrl_flat_idx is None:
            return
        if ctrl_flat_idx < 0 or ctrl_flat_idx >= int(action.shape[0]) or ctrl_flat_idx >= int(traj.shape[0]):
            return

        rollout_step_value = self._trace_scalar(data_batch_for_guidance.get("rollout_step_index"))
        diffusion_step_value = self._trace_scalar(data_batch_for_guidance.get("diffusion_step"))
        guide_iteration_value = self._trace_scalar(data_batch_for_guidance.get("guide_iteration"))
        rollout_step = int(rollout_step_value if rollout_step_value is not None else -1)
        diffusion_step = int(diffusion_step_value if diffusion_step_value is not None else -1)
        guide_iteration = int(guide_iteration_value if guide_iteration_value is not None else -1)
        scene_id = self._trace_scalar(data_batch_for_guidance.get("scene_ids"))

        artifact_dir = os.path.join(
            root_dir,
            f"rollout_{rollout_step:03d}",
            f"{stage_name}_diff_{diffusion_step:03d}_guide_{guide_iteration:03d}",
        )
        os.makedirs(artifact_dir, exist_ok=True)

        action_sel = action[ctrl_flat_idx].detach().cpu().numpy()
        traj_sel = traj[ctrl_flat_idx].detach().cpu().numpy()
        world_traj = self._to_world_positions(
            traj[ctrl_flat_idx : ctrl_flat_idx + 1, :, :2],
            data_batch_for_guidance["world_from_agent"][ctrl_flat_idx : ctrl_flat_idx + 1],
        )[0].detach().cpu().numpy()

        proposal_actions, proposal_states_world = self._extract_proposal_artifacts(
            data_batch_for_guidance,
            ctrl_flat_idx,
        )
        anchor_world, anchor_valid = self._extract_anchor_artifacts(
            data_batch_for_guidance,
            ctrl_flat_idx,
        )

        np.save(os.path.join(artifact_dir, "action_accel_yaw_rate.npy"), action_sel)
        np.save(os.path.join(artifact_dir, "trajectory_local_state.npy"), traj_sel)
        np.save(os.path.join(artifact_dir, "trajectory_world_xy.npy"), world_traj)
        if proposal_actions is not None:
            np.save(os.path.join(artifact_dir, "proposal_action_accel_yaw_rate.npy"), proposal_actions)
        if proposal_states_world is not None:
            np.save(os.path.join(artifact_dir, "proposal_world_xy.npy"), proposal_states_world)
        if anchor_world is not None:
            np.save(os.path.join(artifact_dir, "anchor_world_xy.npy"), anchor_world)
            np.save(os.path.join(artifact_dir, "anchor_valid_mask.npy"), anchor_valid)

        metadata = {
            "scene_id": scene_id,
            "rollout_step_index": rollout_step,
            "diffusion_step": diffusion_step,
            "guide_iteration": guide_iteration,
            "trace_stage": stage_name,
            "ctrl_flat_index": int(ctrl_flat_idx),
            "sample_index": int(sample_index),
        }
        with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self._save_action_profile_plot(
            output_path=os.path.join(artifact_dir, "action_profile.png"),
            action=action_sel,
            title=f"{stage_name} rollout={rollout_step} diff={diffusion_step}",
        )
        self._save_trajectory_plot(
            output_path=os.path.join(artifact_dir, "trajectory_world.png"),
            world_traj=world_traj,
            proposal_world=proposal_states_world,
            anchor_world=anchor_world,
            anchor_valid=anchor_valid,
            title=f"{stage_name} rollout={rollout_step} diff={diffusion_step}",
        )

    @staticmethod
    def _select_ctrl_flat_index(data_batch_for_guidance, sample_index, max_rows=None):
        max_rows = int(max_rows) if max_rows is not None else None
        anchor_mask = data_batch_for_guidance.get("adv_anchor_valid_mask")
        if anchor_mask is not None:
            flat = anchor_mask.reshape(anchor_mask.shape[0], -1).any(dim=1).nonzero(as_tuple=False)
            if flat.numel() > 0:
                ctrl_flat_idx = int(flat.reshape(-1)[0].item())
                if max_rows is None or ctrl_flat_idx < max_rows:
                    return ctrl_flat_idx

        adv_proposals_dict = data_batch_for_guidance.get("adv_proposals_dict")
        if isinstance(adv_proposals_dict, dict):
            adv_idx = adv_proposals_dict.get("adv_idx")
            if adv_idx is not None and len(adv_idx) > 0:
                ctrl_idx = int(adv_idx.detach().reshape(-1)[0].item())
                num_samples = int(data_batch_for_guidance.get("num_samples", 1))
                flat_idx = ctrl_idx * num_samples + int(sample_index)
                if max_rows is None or flat_idx < max_rows:
                    return flat_idx
                if ctrl_idx < max_rows:
                    return ctrl_idx
        return None

    @staticmethod
    def _to_world_positions(local_xy, world_from_agent):
        return batch_nd_transform_points(local_xy, world_from_agent)

    def _extract_proposal_artifacts(self, data_batch_for_guidance, ctrl_flat_idx):
        adv_proposals_dict = data_batch_for_guidance.get("adv_proposals_dict")
        if not isinstance(adv_proposals_dict, dict):
            return None, None
        adv_idx = adv_proposals_dict.get("adv_idx")
        proposal_actions = adv_proposals_dict.get("adv_proposals")
        proposal_states = adv_proposals_dict.get("adv_proposals_states")
        if adv_idx is None or proposal_actions is None or proposal_states is None:
            return None, None

        num_samples = int(data_batch_for_guidance.get("num_samples", 1))
        ctrl_batch_idx = int(ctrl_flat_idx // max(num_samples, 1))
        matches = (adv_idx == ctrl_batch_idx).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return None, None
        proposal_slot = int(matches.reshape(-1)[0].item())
        proposal_action = proposal_actions[proposal_slot, 0].detach().cpu().numpy()
        proposal_local = proposal_states[proposal_slot, 0, :, :2]
        proposal_world = self._to_world_positions(
            proposal_local.unsqueeze(0),
            data_batch_for_guidance["world_from_agent"][ctrl_flat_idx : ctrl_flat_idx + 1],
        )[0]
        return proposal_action, proposal_world.detach().cpu().numpy()

    @staticmethod
    def _extract_anchor_artifacts(data_batch_for_guidance, ctrl_flat_idx):
        anchor_world = data_batch_for_guidance.get("adv_anchor_positions_world")
        anchor_valid = data_batch_for_guidance.get("adv_anchor_valid_mask")
        if anchor_world is None or anchor_valid is None:
            return None, None
        anchor_world = anchor_world[ctrl_flat_idx].detach().cpu().numpy()
        anchor_valid = anchor_valid[ctrl_flat_idx].detach().cpu().numpy().astype(bool)
        return anchor_world, anchor_valid

    @staticmethod
    def _save_action_profile_plot(output_path, action, title):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        steps = np.arange(action.shape[0])
        axes[0].plot(steps, action[:, 0], color="#d64545", linewidth=2.0)
        axes[0].set_ylabel("accel")
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(steps, action[:, 1], color="#2f6db0", linewidth=2.0)
        axes[1].set_ylabel("yaw_rate")
        axes[1].set_xlabel("horizon step")
        axes[1].grid(True, alpha=0.25)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _save_trajectory_plot(output_path, world_traj, proposal_world, anchor_world, anchor_valid, title):
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.plot(world_traj[:, 0], world_traj[:, 1], color="#d64545", linewidth=2.4, label="guided traj")
        ax.scatter(world_traj[0, 0], world_traj[0, 1], color="#d64545", s=40)
        if proposal_world is not None:
            ax.plot(
                proposal_world[:, 0],
                proposal_world[:, 1],
                color="#6a5acd",
                linestyle="--",
                linewidth=2.0,
                label="proposal",
            )
        if anchor_world is not None and anchor_valid is not None and np.any(anchor_valid):
            valid_anchor_world = anchor_world[anchor_valid]
            ax.scatter(
                valid_anchor_world[:, 0],
                valid_anchor_world[:, 1],
                color="#f08c2e",
                marker="x",
                s=55,
                label="anchors",
            )
        ax.set_title(title)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

   
    def apply_conditioning(x, conditions, action_dim):
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()
        return x

    # -------------------------------------Partial Diffusion Related code ---------------------------------
    def add_diffusion_noise(self, a_0, t=10, sample_size=20):
        # expand the shape to sample_size
        num_adv, num_proposals, T, _ = a_0.shape
        assert sample_size % num_proposals == 0
        scale_factor = sample_size // num_proposals

        # reshape a_0 to (num_adv, num_proposals, T, 2) -> (num_adv, num_proposals,scale_sample T, 2)
        a_0 = a_0.unsqueeze(2).repeat(1, 1, scale_factor, 1, 1).reshape(num_adv, num_proposals * scale_factor, T, 2)
        # create t of shape (num_adv*num_proposals*scale_sample)
        t = [t] * num_adv

        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1, 1).to(a_0.device)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1).to(a_0.device)

        e_rand = torch.randn_like(a_0)

        ## add perturb noise to action
        a_pert = c0 * a_0 + c1 * e_rand

        return a_pert  # (num_adv, sample, T, 2)
    def handle_partial_diffusion(
        a_t: torch.Tensor,
        batch_size: int,
        sample: int,
        num_points: int,
        point_dim: int,
        current_step: int,
        partial_cfg: dict,
    ) -> torch.Tensor:
        """
        Replaces noisy trajectories with the partially diffused ones
        if `current_step` matches `partial_cfg["partial_step"]`.
        Parameters:
        - a_t (Tensor): Current noisy trajectory of shape (B*sample, T, 2)
        - batch_size (int): Original batch size
        - sample (int): Number of sampled trajectories
        - num_points (int): Horizon length (T)
        - point_dim (int): Dimensionality of each point (2D or more)
        - current_step (int): Current iteration in the denoising loop
        - partial_cfg (dict): Config with keys:
            * "enabled" (bool): If partial diffusion is used
            * "partial_step" (int): Step index at which partial diffusion is applied
            * "adv_proposals" (Tensor): Proposed partial trajectories (B_adv, ...)
            * "adv_idx" (Tensor): Indices of batch elements that need partial replacement

        Returns:
        - Tensor: Possibly updated `a_t`.
        """
        if partial_cfg is None or not partial_cfg["enabled"]:
            return a_t  # No partial diffusion

        if current_step == partial_cfg["partial_step"]:
            # Perform the partial diffusion replacement
            a_t = a_t.view(batch_size, sample, num_points, point_dim)
            a_t[partial_cfg["adv_idx"]] = partial_cfg["adv_proposals"]
            a_t = a_t.view(batch_size * sample, num_points, point_dim)
        return a_t
    
    @staticmethod
    def apply_fixed_actions(action, fixed_actions, indices):
        action[:, indices] = fixed_actions[:, indices]
        return action

    def _setup_partial_diffusion(self, guide_config, adv_proposals_dict, sample):
        """Setup configuration for partial diffusion if enabled."""
        if guide_config.params.partial_t is not None and adv_proposals_dict is not None:
            return {
                "enabled": True,
                "partial_step": guide_config.params.partial_t,
                "adv_proposals": self.add_diffusion_noise(
                    adv_proposals_dict["adv_proposals"], 
                    t=guide_config.params.partial_t, 
                    sample_size=sample
                ),
                "adv_idx": adv_proposals_dict["adv_idx"],
            }
        return {"enabled": False}

    def _replace_with_partial_diffusion(self, a_t, batch_size, sample, num_points, point_dim, adv_proposals, adv_idx):
        """Replace specified trajectories with partially diffused ones."""
        a_t = a_t.view(batch_size, sample, num_points, point_dim)
        a_t[adv_idx] = adv_proposals
        return a_t.view(batch_size * sample, -1, point_dim)


import matplotlib.pyplot as plt

def plot_variance_schedule(var_schedule):
    # Assuming var_schedule is an instance of VarianceSchedule with the 'cosine' mode
    num_steps = var_schedule.num_steps
    t = torch.arange(0, num_steps + 1).float()  # Assuming t starts from 1 to num_steps

    alpha_bar = var_schedule.alpha_bars[t.long()]
    alpha = var_schedule.alphas[t.long()]
    beta_val = var_schedule.betas[t.long()]
    sigma = var_schedule.sigmas_flex[t.long()]

    alpha_bar_next = var_schedule.alpha_bars[(t - 1).long()]
    c0 = torch.sqrt(alpha_bar_next) * beta_val / (1 - alpha_bar)
    c1 = torch.sqrt(alpha) * (1 - alpha_bar_next) / (1 - alpha_bar)

    c0 = torch.sqrt(alpha_bar)  # (B, 1, 1)
    c1 = torch.sqrt(1 - alpha_bar)  # (B, 1, 1)
    plt.figure(figsize=(12, 8))

    # Plotting c0 and c1
    plt.plot(t, c0.cpu(), label="c0", color="red")
    plt.plot(t, c1.cpu(), label="c1", color="blue")

    # Plotting sigmas, alphas, and alpha_bars for reference
    plt.plot(t, sigma.cpu(), label="Sigmas", color="black")
    plt.plot(t, alpha_bar.cpu(), label="Alpha bars", color="green", linestyle="--")
    plt.plot(t, alpha.cpu(), label="Alphas", color="purple", linestyle="--")

    plt.plot(t, beta_val.cpu(), label="Beta", color="orange", linestyle="--")
    plt.title("Variance Schedule and Parameters")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("Schedules")


if __name__ == "__main__":
    var_schedule = VarianceSchedule(100, beta_T=5e-2, mode="cosine")
    plot_variance_schedule(var_schedule)
