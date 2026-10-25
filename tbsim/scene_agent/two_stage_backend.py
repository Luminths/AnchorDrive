"""Adapt saved ReAct scenes to the original offline Stage2DiffusionPolicy."""
from .roles import is_helper_role
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .anchor_backend import NuscAnchorWorld
from .geometry import wrap
from tbsim.utils.control_projection import DEFAULT_LIMITS

ROOT=Path(__file__).resolve().parents[2]


def align_projection_current_speeds(batch, names, states):
    """Use the same initial speed for projected dynamics and native execution.

    Trajdata's v_lon can be negative or differ from the velocity magnitude.
    This adapter executes forward controls from the measured world speed, so
    retaining v_lon as the projection initial state would invalidate its bounds.
    Keep the encoder history and original world states intact.
    """
    native=batch['curr_speed']
    execution=native.new_tensor([states[aid]['speed'] for aid in names]).reshape(native.shape)
    if not torch.isfinite(execution).all() or (execution<0).any():
        raise ValueError('Projected execution needs finite nonnegative current speeds')
    alignment=dict(
        native_longitudinal_speed_mps_by_actor=dict(zip(names,native.detach().cpu().reshape(-1).tolist())),
        execution_speed_mps_by_actor={aid:float(states[aid]['speed']) for aid in names})
    batch['curr_speed']=execution
    return alignment


def collision_decay_start_steps(last_anchor_steps, start_mode='final_anchor', stage1_end_step=None):
    """Keep legacy per-actor cutoffs or use the shared executed Stage 1 end frame."""
    if start_mode=='final_anchor':
        return dict(last_anchor_steps)
    if start_mode!='stage1_end':
        raise ValueError('Unknown post_anchor_collision_start: '+str(start_mode))
    if stage1_end_step is None or stage1_end_step<0 or int(stage1_end_step)!=stage1_end_step:
        raise ValueError('stage1_end requires the executed Stage 1 end step')
    return {aid:int(stage1_end_step) for aid in last_anchor_steps}


def collision_time_schedule(names, last_anchor_steps, step_index, horizon, factor, device, dtype,
                            decay_duration_s=None, dt=.1):
    """Absolute-time schedule; linear mode drops at the supplied cutoff then fades."""
    if not 0. < factor <= 1.:
        raise ValueError('post_anchor_collision_factor must be in (0, 1]')
    active=torch.ones(len(names),horizon,device=device,dtype=torch.bool)
    future_steps=step_index+torch.arange(1,horizon+1,device=device)
    scale=torch.ones(len(names),horizon,device=device,dtype=dtype)
    if decay_duration_s is not None and (not math.isfinite(decay_duration_s) or decay_duration_s<=0 or dt<=0):
        raise ValueError('Collision decay duration and dt must be positive')
    for i,aid in enumerate(names):
        if aid in last_anchor_steps:
            if decay_duration_s is None:
                active[i]=future_steps<=last_anchor_steps[aid]
                scale[i]=torch.where(active[i],1.,factor)
            else:
                active[i]=future_steps<last_anchor_steps[aid]
                remaining=(1.-(future_steps-last_anchor_steps[aid]).to(dtype)*dt/decay_duration_s).clamp(0.,1.)
                scale[i]=torch.where(active[i],torch.ones_like(remaining),factor*remaining)
    return active,scale



def resolve_sampling_contract(training_steps, options):
    params=options['params']
    stride=params.get('sample_step',1)
    if not isinstance(stride,int) or stride<1 or training_steps%stride:
        raise ValueError('Sampling stride must divide the trained diffusion schedule')
    if stride!=1 and params['sampling_mode']!='ddim':
        raise ValueError('Reduced sampling must use DDIM; legacy DDPM coefficients are single-step')
    steps=training_steps//stride
    if options.get('denoising_steps',steps)!=steps:
        raise ValueError('Requested denoising steps differ from actual trained-schedule sampling')
    return dict(training_diffusion_steps=training_steps,denoising_steps=steps,
        sampling_mode=params['sampling_mode'],sample_step=stride,
        num_samples=params['num_samples'],n_guide_steps=params['n_guide_steps'])


def load_two_stage_policy(device, options, case_dir, output):
    from tbsim.algos.algos import DiffusionTrafficModel
    from tbsim.configs.config import Dict
    from tbsim.configs.eval_config import EvaluationConfig
    from tbsim.configs.guidance_config import GuidanceConfig
    from tbsim.policies.stage2_diffusion import Stage2DiffusionPolicy
    from tbsim.utils.batch_utils import batch_utils, set_global_batch_type
    from tbsim.utils.config_utils import get_experiment_config_from_file

    set_global_batch_type('trajdata')
    cfg=get_experiment_config_from_file(str(ROOT/options['checkpoint_config']))
    guide=GuidanceConfig()
    guide.set_guidance_fn(options['guidance_names'])
    guide.update_params(options['params'])
    guide.update_combine_loss(options['combine_loss'])
    guide.update_configs(options['guidance_overrides'])
    cfg.algo.do_guidance=True
    cfg.algo.guide_config=Dict(guide.to_dict())
    model=DiffusionTrafficModel.load_from_checkpoint(str(ROOT/options['checkpoint']),
        algo_config=cfg.algo,modality_shapes=batch_utils().get_modality_shapes(cfg),
        map_location='cpu').to(device).eval()
    resolve_sampling_contract(model.nets['policy'].diffusion.var_sched.num_steps,options)
    model.requires_grad_(False)
    model.nets['policy'].Loss_Calculater.to_device(device)
    evaluation=EvaluationConfig()
    evaluation.stage2.enabled=True
    evaluation.stage2.mode='case_anchor'
    evaluation.stage2.case_dir=str(case_dir)
    evaluation.results_dir=str(output)
    policy=Stage2DiffusionPolicy(model,evaluation,str(case_dir),require_anchor=True)
    return policy,guide.to_dict()


class NuscTwoStageWorld(NuscAnchorWorld):
    """Original Stage 2 sampling/loss schedule; native state/layout adapter only."""
    trajectory_method='original_offline_stage2_case_anchor'

    def begin_two_stage(self, goal, case_dir, output, track_ids):
        participants={p['actor_id'] for p in goal['participants']}
        super().begin_anchor_execution(participants)
        self.track_ids=track_ids
        self.helpers={p['actor_id'] for p in goal['participants'] if is_helper_role(p['role'])}
        self.attackers=participants-self.helpers
        self.diffusion_device=torch.device(self.config['diffusion_device'])
        if self.diffusion_device.type == "cuda":
            torch.cuda.set_device(self.diffusion_device)
        torch.manual_seed(20261001+self.scene_index)
        self.diffusion_options=copy.deepcopy(self.config['two_stage'])
        self.projection_enabled=bool(self.diffusion_options.get('control_projection',False))
        self.diffusion_previous={aid:[0.,0.] for aid in self.states if aid!='ego'}
        if self.projection_enabled:self._safesim_env.preserve_action_precision=True
        self.stage2_policy,self.resolved_guidance=load_two_stage_policy(
            self.diffusion_device,self.diffusion_options,case_dir,output)
        self.reference_ego_by_step={}
        self.reference_paths_by_actor={}
        if self.diffusion_options.get('anchor_adaptation') or self.diffusion_options.get('anchor_phase_adaptation'):
            for entry in self.stage2_policy._anchor_registry.values():
                case=json.loads(Path(entry['case_path']).read_text())
                for point in case['ego_rollout_traj']:
                    xy=point['position_world'];step=point['step']
                    if step in self.reference_ego_by_step and self.reference_ego_by_step[step]!=xy:
                        raise ValueError('Stage 1 participants disagree on measured EGO trajectory')
                    self.reference_ego_by_step[step]=xy
                self.reference_paths_by_actor[case['actor_id']]={
                    p['step']:p['position_world'] for p in case['adv_rollout_traj']}
        self.sampling_contract=resolve_sampling_contract(
            self.stage2_policy.policy.nets['policy'].diffusion.var_sched.num_steps,self.diffusion_options)
        self.last_anchor_steps={}
        if self.diffusion_options.get('post_anchor_collision_factor') is not None:
            for aid in self.attackers:
                entry=self.stage2_policy._anchor_registry[(self.scene_index,track_ids[aid])]
                self.last_anchor_steps[aid]=max(int(a['step']) for a in entry['adv_anchor_plan'])
        self.collision_decay_start_steps=collision_decay_start_steps(self.last_anchor_steps,
            self.diffusion_options.get('post_anchor_collision_start','final_anchor'),
            self.config.get('stage1_end_step'))
        self.diffusion_plans={}
        self.diffusion_records=[]
        self.road_body_field=None
        if self.diffusion_options.get('road_body_guidance'):
            from tbsim.utils.road_body_guidance import build_road_field
            self.road_body_field=build_road_field(self._road_union(),self.states,self.diffusion_device,
                                                 **self.diffusion_options['road_body_guidance'])
        self.initial_collision_pairs=[]
        if self.diffusion_options.get('exclude_initial_collision_pairs'):
            from shapely.geometry import Polygon
            from .geometry import corners
            originals=[a for a,s in self.states.items() if a!='ego' and s['source']=='existing']
            bodies={a:Polygon(corners(self.states[a])) for a in originals}
            for i,a in enumerate(originals):
                for b in originals[i+1:]:
                    if bodies[a].intersects(bodies[b]):self.initial_collision_pairs.append((a,b))
            (Path(output)/'initial_overlap_manifest.json').write_text(json.dumps(dict(
                pairs=self.initial_collision_pairs,rule='Original-original pre-existing pairs excluded from incident collision loss and metrics; vehicles retained.'),indent=2))
        # The historical wrapper supports per-track anchor registries. Extend only
        # collision pairing from one ADV to all attackers in these new multi-car goals.
        self.stage2_policy.policy.nets['policy'].Loss_Calculater.loss_calculator_dict[
            'causecollision'].all_ctrl_per_ego=True

    def prepare_stage2(self):
        from tbsim.utils import tensor_utils as TensorUtils
        start=time.perf_counter()
        env=self._safesim_env
        env._cached_observation=None
        batch=copy.deepcopy(env.get_observation(include_ego_obs=True)['agents'])
        names=list(env.current_agent_names)
        batch['track_id']=np.array([self.track_ids[a] for a in names])
        batch['batch_ctrl_indices']=np.array([a in self.anchor_participants for a in names])
        batch['batch_ego_indices']=np.array([a=='ego' for a in names])
        self.ego_current_plan()
        ego=self.states['ego'];c,s=math.cos(ego['yaw']),math.sin(ego['yaw'])
        batch['ego_plan']=np.array([[[c*(p['x']-ego['x'])+s*(p['y']-ego['y']),
            -s*(p['x']-ego['x'])+c*(p['y']-ego['y'])]
            for p in self._ego_plan['states'][1:]]],dtype=np.float32)
        batch['ego_plan_yaw_world']=np.array([[p['yaw'] for p in self._ego_plan['states'][1:]]],dtype=np.float32)
        batch=TensorUtils.to_torch(batch,device=self.diffusion_device,ignore_if_unspecified=True)
        speed_alignment=align_projection_current_speeds(batch,names,self.states) if self.projection_enabled else {}
        step_index=round(self.time/self.dt)
        batch['rollout_step_index']=step_index
        if self.road_body_field is not None:batch['road_body_field']=self.road_body_field
        if self.initial_collision_pairs:
            mask=torch.zeros(len(names),len(names),device=self.diffusion_device,dtype=torch.bool)
            for a,b in self.initial_collision_pairs:
                i,j=names.index(a),names.index(b);mask[i,j]=mask[j,i]=True
            batch['initial_collision_pair_mask']=mask
        augmentation=self.stage2_policy._build_stage2_tensors(batch,step_index)
        anchor_adaptation=None
        if self.diffusion_options.get('anchor_phase_adaptation'):
            from .phase_anchors import retime_anchor_targets
            effective,anchor_adaptation=retime_anchor_targets(
                augmentation['adv_anchor_positions_world'],augmentation['adv_anchor_valid_mask'],names,
                self.reference_ego_by_step,self.reference_paths_by_actor,self._ego_plan['states'],
                self.states,step_index,self.attackers,dt=self.dt,
                **self.diffusion_options['anchor_phase_adaptation'])
            augmentation['adv_anchor_positions_world']=effective
        if self.diffusion_options.get('anchor_adaptation'):
            if anchor_adaptation is not None:raise ValueError('Choose one anchor adaptation method')
            from .adaptive_anchors import adapt_anchor_targets
            effective,anchor_adaptation=adapt_anchor_targets(
                augmentation['adv_anchor_positions_world'],augmentation['adv_anchor_valid_mask'],names,
                self.reference_ego_by_step,self._ego_plan['states'],step_index,
                **self.diffusion_options['anchor_adaptation'])
            augmentation['adv_anchor_positions_world']=effective
        weights=augmentation.get('dynamic_batch_weights')
        # Helpers retain their extracted path and BG avoidance; they never receive
        # collision attraction. The original single-ADV interface had no helper role.
        if self.helpers:
            anchor_idx=self.diffusion_options['guidance_names'].index('advanchor')
            for i,aid in enumerate(names):
                if aid in self.helpers:
                    weights[i]=torch.tensor(self.diffusion_options['combine_loss']['weights'],
                        device=self.diffusion_device)
                    if augmentation['adv_anchor_valid_mask'][i].any():weights[i,anchor_idx]=1.
        from .guidance_weights import attacker_weight_override
        weights, weight_override = attacker_weight_override(weights, names, self.attackers,
            self.diffusion_options['guidance_names'], self.diffusion_options.get('attacker_guidance_weights'))
        if weight_override is not None:
            augmentation['dynamic_batch_weights'] = weights
        batch.update(augmentation)
        batch['joint_candidate_selection'] = self.diffusion_options.get('joint_candidate_selection')
        time_schedule={}
        if weight_override is not None:
            time_schedule['attacker_weight_override'] = weight_override
        factor=self.diffusion_options.get('post_anchor_collision_factor')
        if factor is not None:
            active,scale=collision_time_schedule(names,self.collision_decay_start_steps,step_index,
                self.stage2_policy._horizon,factor,self.diffusion_device,batch['centroid'].dtype,
                self.diffusion_options.get('post_anchor_collision_decay_s'),self.dt)
            batch['causecollision_active_time_mask']=active
            batch['causecollision_time_scale']=scale
            cause_idx=self.diffusion_options['guidance_names'].index('causecollision')
            time_schedule.update(
                final_anchor_time_s_by_actor={a:step*self.dt for a,step in self.last_anchor_steps.items()},
                collision_decay_start_time_s_by_actor={a:step*self.dt for a,step in self.collision_decay_start_steps.items()},
                collision_decay_start_source=self.diffusion_options.get('post_anchor_collision_start','final_anchor'),
                stage1_end_time_s=self.config.get('stage1_end_step',0)*self.dt,
                post_anchor_collision_decay_s=self.diffusion_options.get('post_anchor_collision_decay_s'),
                prediction_times_s=[(step_index+j+1)*self.dt for j in range(self.stage2_policy._horizon)],
                collision_time_scale_by_actor={a:scale[i].cpu().tolist() for i,a in enumerate(names) if a in self.attackers},
                effective_collision_weights_by_actor={a:(scale[i]*weights[i,cause_idx]).cpu().tolist()
                    for i,a in enumerate(names) if a in self.attackers})
        loss=self.stage2_policy.policy.nets['policy'].Loss_Calculater
        loss.update_config(dict(batch_ctrl_indices=batch['batch_ctrl_indices'].cpu().numpy(),
            batch_ego_indices=batch['batch_ego_indices'].cpu().numpy()))
        loss.loss_calculator_dict['causecollision'].update_config(dict(
            batch_ctrl_indices=np.array([a in self.attackers for a in names]),
            batch_ego_indices=batch['batch_ego_indices'].cpu().numpy()))
        if self.diffusion_options.get('collision_exclusion_attackers_only'):
            loss.loss_calculator_dict['collision'].update_config(dict(
                batch_ctrl_indices=np.array([a in self.attackers for a in names]),
                batch_ego_indices=batch['batch_ego_indices'].cpu().numpy()))
        projection={}
        if self.projection_enabled:
            limits=dict(DEFAULT_LIMITS,**self.diffusion_options.get('control_projection_limits',{}))
            previous={a:self.diffusion_previous.get(a,[0.,0.]) for a in names}
            stats=dict(calls_by_stage={},calls_by_backend={})
            batch['control_projection']=dict(previous_controls=torch.tensor([previous[a] for a in names],
                device=self.diffusion_device,dtype=torch.float32),limits=limits,
                backend=self.diffusion_options.get('control_projection_backend','torch'),stats=stats)
            batch['projection_preserve_stationary_mask']=True
            projection=dict(enabled=True,limits=limits,previous_controls_by_actor=previous,
                requested_backend=batch['control_projection']['backend'],stats=stats,**speed_alignment)
        else:
            assert 'control_projection' not in batch
        assert 'fixed_guidance_weights' not in batch
        if self.diffusion_device.type == "cuda":
            torch.cuda.synchronize(self.diffusion_device)
        sample_start=time.perf_counter()
        # Call the wrapped model after applying the ORIGINAL wrapper augmentation,
        # allowing the helper compatibility adjustment without recomputing weights.
        denoiser_calls=[]
        denoiser=self.stage2_policy.policy.nets['policy'].diffusion.net
        handle=denoiser.register_forward_hook(lambda *_:denoiser_calls.append(1))
        try:
            with torch.no_grad():
                action,info=self.stage2_policy.policy.get_action(batch,step_index=step_index)
        finally:
            handle.remove()
        if len(denoiser_calls)!=self.sampling_contract['denoising_steps']:
            raise ValueError('Actual denoiser call count differs from configured sampling steps')
        if self.diffusion_device.type == "cuda":
            torch.cuda.synchronize(self.diffusion_device)
        self.stage2_policy._update_cache(batch,step_index,info)
        sampling_s=time.perf_counter()-sample_start
        local_xy=action.positions.detach().cpu().numpy()
        local_yaw=action.yaws.detach().cpu().numpy()
        controls=info['selected_controls'].detach().cpu().numpy()
        plans={}
        for i,aid in enumerate(names):
            if aid=='ego':continue
            actor=self.states[aid];c,s=math.cos(actor['yaw']),math.sin(actor['yaw'])
            if self.projection_enabled:
                speed=actor['speed']+np.cumsum(controls[i,:,0].astype(np.float64)*self.dt)
                heading=actor['yaw']+np.cumsum(controls[i,:,1].astype(np.float64)*self.dt)
                xy=np.array([actor['x'],actor['y']])+np.cumsum(
                    np.column_stack((speed*np.cos(heading),speed*np.sin(heading)))*self.dt,axis=0)
            else:
                xy=np.column_stack((c*local_xy[i,:,0]-s*local_xy[i,:,1],
                                    s*local_xy[i,:,0]+c*local_xy[i,:,1]))+[actor['x'],actor['y']]
                heading=actor['yaw']+local_yaw[i,:,0]
                speed=np.linalg.norm(np.diff(np.vstack(([actor['x'],actor['y']],xy)),axis=0),axis=1)/self.dt
            if not np.isfinite(xy).all() or not np.isfinite(heading).all():
                raise ValueError('Non-finite original Stage 2 trajectory for '+aid)
            plans[aid]=dict(created_at_s=self.time,states=[dict(actor,x=float(p[0]),y=float(p[1]),
                yaw=wrap(float(h)),speed=float(v)) for p,h,v in zip(xy,heading,speed)],
                controls=controls[i].tolist())
        self.diffusion_plans=plans
        record=dict(time_s=self.time,execution_interval_s=self.diffusion_options['diffusion_replan_steps']*self.dt,
            generation_wall_s=time.perf_counter()-start,sampling_wall_s=sampling_s,
            measured_denoiser_calls=len(denoiser_calls),
            device=str(self.diffusion_device),options=self.diffusion_options,sampling_contract=self.sampling_contract,plans=copy.deepcopy(plans),
            generated_vehicle_count=len(plans),background_vehicle_count=len(plans)-len(self.anchor_participants),
            anchor_counts_by_actor={a:int(augmentation['adv_anchor_valid_mask'][i].sum().item()) for i,a in enumerate(names)},
            guidance_weights_by_actor={a:weights[i].detach().cpu().tolist() for i,a in enumerate(names)})
        record.update(time_schedule)
        if anchor_adaptation:record['anchor_adaptation']=anchor_adaptation
        if projection:record['control_projection']=copy.deepcopy(projection)
        if info.get('joint_candidate_selection') is not None:
            record['joint_candidate_selection']=copy.deepcopy(info['joint_candidate_selection'])
        self.diffusion_records.append(record)
        return record

    def _non_ego_step(self,aid,actor,before,next_frame):
        plan=self.diffusion_plans[aid]
        index=round((self.time-plan['created_at_s'])/self.dt)
        if not 0<=index<len(plan['states']):raise ValueError('Stage 2 future expired')
        return copy.deepcopy(plan['states'][index])

    def step(self):
        result=super().step()
        if self.projection_enabled:
            for aid,plan in self.diffusion_plans.items():
                index=round((self.time-plan['created_at_s'])/self.dt)-1
                self.diffusion_previous[aid]=copy.deepcopy(plan['controls'][index])
        return result
