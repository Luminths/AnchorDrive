"""Bridge the retained revision-era planner/assessor to saved ReAct goals.

The old policy owns prompts, parsing, local checks, limited replanning, cached
remaining waypoints and final-plan fallback. This module only translates native
observations/actions and uses the current safe credential/logging client.
"""
import base64
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from tbsim.configs.eval_config import EvaluationConfig
from tbsim.policies.llm_adversary import AnchorDriveLLMAdversaryPolicy
from tbsim.scene_agent.anchor_backend import NuscAnchorWorld
from tbsim.scene_agent.geometry import wrap


class LegacyChatBridge:
    def __init__(self, client, root):
        self.client = client
        self.root = Path(root)
        self.fatal_error = None
        self.exchanges = []
        self.planner_body_gap_m = None
        self.planner_request_meta = None

    def complete(self, system_prompt, user_prompt, user_image_data_url=None):
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        phase = 'assessor' if any(label in system_prompt.casefold() for label in
            ('closed-loop planning evaluation expert', 'scenario execution auditor',
             'scenario execution reviewer', 'post-trial assessor')) else 'planner'
        if phase=='planner' and not getattr(self.client,'config',{}).get('planner_send_bev',True):
            user_image_data_url=None
        content = [{'type': 'text', 'text': user_prompt}]
        record = dict(call=len(self.exchanges)+1, phase=phase,
                      system_prompt=system_prompt, user_prompt=user_prompt)
        if phase=='planner' and self.planner_request_meta is not None:
            record['planner_metadata']=dict(self.planner_request_meta)
        if user_image_data_url:
            extension = 'jpg' if 'image/jpeg' in user_image_data_url.split(',')[0] else 'png'
            name = 'legacy_request_%03d.%s' % (record['call'], extension)
            (self.root/name).write_bytes(base64.b64decode(user_image_data_url.split(',', 1)[1]))
            record['image_name'] = name
            content.append(dict(type='image_url', image_url=dict(url=user_image_data_url)))
        try:
            previous_config=getattr(self.client,'config',None)
            if previous_config is not None:
                effective=dict(previous_config)
                if phase=='planner' and effective.get('planner_model'):
                    effective['model']=effective['planner_model']
                if phase=='planner' and effective.get('planner_near_model'):
                    try:gap=json.loads(user_prompt).get('current_body_geometry',{}).get('body_gap_m')
                    except (ValueError,TypeError,AttributeError):gap=self.planner_body_gap_m
                    threshold=float(effective.get('planner_near_distance_m',4.))
                    if type(gap) in (int,float) and 0<=gap<=threshold:
                        effective['model']=effective['planner_near_model']
                    record['planner_model_routing']=dict(measured_body_gap_m=gap,
                        near_threshold_m=threshold,selected_model=effective.get('model'))
                if 'planner_thinking' in effective:
                    effective['enable_thinking']=bool(effective['planner_thinking']) if phase=='planner' else bool(effective.get('enable_thinking',False))
                if phase=='planner' and effective.get('planner_response_max_tokens'):
                    effective['response_max_tokens']=effective['planner_response_max_tokens']
                self.client.config=effective
                record['model']=effective.get('model')
                record['thinking_enabled']=bool(self.client.config.get('enable_thinking',False))
                record['response_max_tokens']=effective.get('response_max_tokens')
                if effective.get('enable_thinking') and effective.get('reasoning_effort'):
                    record['reasoning_effort']=effective['reasoning_effort']
            try:
                answer = self.client.chat([dict(role='system', content=system_prompt),
                                           dict(role='user', content=content)])
            finally:
                if previous_config is not None:self.client.config=previous_config
            record['response_text'] = answer.get('content') or ''
            return record['response_text']
        except Exception as exc:
            record['error'] = str(exc)
            # Old policy catches API errors and may accept an unaudited plan.
            # Infrastructure failure must remain an error in this experiment.
            if isinstance(exc, RuntimeError):
                self.fatal_error = str(exc)
            raise
        finally:
            self.exchanges.append(record)
            with (self.root/'legacy_api_exchanges.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')


def legacy_eval_config(config, saved, root):
    cfg = EvaluationConfig()
    cfg.results_dir = str(root)
    execution_steps = round(config.get('decision_interval_s', 2.) / config.get('dt', .1))
    cfg.n_step_action = execution_steps
    llm = cfg.llm
    llm.results_dir = str(root)
    llm.model = config['model']
    llm.prompt_version = 'v1'
    llm.scenario_goal = saved['goal']['language_goal']
    llm.trajectory_horizon = 30
    llm.waypoint_horizon_steps = 30
    llm.waypoint_interval_steps = config.get('legacy_waypoint_interval_steps', 5)
    llm.call_interval_steps = execution_steps
    llm.enable_plan_assessor = True
    llm.plan_assessor_max_replans = config.get('legacy_assessor_max_replans', 3)
    llm.max_retries = config.get('legacy_parse_max_retries', 2)
    if 'validation_max_abs_yaw_rate_rps' in config:
        yaw_limit = float(config['validation_max_abs_yaw_rate_rps'])
        if not math.isfinite(yaw_limit) or yaw_limit <= 0:
            raise ValueError('validation_max_abs_yaw_rate_rps must be finite and positive')
        llm.validation_max_abs_yaw_rate_rps = yaw_limit
    llm.send_nearby_future_trajectory = False
    llm.map_info_mode = 'local_radius'
    llm.map_lane_radius_m = 20.
    llm.send_bev_to_llm = config.get('legacy_send_bev', True)
    llm.save_bev_debug = True
    # Keep archived validation defaults, including soft last-plan fallback.
    return cfg


def sanitize_legacy_observation(batch, states, names, participants, ego_states, dt):
    """Replace all recorded future targets by current-state extrapolations."""
    from tbsim.utils import tensor_utils as TensorUtils
    batch = copy.deepcopy(batch)
    # parse_trajdata_batch prepends 11 dynamic-history channels to "image";
    # the archived BEV encoder takes its first three channels as RGB. Supply
    # the static map in the same raster frame, rather than history occupancy.
    # This corrects input adaptation without changing planner/assessor logic.
    if batch.get('maps') is not None:
        batch['image'] = copy.deepcopy(batch['maps'])
    horizon = 32
    targets = np.zeros((len(names), horizon, 2), dtype=np.float32)
    for i, aid in enumerate(names):
        actor = states[aid]
        # Constant heading/velocity in the actor frame; not recorded future.
        targets[i, :, 0] = actor['speed']*dt*np.arange(1, horizon+1)
        batch['curr_speed'][i] = actor['speed']
    for key in list(batch):
        if ('future' in key or '_fut' in key or key.startswith('robot_fut')):
            del batch[key]
    for key in list(batch.get('extras', {})):
        if 'fut' in key:
            del batch['extras'][key]
    batch['target_positions'] = targets
    batch['target_yaws'] = np.zeros((len(names), horizon, 1), dtype=np.float32)
    batch['target_speeds'] = np.repeat(batch['curr_speed'][:, None], horizon, axis=1)
    batch['target_availabilities'] = np.ones((len(names), horizon), dtype=bool)
    batch['batch_ctrl_indices'] = np.array([aid in participants for aid in names])
    batch['batch_ego_indices'] = np.array([aid == 'ego' for aid in names])
    ego_index = names.index('ego')
    batch['ego_idx'] = [ego_index]
    origin = states['ego']
    c, s = math.cos(origin['yaw']), math.sin(origin['yaw'])
    batch['ego_plan'] = np.asarray([[
        [c*(p['x']-origin['x'])+s*(p['y']-origin['y']),
         -s*(p['x']-origin['x'])+c*(p['y']-origin['y'])]
        for p in ego_states[1:]]], dtype=np.float32)
    return TensorUtils.to_torch(batch, device=torch.device('cpu'), ignore_if_unspecified=True)


class LegacySavedGoalPolicy(AnchorDriveLLMAdversaryPolicy):
    def __init__(self, config, saved, root, bridge, background_policy):
        super().__init__(torch.device('cpu'), legacy_eval_config(config, saved, root), background_policy)
        self._client = bridge
        self.saved_goal = saved
        self.actor_names = []

    def _build_prompt_context(self, obs, ctrl_idx, ego_idx, step_index,
                              planning_state=None, failure_feedback=None):
        context = super()._build_prompt_context(obs, ctrl_idx, ego_idx, step_index,
                                              planning_state, failure_feedback)
        # Present the same qualitative goal with unambiguous track-ID aliases.
        mapping = {aid: int(obs['track_id'][i]) for i, aid in enumerate(self.actor_names)}
        roles = [dict(actor_id=p['actor_id'], track_id=mapping[p['actor_id']], role=p['role'])
                 for p in self.saved_goal['goal']['participants']]
        context['scenario_goal'] += "\nThe actor_id of the controlled ADV in this round="+self.actor_names[ctrl_idx]+\
            "; Fixed participating vehicle correspondence: "+json.dumps(roles, ensure_ascii=False)
        return context


class NuscLegacyWaypointWorld(NuscAnchorWorld):
    trajectory_method = 'revision_legacy_waypoint_planner_assessor'

    def prepare_legacy(self, policy):
        env = self._safesim_env
        if self.config.get('physical_waypoint_tracking'):
            env.preserve_action_precision=True
        env._cached_observation = None
        batch = env.get_observation(include_ego_obs=True)['agents']
        names = list(env.current_agent_names)
        self.ego_current_plan()
        obs = sanitize_legacy_observation(batch, self.states, names,
            self.anchor_participants, self._ego_plan['states'], self.dt)
        policy.actor_names = names
        log_start = len(policy.log_path.read_text().splitlines()) if policy.log_path.exists() else 0
        with torch.no_grad():
            action, _ = policy.get_action(obs, step_index=round(self.time/self.dt))
        if policy._client.fatal_error:
            raise RuntimeError(policy._client.fatal_error)
        rows = [json.loads(line) for line in policy.log_path.read_text().splitlines()[log_start:]]
        count = min(policy.executed_steps_per_call, action.positions.shape[1],
                    round((self.config['max_simulation_s']-self.time)/self.dt))
        plans = {}
        decision_by_actor={policy.actor_names[d['ctrl_agent_index']]:d for d in rows
                           if 'ctrl_agent_index' in d}
        for i, aid in enumerate(names):
            if aid == 'ego':
                continue
            if self.config.get('background_mode','recorded')=='recorded' and aid not in self.anchor_participants:
                continue
            points = policy._to_world(action.positions[i, :count], obs['world_from_agent'][i]).cpu().numpy()
            headings = action.yaws[i, :count, 0].cpu().numpy()+self.states[aid]['yaw']
            if self.config.get('physical_waypoint_tracking') and aid in self.anchor_participants:
                meta=decision_by_actor.get(aid,{}).get('parse_meta') or {}
                if not meta.get('physical_tracking'):
                    # The retained policy may return a cached/CV fallback after
                    # format retries are exhausted. Bound that action too, and
                    # record it explicitly rather than silently executing an
                    # unprojected fallback or claiming the LLM plan was valid.
                    from .feasible_waypoints import previous_executed_control,track_waypoints
                    references=[dict(step=k+1,position_world=p.tolist()) for k,p in enumerate(points)]
                    previous=previous_executed_control(policy.episode.trace,aid,self.time,self.dt)
                    poses,tracking=track_waypoints(self.states[aid],references,count,self.dt,previous,
                        self.config.get('control_projection_limits'))
                    meta=dict(meta,physical_tracking=tracking,physical_tracking_reference_source='legacy_fallback_action',
                        trajectory_world_used=poses[:,:2].tolist(),yaws_world_used=poses[:,3].tolist(),
                        preview_positions_world=poses[:,:2].tolist(),preview_yaws_world=poses[:,3].tolist())
                    decision_by_actor[aid]['parse_meta']=meta
                # Use the same precise poses that appear in the trial preview.
                # Re-transforming float32 actions would corrupt finite-difference
                # acceleration/jerk and shift low-speed initial frames.
                points=np.asarray(meta['trajectory_world_used'][:count],dtype=np.float64)
                headings=np.asarray(meta['yaws_world_used'][:count],dtype=np.float64)
            if not np.isfinite(points).all() or not np.isfinite(headings).all():
                raise ValueError('Non-finite legacy action')
            actor = self.states[aid]
            previous = np.vstack(([actor['x'], actor['y']], points[:-1]))
            speeds = np.linalg.norm(points-previous, axis=1)/self.dt
            plans[aid] = [dict(actor, x=float(x), y=float(y), yaw=wrap(float(yaw)), speed=float(v))
                          for (x, y), yaw, v in zip(points, headings, speeds)]
        self.legacy_plan_time = self.time
        self.legacy_plans = plans
        return count, rows

    def _non_ego_step(self, aid, actor, before, next_frame):
        if self.config.get('background_mode','recorded')=='recorded' and aid not in self.anchor_participants:
            return super()._non_ego_step(aid,actor,before,next_frame)
        index = round((self.time-self.legacy_plan_time)/self.dt)
        return copy.deepcopy(self.legacy_plans[aid][index])
