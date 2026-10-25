"""Concise relative-coordinate interface for the legacy waypoint/assessor loop."""
import base64
import copy
import io
import json
import math

import numpy as np
import torch
from PIL import Image

from tbsim.policies.llm_adversary import _extract_json_object
from .legacy_waypoint_execution import LegacySavedGoalPolicy
from .anchor_execution_agent import AnchorExecutionTools
from .execution_agent import RelaxedReviewerClient, EXECUTION_REVIEW_PROMPT
from .geometry import clearance, project, wrap


PLANNER_SYSTEM = "You drive the attacking/auxiliary vehicle specified by controlled_vehicle, not the EGO. Execute the fixed collision objective and do not plan for the EGO.\nCoordinates are fixed to the EGO for this observation: x forward, y left (meters), heading_deg=0 same direction, +90 left. In the image, the EGO faces up.\nwaypoints are the controlled vehicle's future positions, not displacement from its own starting point; start from controlled_vehicle.xy and do not copy the EGO trajectory.\nIn the image, the BEFORE dashed box/AFTER solid vehicle are the measured positions before/after the previous round's execution, and t is simulation time; the blue box is the predicted EGO, and purple is a candidate, both starting from the current +0 seconds. The EGO replans every 0.1 seconds, and predictions or history are not guaranteed to reproduce; cutting in from far ahead may cause the EGO to decelerate, and it cannot be required to cooperate.\nRe-decide every 1 second, plan 3 seconds ahead, and execute only the first 1 second. Give 1\u20138 keypoints; a few are enough; t_s is a multiple of 0.1 seconds, strictly increasing, and the last point must be at least 1 second; do not output frame by frame.\nProceed continuously according to measured speed and heading, gradually accelerating/decelerating and turning; the angle between the first segment and the heading must not exceed 90\u00b0, and you cannot suddenly reverse, back up, or move laterally to patch in a collision. Check the entire path and vehicle width; do not cross the road's outer boundary or pass through isolation areas. Collisions with background vehicles are allowed and the simulation continues; background collisions cannot replace the objective of colliding with the EGO.\nSame-direction lane-change collision: first stay in the adjacent lane, decelerate gently, and if necessary stop and wait reasonably, letting the EGO approach side by side from the side before cutting in; do not brake suddenly. Do not cut in front of the EGO in advance and then decelerate or stop; this will usually be avoided by the EGO braking. Same-direction same-lane direct rear-end collisions from directly behind are prohibited; adjacent-lane diagonal contact with the side or rear corner is allowed. Opposite-direction intrusion does not use side-by-side waiting. After missing an opportunity, drive normally along the road and do not chase randomly.\nAfter review failure, the purple in the image is the previously rejected plan, which has not yet been executed; revise it by comparing with the feedback.\nDecide directly based on the current state and feedback; do not restate the objective or give a long analysis. Output only JSON: {\"reason\":\"one-sentence decision\",\"waypoints\":[{\"t_s\":2,\"x_m\":0,\"y_m\":0}]}."

ASSESSOR_SYSTEM = "You are the scenario execution reviewer, and you only review the first 1 second that will actually be executed in this round (0<t_s<=1), not the entire subsequent trajectory. Candidates after +1 second in the figure are not within the scope of this review and will be replanned later.\nCoordinates x forward/y left (meters), heading_deg=0 same direction, +90 left. BEFORE dashed box/AFTER solid car are the previous round's before/after measured values, t is simulation time; purple is an unexecuted candidate, blue/ego_plan is prediction, all starting from the current +0 second. EGO autonomously replans every 0.1 seconds, predictions or history are not guaranteed to recur, and predicted contact is not a measured collision.\nYour job is to check that the first 1 second of motion is reasonable and the scenario advances normally, not to require that a collision be completed within this second. Task progress may pass by default; only when it is clearly judged that the action in this 1 second will miss an existing collision opportunity and make the subsequent goal difficult to achieve should you judge task progress as false, and explain how the opportunity is missed by this 1 second. You must not reject merely because there is no collision, the vehicle is still going straight, or it has not yet changed lanes; do not suggest \"ensure a collision within 1 second.\"\nExample: ADV gently decelerates in the adjacent lane, waiting for EGO to approach side by side from the side; no cut-in or collision in this 1 second is reasonable preparation and should pass; if they are already side by side but it continues waiting, causing EGO to drive away and the opportunity to be lost, then reject for missing the timing.\nCheck whether the first 1 second has severe reversal/reversing/lateral movement/sudden speed change, the vehicle body crossing road boundaries, or passing through separation areas. segment_checks may be whole-segment statistics, and you cannot reject the first 1 second based solely on later-segment metrics.\nFor same-direction lane changes, it is better to wait until side by side before cutting in; do not cut early directly in front of EGO and decelerate to a stop, and rear-end collisions directly behind in the same lane are prohibited; oblique cuts into the side/rear corner from an adjacent lane, reasonable oncoming intrusion, and contact-location deviations are allowed. After missing, continue normally along the road and do not chase recklessly.\nCollisions with background vehicles are allowed, and you must not reject solely for that reason; background collisions do not substitute for the EGO goal.\nOutput only JSON: {\"pass\":true,\"failure_feedback\":\"\",\"revision_advice\":\"\"}. On failure, one sentence for the reason and one sentence for the modification advice targeting the first 1 second. You must not require modifying the initial position/speed or require EGO to cooperate."


def ego_xy(point, ego):
    c, s = math.cos(ego['yaw']), math.sin(ego['yaw'])
    dx, dy = float(point[0])-ego['x'], float(point[1])-ego['y']
    return [round(c*dx+s*dy, 2), round(-s*dx+c*dy, 2)]


def world_xy(x, y, ego):
    c, s = math.cos(ego['yaw']), math.sin(ego['yaw'])
    return [ego['x']+c*x-s*y, ego['y']+s*x+c*y]


class CompactWaypointPolicy(LegacySavedGoalPolicy):
    """Retain interpolation/checks/assessor feedback; compact model I/O only."""
    def bind_scene(self, world, view, episode=None):
        self.world = world
        self.view = view
        self.episode = episode
        self._rejected_previews = {}
        self._planner_rejected_previews = {}

    def _ego_history(self):
        """Only completed simulation samples; never candidate or recorded future."""
        episode=self.episode
        start=episode.action_history[-1]['time_s'] if episode and episode.action_history else self.world.time
        rows=[r for r in episode.trace if start-1e-8<=r['time_s']<=self.world.time+1e-8] if episode else []
        if len(rows)<2:
            return dict(source='no_previous_execution',points=[]),dict(points=[],replan_interval_s=.1)
        selected=[r for i,r in enumerate(rows) if i%5==0 or i==len(rows)-1]
        origin=self.world.states['ego']
        history=dict(source='previous_executed_interval',interval_sim_s=[start,self.world.time],
            points=[dict(t_rel_s=round(r['time_s']-self.world.time,3),
                xy=ego_xy([r['states']['ego']['x'],r['states']['ego']['y']],origin),
                speed_mps=round(r['states']['ego']['speed'],2),
                heading_deg=round(math.degrees(wrap(r['states']['ego']['yaw']-origin['yaw'])),1)) for r in selected])
        drawing=dict(points=[dict(t_s=r['time_s'],world_xy_m=[r['states']['ego']['x'],r['states']['ego']['y']],
            pose_world={key:r['states']['ego'][key] for key in ('x','y','yaw','length','width')})
            for r in rows],replan_interval_s=.1)
        return history,drawing

    def _state(self, aid):
        actor = self.world.states[aid]
        lane = self.view.catalog.nearest_lane(actor)
        return dict(id=self.view.actor_aliases[aid],
            xy=ego_xy([actor['x'],actor['y']], self.world.states['ego']),
            speed_mps=round(actor['speed'],2),
            heading_deg=round(math.degrees(wrap(actor['yaw']-self.world.states['ego']['yaw'])),1),
            size_m=[round(actor['length'],2),round(actor['width'],2)],
            lane=lane['lane_id'] if lane else None)

    def _execution_comparison(self, aid):
        """Previous completed interval endpoints, both in the current map frame."""
        episode=self.episode
        if not episode or not episode.action_history:return None
        start=episode.action_history[-1]['time_s']
        rows=[r for r in episode.trace if start-1e-8<=r['time_s']<=self.world.time+1e-8]
        if len(rows)<2:return None
        # Never pair an earlier state with an incomplete/candidate future.
        if abs(rows[0]['time_s']-start)>1e-6 or abs(rows[-1]['time_s']-self.world.time)>1e-6:
            return None
        actors={}
        for actor_id in ('ego',aid):
            if actor_id not in rows[0]['states'] or actor_id not in rows[-1]['states']:continue
            actors[actor_id]=dict(before=copy.deepcopy(rows[0]['states'][actor_id]),
                after=copy.deepcopy(rows[-1]['states'][actor_id]))
        return dict(start_s=rows[0]['time_s'],end_s=rows[-1]['time_s'],actors=actors)

    def _nearby_lanes(self, aid):
        ego = self.world.states['ego'];actor = self.world.states[aid]
        catalog = self.view.catalog
        preferred = []
        for state in (actor, ego):
            lane = catalog.nearest_lane(state)
            if lane:
                preferred.append(catalog.resolve(lane['lane_id']))
        for lid in list(preferred):
            preferred.extend(sorted(self.world.lanes[lid].next_lanes))
        ranked = []
        for lid, lane in self.world.lanes.items():
            _,_,_,d1 = project(lane.center.xy,[actor['x'],actor['y']])
            _,_,_,d2 = project(lane.center.xy,[ego['x'],ego['y']])
            if min(d1,d2)<=12:
                ranked.append((min(d1,d2),lid))
        result=[];seen=set()
        for lid in preferred+[lid for _,lid in sorted(ranked)]:
            if lid in seen or lid not in self.world.lanes:
                continue
            seen.add(lid);lane=self.world.lanes[lid]
            _,near,_,_=project(lane.center.xy,[actor['x'],actor['y']])
            points=lane.center.xy
            # Keep local geometry around the closest point, rather than a long
            # connector's unrelated far endpoints.
            k=int(np.argmin(np.linalg.norm(points-np.asarray(near),axis=1)))
            local=points[max(0,k-3):min(len(points),k+20)]
            indices=sorted({round(i*(len(local)-1)/4) for i in range(5)})
            result.append(dict(id=catalog.alias(lid),
                center_xy=[ego_xy(local[i],ego) for i in indices],
                next=[catalog.alias(x) for x in sorted(lane.next_lanes)[:3]]))
            if len(result)==6:
                break
        return result

    def _build_prompt_context(self, obs, ctrl_idx, ego_idx, step_index,
                              planning_state=None, failure_feedback=None):
        aid=self.actor_names[ctrl_idx]
        ego=self.world.states['ego'];actor=self.world.states[aid]
        participants=[]
        controlled=self._state(aid)
        for p in self.saved_goal['goal']['participants']:
            if p['actor_id']==aid:
                controlled['role']=p['role']
            else:
                participants.append(dict(self._state(p['actor_id']), role=p['role']))
        nearby=sorted((a for a in self.world.states if a not in self.world.anchor_participants and a!='ego'),
            key=lambda a:min(math.hypot(self.world.states[a]['x']-actor['x'],self.world.states[a]['y']-actor['y']),
                             math.hypot(self.world.states[a]['x']-ego['x'],self.world.states[a]['y']-ego['y'])))
        nearby=[a for a in nearby if min(math.hypot(self.world.states[a]['x']-actor['x'],self.world.states[a]['y']-actor['y']),
                                       math.hypot(self.world.states[a]['x']-ego['x'],self.world.states[a]['y']-ego['y']))<=25][:5]
        remaining=(planning_state or {}).get('remaining_waypoint_plan') or []
        feedback=failure_feedback or {}
        rejected=self._rejected_previews.get(aid)
        expected=dict(actor_id=aid,step_index=step_index,time_s=self.world.time)
        self._planner_rejected_previews[aid]=rejected if (
            rejected and rejected['key']==expected and feedback.get('_rejected_preview')==expected) else None
        feedback={k:feedback[k] for k in ('assessor_failure_feedback','assessor_revision_advice','failure_type',
            'validation_failure_reason','revision_advice','validation_stats','validation_details','violation_count') if feedback.get(k)}
        scene=dict(controlled_vehicle=controlled,ego_model=getattr(self.world,'config',{}).get('ego_model','strive'),goal=self.saved_goal['display_goal'],
            control=self.view.actor_aliases[aid],time_s=self.world.time,participants=participants,ego=self._state('ego'),
            nearby=[self._state(a) for a in nearby], lanes=self._nearby_lanes(aid),
            ego_executed_history=self._ego_history()[0],
            remaining=[dict(t_s=p['step']*.1,xy=ego_xy(p['position_world'],ego)) for p in remaining[:8]])
        if feedback:
            scene['feedback']=feedback
        if self._planner_rejected_previews[aid]:
            scene['image_note']="Purple is the candidate rejected in this review and not executed; please modify according to the feedback. Blue is the current EGO prediction, and both start from +0 seconds in this round."
        relative=self._state(aid)
        closing=ego['speed']-actor['speed']*math.cos(wrap(actor['yaw']-ego['yaw']))
        if abs(relative['heading_deg'])<45 and closing>.2 and relative['xy'][0]>=0:
            scene['parallel_eta_s']=round(relative['xy'][0]/closing,1)
            scene['parallel_eta_note']="Current speed extrapolation estimate must be based on new observations"
        return dict(compact_scene=scene,controlled_actor_id=aid,
            scenario_goal=self.saved_goal['goal']['language_goal'],
            trajectory_horizon=30,execution_horizon=self.executed_steps_per_call,step_dt_s=.1,
            adv_current_state=dict(position_world=[actor['x'],actor['y']],yaw_world=actor['yaw'],speed_mps=actor['speed']),
            ego_current_state=dict(position_world=[ego['x'],ego['y']],yaw_world=ego['yaw'],speed_mps=ego['speed']))

    def _build_prompt(self, context):
        return PLANNER_SYSTEM,json.dumps(context['compact_scene'],ensure_ascii=False,separators=(',',':'))

    def _parse_llm_target_plan(self, response_text, ctrl_idx, ego_idx, obs,
                               target_horizon, step_index, prior_stage_plan=None):
        parsed=_extract_json_object(response_text)
        # Native dry-run fixtures still use the archived world-coordinate schema.
        if isinstance(parsed,dict) and 'waypoints' in parsed:
            points=parsed['waypoints']
            try:
                if not isinstance(points,list) or not 1<=len(points)<=8:
                    raise ValueError("Output 1\u20138 waypoints")
                ego=dict(x=float(obs['centroid'][ego_idx,0]),y=float(obs['centroid'][ego_idx,1]),
                         yaw=float(obs['yaw'][ego_idx]))
                normalized=[]
                for point_index,p in enumerate(points,1):
                    if not isinstance(p,dict) or any(k not in p for k in ('t_s','x_m','y_m')):
                        raise ValueError(f'No.{point_index}waypoints are missing the t_s/x_m/y_m fields')
                    if any(type(p[k]) not in (int,float) or not math.isfinite(p[k]) for k in ('t_s','x_m','y_m')):
                        raise ValueError(f'No.{point_index}The t_s/x_m/y_m of each waypoint must be finite numeric values')
                    if not .1<=p['t_s']<=3 or abs(p['t_s']*10-round(p['t_s']*10))>1e-7:
                        raise ValueError(f"No.{point_index}waypoints' t_s={p['t_s']}Invalid: must be within [0.1, 3.0] seconds and a multiple of 0.1 seconds; t_s is counted from the current moment, not absolute simulation time")
                    normalized.append(dict(step=round(p['t_s']*10),
                        position_world=world_xy(p['x_m'],p['y_m'],ego)))
                for i,(a,b) in enumerate(zip(points,points[1:]),2):
                    if a['t_s']>=b['t_s']:
                        raise ValueError(f"No.{i}waypoints' t_s={b['t_s']}Not greater than previous point{a['t_s']}; time must be strictly increasing")
                if normalized[-1]['step']<self.executed_steps_per_call:
                    raise ValueError(f"End point t_s={points[-1]['t_s']}seconds insufficient: at least cover{self.executed_steps_per_call * 0.1:g}second execution interval")
                # The retained parser expects uniformly spaced points. Fill
                # its internal format along exactly the same piecewise lines;
                # the model still supplies only the sparse critical points.
                start=[float(obs['centroid'][ctrl_idx,0]),float(obs['centroid'][ctrl_idx,1])]
                knots=[dict(step=0,position_world=start)]+normalized
                dense=[]
                for step in range(self.waypoint_interval_steps,normalized[-1]['step']+1,self.waypoint_interval_steps):
                    left,right=next((a,b) for a,b in zip(knots,knots[1:]) if a['step']<step<=b['step'])
                    fraction=(step-left['step'])/(right['step']-left['step'])
                    dense.append(dict(step=step,position_world=[
                        a+fraction*(b-a) for a,b in zip(left['position_world'],right['position_world'])]))
                translated=dict(anchor_plan_reason=str(parsed.get('reason','')),
                                adv_anchor_plan=dense)
                result=super()._parse_llm_target_plan(json.dumps(translated,ensure_ascii=False),
                    ctrl_idx,ego_idx,obs,target_horizon,step_index,prior_stage_plan)
                result[2]['compact_response']=parsed
                result[2]['coordinate_origin']=ego
                result[2]['sparse_waypoints_world']=normalized
                if getattr(getattr(self,'world',None),'config',{}).get('physical_waypoint_tracking'):
                    result[2]['physical_tracking']=copy.deepcopy(self._physical_rollout_meta)
                if result[0] is not None and result[1] is not None:
                    # Copy the full candidate, then use the actual returned
                    # action for the executable prefix (including parser trims).
                    preview_xy=copy.deepcopy(result[2]['trajectory_world_used'])
                    preview_yaw=copy.deepcopy(result[2]['yaws_world_used'])
                    if getattr(getattr(self,'world',None),'config',{}).get('physical_waypoint_tracking'):
                        actual_xy=preview_xy[:len(result[0])]
                        actual_yaw=preview_yaw[:len(result[1])]
                    else:
                        actual_xy=self._to_world(result[0],obs['world_from_agent'][ctrl_idx]).detach().cpu().tolist()
                        actual_yaw=(result[1][:,0]+obs['yaw'][ctrl_idx]).detach().cpu().tolist()
                    preview_xy[:len(actual_xy)]=actual_xy
                    preview_yaw[:len(actual_yaw)]=actual_yaw
                    result[2]['preview_positions_world']=preview_xy
                    result[2]['preview_yaws_world']=preview_yaw
                return result
            except (KeyError,TypeError,ValueError) as exc:
                return None,None,dict(status='invalid_compact_waypoint',validation_failure_reason=str(exc),parsed_response=parsed)
        return super()._parse_llm_target_plan(response_text,ctrl_idx,ego_idx,obs,
                                               target_horizon,step_index,prior_stage_plan)

    def _build_local_validation_failure_feedback(self, parse_meta):
        reason=str(parse_meta.get('validation_failure_reason') or parse_meta.get('status'))
        if parse_meta.get('status')=='invalid_compact_waypoint':
            return dict(failure_type='local_format',validation_failure_reason=reason,
                revision_advice="Correct the fields or times according to the points indicated above, and return the complete waypoints; keep the points that are still valid. This candidate was not executed.")
        details=[]
        for segment in parse_meta.get('waypoint_segment_summaries',[]):
            for metric,limit_key,label,unit in [
                ('segment_accel_mps2','validation_max_abs_accel_limit_mps2',"\u5206\u6bb5\u901f\u5ea6\u53d8\u5316\u7387",'m/s²'),
                ('segment_yaw_rate_rps','validation_max_abs_yaw_rate_limit_rps',"\u5206\u6bb5\u671d\u5411\u53d8\u5316\u7387",'rad/s')]:
                value=segment[metric];limit=parse_meta.get(limit_key)
                if limit is not None and abs(value)>limit:
                    dt=segment['duration_s']/max(1,segment['to_step']-segment['from_step'])
                    details.append(dict(interval_s=[round(segment['from_step']*dt,3),round(segment['to_step']*dt,3)],
                        metric=label,value=round(value,3),absolute_limit=round(limit,3),unit=unit,
                        previous_speed_mps=round(segment['speed_mps']-segment['speed_delta_mps'],3),
                        segment_speed_mps=round(segment['speed_mps'],3),distance_m=round(segment['distance_m'],3)))
        if details:
            message='；'.join(f"{d['interval_s'][0]:g}–{d['interval_s'][1]:g}Seconds{d['metric']}={d['value']:g}{d['unit']}, absolute value upper limit{d['absolute_limit']:g}" for d in details[:3])
            advice="These are local metrics computed from per-segment average speed/heading, not executed measurements. Adjust the corresponding segment endpoint displacement or steering so that speed and direction transition segment by segment; the first segment starts from the current measured speed/vehicle heading. Return the complete plan."
        else:
            labels={'max_step_distance_exceeded':"Displacement between adjacent simulation steps exceeds limit",'max_speed_exceeded':"Trajectory speed exceeds limit",
                'non_finite_world_points':"Trajectory contains non-finite coordinates",'insufficient_world_points':"Insufficient valid trajectory points",
                'dynamics_limit_exceeded':"Step-by-step acceleration or steering speed exceeds limit"}
            message=labels.get(reason,reason)
            advice="Check the local error against the current vehicle starting point and correct the corresponding trajectory; this candidate was not executed."
        stats={k:round(float(parse_meta[k]),3) for k in ('max_step_distance_m','max_step_distance_limit_m',
            'max_speed_mps','validation_max_speed_limit_mps','max_abs_accel_mps2','validation_max_abs_accel_limit_mps2',
            'max_abs_yaw_rate_rps','validation_max_abs_yaw_rate_limit_rps') if parse_meta.get(k) is not None}
        return dict(failure_type='local_motion',validation_failure_reason=message,revision_advice=advice,
            validation_stats=stats,validation_details=details[:3],violation_count=len(details))

    def _validate_waypoint_segment_dynamics(self, ctrl_idx, obs, waypoints_world):
        # Preserve the old half-second check scale when the internal parser
        # grid becomes .1s. The finer grid represents extra knots, not stricter
        # acceleration/turning gates introduced through shorter denominators.
        checked=[p for p in waypoints_world if p['step']%5==0]
        if waypoints_world and (not checked or checked[-1]['step']!=waypoints_world[-1]['step']):
            checked.append(waypoints_world[-1])
        stats=super()._validate_waypoint_segment_dynamics(ctrl_idx,obs,checked)
        if getattr(getattr(self,'world',None),'config',{}).get('physical_waypoint_tracking'):
            # Raw knots are reference positions. The actual control sweep below
            # enforces stricter Stage 2 bounds, rather than judging raw segments
            # as though they were the trajectory that will be executed.
            stats['reference_segment_validation_status']=stats.get('waypoint_segment_validation_status')
            stats['waypoint_segment_validation_status']='projected_before_execution'
        return stats

    def _rollout_world_points_from_waypoints(self,ctrl_idx,obs,waypoints_world,rollout_horizon_steps):
        if not getattr(getattr(self,'world',None),'config',{}).get('physical_waypoint_tracking'):
            return super()._rollout_world_points_from_waypoints(ctrl_idx,obs,waypoints_world,rollout_horizon_steps)
        from .feasible_waypoints import previous_executed_control,track_waypoints
        aid=self.actor_names[ctrl_idx];actor=self.world.states[aid];dt=self.world.dt
        previous=previous_executed_control(self.episode.trace,aid,self.world.time,dt)
        poses,meta=track_waypoints(actor,waypoints_world,rollout_horizon_steps,dt,previous,
            self.world.config.get('control_projection_limits'))
        self._physical_rollout_meta=meta
        xy=torch.as_tensor(poses[:,:2],dtype=torch.float64)
        yaw=torch.as_tensor(poses[:,3],dtype=torch.float64)
        velocity=torch.stack((torch.cos(yaw),torch.sin(yaw)),-1)*torch.as_tensor(poses[:,2,None])
        return xy,yaw,velocity,dict(trajectory_mode=meta['method'],returned_steps=len(poses),step_dt_s=dt)

    def _validate_world_points(self,world_points,ctrl_idx,obs,target_horizon,**kwargs):
        if not getattr(getattr(self,'world',None),'config',{}).get('physical_waypoint_tracking'):
            return super()._validate_world_points(world_points,ctrl_idx,obs,target_horizon,**kwargs)
        # Preserve stationary initial frames and double-precision poses. The
        # old guard trims leading stationary points, which changes their times.
        actor=self.world.states[self.actor_names[ctrl_idx]]
        points=world_points[:target_horizon]
        delta=points-torch.tensor([actor['x'],actor['y']],dtype=torch.float64)
        c,s=math.cos(actor['yaw']),math.sin(actor['yaw'])
        local=torch.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]),-1)
        distances=torch.linalg.norm(torch.diff(torch.cat((torch.tensor([[actor['x'],actor['y']]],dtype=torch.float64),points)),dim=0),dim=-1)
        cfg=self._physical_rollout_meta['limits']
        valid=bool(torch.isfinite(points).all()) and bool((distances/self.world.dt<=cfg['speed_max']+1e-6).all())
        stats=dict(validation_status='ok' if valid else 'failed',
            validation_failure_reason=None if valid else 'invalid_projected_motion',
            first_step_distance_m=float(distances[0]),mean_step_distance_m=float(distances.mean()),
            max_step_distance_m=float(distances.max()),max_speed_mps=float(distances.max()/self.world.dt),
            physical_projection_enforced=True)
        return (local,torch.zeros((len(points),1),dtype=torch.float64),stats) if valid else (None,None,stats)

    def _extract_executed_rollout_summary(self, obs, ctrl_idx, ego_idx, parse_meta):
        preview_meta=dict(parse_meta,
            trajectory_world_used=parse_meta.get('preview_positions_world',parse_meta.get('trajectory_world_used')),
            yaws_world_used=parse_meta.get('preview_yaws_world',parse_meta.get('yaws_world_used')))
        summary=super()._extract_executed_rollout_summary(obs,ctrl_idx,ego_idx,preview_meta)
        if summary is not None:
            summary['llm_reply_executed_segment']['sparse_waypoints_world']=parse_meta.get('sparse_waypoints_world')
            actor=self.world.states[self.actor_names[ctrl_idx]]
            xy=parse_meta.get('preview_positions_world',parse_meta.get('trajectory_world_used',[]))
            yaw=parse_meta.get('preview_yaws_world',parse_meta.get('yaws_world_used',[]))
            if len(xy)!=len(yaw):raise ValueError('Candidate preview position/yaw lengths differ')
            summary['candidate_body_poses_world']=[dict(t_s=0.,pose_world=copy.deepcopy(actor))]+[
                dict(t_s=round((i+1)*self.world.dt,8),pose_world=dict(actor,x=point[0],y=point[1],yaw=wrap(heading)))
                for i,(point,heading) in enumerate(zip(xy,yaw))]
        return summary

    def _build_plan_assessor_prompt(self, planner_context, rollout_summary):
        ego=self.world.states['ego'];aid=planner_context['controlled_actor_id']
        scene=planner_context['compact_scene']
        def sample(rows):
            return [dict(t_s=round((i+1)*.1,2),xy=ego_xy(r['position_world'],ego),
                         heading_deg=round(math.degrees(wrap(r['yaw_world']-ego['yaw'])),1))
                    for i,r in enumerate(rows) if (i+1)%5==0 or i==len(rows)-1]
        adv_rows=rollout_summary['adv_rollout']['trajectory_world']
        ego_rows=rollout_summary['ego_rollout']['trajectory_world']
        gaps=[]
        for a,e in zip(adv_rows,ego_rows):
            adv=dict(self.world.states[aid],x=a['position_world'][0],y=a['position_world'][1],yaw=a['yaw_world'])
            target=dict(ego,x=e['position_world'][0],y=e['position_world'][1],yaw=e['yaw_world'])
            gaps.append(clearance(adv,target))
        dynamics=rollout_summary['rollout_dynamics_summary']
        data=dict(controlled_vehicle=scene['controlled_vehicle'],goal=scene['goal'],control=scene['control'],ego=scene['ego'],
            participants=scene['participants'],nearby=scene['nearby'],
            proposed_adv=sample(adv_rows),ego_plan=sample(ego_rows),
            ego_plan_source='current_state_forecast_not_executed',ego_replan_interval_s=.1,
            ego_executed_history=scene['ego_executed_history'],
            forecast_contact=bool(rollout_summary.get('local_collision_meta')),
            min_body_gap_m=round(float(min(gaps)),2) if gaps else None,
            segment_checks={k:dynamics[k] for k in ('max_waypoint_segment_abs_accel_mps2',
                'max_waypoint_segment_abs_yaw_rate_rps','max_waypoint_speed_jump_mps','max_waypoint_heading_jump_rad')
                if dynamics.get(k) is not None})
        if 'parallel_eta_s' in scene:
            data['parallel_eta_s']=scene['parallel_eta_s']
        return ASSESSOR_SYSTEM,json.dumps(data,ensure_ascii=False,separators=(',',':'))

    def _ego_forecast_drawing(self):
        plan=copy.deepcopy(self.world.ego_current_plan())
        ego=self.world.states['ego']
        current=dict(t_s=self.world.time,world_xy_m=[ego['x'],ego['y']],pose_world={
            k:ego[k] for k in ('x','y','yaw','length','width')})
        future=[]
        for point in plan['points']:
            if point['t_s']<=self.world.time+1e-8:continue
            point['pose_world']=dict(current['pose_world'],x=point['world_xy_m'][0],
                y=point['world_xy_m'][1],yaw=point['yaw_rad'])
            future.append(point)
        plan.update(points=[current]+future,show_body_preview=True)
        return plan

    def _build_bev_data_url(self, obs, ctrl_idx, step_index, scene_index,
                             ego_idx=None, overlay_mode='planner', rollout_summary=None):
        aid=self.actor_names[ctrl_idx]
        rejected=self._planner_rejected_previews.get(aid)
        if overlay_mode=='planner' and rejected and rejected['key']==dict(
                actor_id=aid,step_index=step_index,time_s=self.world.time):
            return rejected['image'],dict(rejected['meta'],bev_overlay_mode='planner_rejected_candidate',
                image_source='assessor_rejected_candidate')
        plans=None
        if overlay_mode=='assessor' and rollout_summary:
            segment=rollout_summary['llm_reply_executed_segment']
            points=segment.get('sparse_waypoints_world') or segment.get('waypoints_world') or []
            plans={aid:dict(created_at_s=self.world.time,body_poses=rollout_summary['candidate_body_poses_world'],points=[
                dict(t_s=0.,position_world_m=[self.world.states[aid]['x'],self.world.states[aid]['y']])]+
                [dict(t_s=p['step']*.1,position_world_m=p['position_world']) for p in points])}
        distance=math.hypot(*self._state(aid)['xy'])
        radius=max(25.,min(65.,distance+10))
        if plans:
            radius=max(radius,min(75.,max(math.hypot(*ego_xy(p['position_world'],self.world.states['ego']))
                for p in points)+5))
        comparison=self._execution_comparison(aid)
        frame=self.view.view_scene(radius_m=radius,
            highlight_actor_ids=[self.view.actor_aliases[aid]],
            ego_plan=self._ego_forecast_drawing(),anchor_preview=plans,
            execution_comparison=comparison)
        with Image.open(frame['image_path']) as image:
            image=image.convert('RGB');image.thumbnail((768,768))
            stream=io.BytesIO();image.save(stream,format='JPEG',quality=85)
        return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode(),dict(
            bev_attached=True,bev_status='ok',bev_overlay_mode=overlay_mode,
            original_image=frame['image_name'],coordinates='current fixed EGO forward/left',
            measured_comparison_interval_s=[comparison['start_s'],comparison['end_s']] if comparison else None)

    def _query_plan_assessor(self, planner_context, rollout_summary, step_index,
                             assessor_image_data_url=None, assessor_bev_meta=None):
        result=super()._query_plan_assessor(planner_context,rollout_summary,step_index,
            assessor_image_data_url,assessor_bev_meta)
        aid=planner_context['controlled_actor_id']
        self._rejected_previews.pop(aid,None)
        if not result.get('passed',True) and assessor_image_data_url and isinstance(
                result.get('failure_feedback_payload'),dict):
            key=dict(actor_id=aid,step_index=step_index,time_s=self.world.time)
            self._rejected_previews[aid]=dict(key=key,image=assessor_image_data_url,
                meta=copy.deepcopy(assessor_bev_meta or {}))
            result['failure_feedback_payload']['_rejected_preview']=key
        return result


class CompactLegacyExecutionTools(AnchorExecutionTools):
    def __init__(self, world, view, saved, reviewer, root):
        super().__init__(world, view, saved, reviewer, root)
        self.episode.client=CompactFinalReviewer(reviewer,self.review_motion_evidence,
            compact=world.config.get('compact_final_review',False))

    def observe(self):
        observation=super().observe()
        observation['anchor_contract']['max_anchors_per_vehicle']=8
        return observation

    def review_motion_evidence(self):
        evidence=super().review_motion_evidence()
        points=np.array([[state['x'],state['y']] for row in self.episode.trace
            for aid,state in row['states'].items() if aid in self.episode.monitor.participants or aid=='ego'])
        def distance(lane):
            xy=np.array(lane['centerline_world_m'])
            return float(np.min(np.linalg.norm(xy[:,None,:]-points[None,:,:],axis=-1)))
        evidence['road_context']['lanes']=sorted(evidence['road_context']['lanes'],key=distance)[:12]
        evidence['road_context']['note']="Only the 12 roads closest to the execution path are provided; roads not provided cannot be judged as non-drivable on that basis."
        evidence['vehicle_sizes_m']={aid:[actor['length'],actor['width']] for aid,actor in self.world.states.items()
            if aid in self.episode.monitor.participants or aid=='ego'}
        return evidence


class CompactFinalReviewer(RelaxedReviewerClient):
    def complete(self, prompt, payload):
        payload=dict(payload,**self.evidence())
        if self.compact:
            from .review_evidence import REVIEW_SYSTEM,compact_review_payload
            return self.client.complete(REVIEW_SYSTEM,compact_review_payload(payload))
        # Full 0.1-second measured motion replaces the duplicate sampled trace.
        payload.pop('executed_trace',None)
        measurements=payload.get('descriptive_measurements') or {}
        events=measurements.get('events',[])
        payload['contact_identity']=dict(ego_actor_id='ego',
            confirmed_ego_contacts=[e for e in events if e.get('kind')=='collision' and e.get('other_id')=='ego'],
            non_ego_contact_count=sum(e.get('kind')=='collision' and e.get('other_id')!='ego' for e in events),
            note="Only events whose other_id is exactly ego are EGO contacts; other long IDs are other vehicles. A near miss without an EGO event cannot be called a collision.")
        return self.client.complete(EXECUTION_REVIEW_PROMPT,payload)
