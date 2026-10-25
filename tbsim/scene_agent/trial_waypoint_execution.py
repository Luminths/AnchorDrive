"""Trial the configured joint execution interval before asking the Assessor."""
import base64
import copy
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tbsim.policies.llm_adversary import _extract_json_object
from .backend import ForbiddenRearEnd
from .compact_waypoint_execution import CompactWaypointPolicy, ego_xy
from .geometry import corners, wrap, relative
from .road_evidence import trial_road_evidence
from .planning_cues import motion_cues, ego_forecast_cues, measured_motion_summary, straight_speed_references
from .interaction_cues import forecast_interaction
from .prior_execution_cues import prior_execution_reference,prior_gap_reference
from .assessor_input import readable_assessor_input

from .planner_input import PLANNER_SYSTEM as POST_PLANNER_SYSTEM, readable_planner_input

POST_ASSESSOR_SYSTEM = Path(__file__).with_name('posttrial_assessor.md').read_text()


def call_trial_assessor(client, user, image, max_format_retries=1):
    """Repair invalid JSON on the same measured trial; never retry a verdict."""
    request=user
    for attempt in range(max_format_retries+1):
        raw=client.complete(POST_ASSESSOR_SYSTEM,request,image)
        try:
            verdict=_extract_json_object(raw)
        except (ValueError,TypeError):
            verdict=None
        if isinstance(verdict,dict) and type(verdict.get('pass')) is bool:
            return verdict,raw,attempt
        if attempt==max_format_retries:
            raise ValueError('Post-trial Assessor must return a JSON boolean pass; format repair exhausted')
        request=user+"\nThe last reply could not be parsed as JSON containing a boolean pass. Only fix the output format, and still independently review the same actual trial run above; it is not required to change it to pass, and the simulation is not rerun. Output the complete JSON, and string quotes must be closed."
    raise AssertionError('Unreachable format repair loop')


def trial_relative_samples(trial):
    """Measured same-time geometry, without an automatic opportunity verdict."""
    return {aid:[dict(t_s=round(row['time_s']-trial['start_s'],3),
                     **relative(row['states'][aid],row['states']['ego']))
                 for i,row in enumerate(trial['trace']) if i%5==0 or i==len(trial['trace'])-1]
            for aid in trial['plans']}


def native_trial_geometry(trial,aid,origin):
    """Explain measured body separation, without choosing or rejecting actions."""
    from shapely.geometry import Polygon
    from shapely.ops import nearest_points
    samples=[]
    for row in trial['trace']:
        a,b=nearest_points(Polygon(corners(row['states'][aid])),Polygon(corners(row['states']['ego'])))
        dx,dy=b.x-a.x,b.y-a.y;c,s=math.cos(origin['yaw']),math.sin(origin['yaw'])
        samples.append(dict(t_s=round(row['time_s']-trial['start_s'],3),
            body_gap_m=round(math.hypot(dx,dy),3),
            gap_toward_ego_xy_m=[round(c*dx+s*dy,3),round(-s*dx+c*dy,3)]))
    return dict(closest=min(samples,key=lambda p:p['body_gap_m']),end=samples[-1],
        note="The difference between the closest points on the vehicle bodies in this actual trial run; the vector points from the controlled vehicle to EGO, using the fixed coordinates of this round's starting point. It explains a response that has already occurred, cannot directly translate the current waypoint, and does not mean that the next EGO will necessarily repeat it.")


class PostTrialWaypointPolicy(CompactWaypointPolicy):
    def bind_scene(self, world, view, episode=None):
        super().bind_scene(world,view,episode)
        self.trial_feedback={}
        self.failed_trial=None
        self.drawing_trial=None
        self.remaining_plans={}
        self.round_origin=copy.deepcopy(world.states['ego'])
        self.enable_local_collision_termination=False

    def _build_prompt_context(self, obs, ctrl_idx, ego_idx, step_index, planning_state=None, failure_feedback=None):
        aid=self.actor_names[ctrl_idx]
        context=super()._build_prompt_context(obs,ctrl_idx,ego_idx,step_index,planning_state,
            self.trial_feedback.get(aid) or failure_feedback)
        context['compact_scene']['motion_cues']=motion_cues(self.world.states[aid],self.world.states['ego'])
        context['compact_scene']['ego_plan']=ego_forecast_cues(self.world)
        context['compact_scene']['interaction_reference']=forecast_interaction(
            self.world.states[aid],self.world.states['ego'],context['compact_scene']['ego_plan'])
        context['compact_scene']['scenario_roles']=[dict(id=self.view.actor_aliases[p['actor_id']],role=p['role']) for p in self.saved_goal['goal']['participants']]
        if self.world.config.get('physical_waypoint_tracking'):
            from .feasible_waypoints import previous_executed_control
            from tbsim.utils.control_projection import DEFAULT_LIMITS
            trace=getattr(getattr(self,'episode',None),'trace',[])
            context['compact_scene']['execution_physics']=dict(
                limits=dict(DEFAULT_LIMITS),
                previous_executed_control=previous_executed_control(trace,aid,self.world.time,self.world.dt),
                control_units=['acceleration_mps2','yaw_rate_radps'],
                note="Waypoints are for reference; actual tracking follows continuous vehicle-body motion. These are the current execution boundaries, not actions that can be output directly.")
            context['compact_scene']['straight_speed_reference']=straight_speed_references(
                self.world.states[aid],self.world.states['ego'],
                context['compact_scene']['execution_physics']['previous_executed_control'][0],self.world.dt)
        geometry=native_trial_geometry(dict(start_s=self.world.time,
            trace=[dict(time_s=self.world.time,states=self.world.states)]),aid,self.world.states['ego'])
        context['compact_scene']['current_body_geometry']=geometry['closest']
        # Descriptive measured evidence; never an action/acceptance gate.
        ep=getattr(self,'episode',None)
        if ep and ep.action_history:
            start=ep.action_history[-1]['time_s']
            rows=[r for r in ep.trace if start-1e-8<=r['time_s']<=self.world.time+1e-8]
            if len(rows)>1 and abs(rows[-1]['time_s']-self.world.time)<1e-6:
                trial=dict(start_s=start,trace=rows)
                last_action=ep.action_history[-1]
                for decision in last_action.get('legacy_decisions',[]):
                    if self.actor_names[decision['ctrl_agent_index']]!=aid:continue
                    assessment=decision.get('assessor_meta') or {}
                    if not assessment:continue
                    verdict=assessment.get('parsed_response') or {}
                    context['compact_scene']['last_interval_assessment']=dict(
                        passed=bool(assessment.get('passed')),
                        soft_fallback=bool(last_action.get('soft_fallback')),
                        reason=verdict.get('reason',''),
                        failure_feedback=verdict.get('failure_feedback',''),
                        revision_advice=verdict.get('revision_advice',''),
                        note="Model review comments on the plan submitted in the previous execution interval; it may still be rejected and the old fallback executed. When it conflicts with measurements, verify against measurements; it is not approval of the current candidate.")
                context['compact_scene']['latest_accepted_execution']=dict(
                    interval_sim_s=[start,self.world.time],
                    controlled=measured_rows(rows,aid,self.world.states['ego']),
                    ego=measured_rows(rows,'ego',self.world.states['ego']),
                    motion_summary=measured_motion_summary(rows,aid),
                    body_geometry=native_trial_geometry(trial,aid,self.world.states['ego']),
                    note="Measurements submitted in the previous execution interval; positions are uniformly converted to the current EGO fixed coordinates, and t_s is calculated from the start of that completed interval, not the current future.")
        peers={}
        if 'scene_index' in obs and 'track_id' in obs:
            for index,peer in enumerate(self.actor_names):
                if peer==aid or peer not in self.world.anchor_participants:continue
                key=(int(obs['scene_index'][index]),int(obs['track_id'][index]))
                cached=getattr(self,'_last_plan_cache_by_agent',{}).get(key,{})
                if cached.get('step_index')!=step_index:continue
                positions=cached.get('trajectory_world_used') or []
                points=[dict(t_s=round((i+1)*self.world.dt,3),xy=ego_xy(positions[i],self.world.states['ego']))
                    for i in (4,9,19,29) if i<len(positions)]
                if points:peers[self.view.actor_aliases[peer]]=points
        if peers:context['compact_scene']['peer_candidate_plans']=peers
        if self.world.config.get('prior_attempt_lessons'):
            lesson=dict(self.world.config['prior_attempt_lessons'])
            reference=lesson.pop('reference_trace',None)
            gaps=lesson.pop('closest_by_actor',{})
            for key in ('reference_source','closest_relative','previous_attempts'):
                lesson.pop(key,None)
            context['compact_scene']['previous_experiment']=lesson
            gap_reference=prior_gap_reference(gaps,aid,self.world.states['ego'])
            if gap_reference:context['compact_scene']['prior_gap_reference']=gap_reference
            if reference:
                context['compact_scene']['prior_execution_reference']=prior_execution_reference(
                    reference,aid,self.world.states['ego'],self.world.time)
                ego_reference=prior_execution_reference(reference,'ego',self.world.states['ego'],self.world.time)
                if ego_reference['available']:
                    context['compact_scene']['prior_ego_response']=ego_reference

        if self.failed_trial:
            context['compact_scene']['ego_trial_execution']=measured_rows(
                self.failed_trial['trace'],'ego',self.world.states['ego'])
            context['compact_scene']['controlled_trial_execution']=measured_rows(
                self.failed_trial['trace'],aid,self.world.states['ego'])
            context['compact_scene']['rejected_trial_geometry']=native_trial_geometry(
                self.failed_trial,aid,self.world.states['ego'])
            context['compact_scene']['rejected_trial_motion']=measured_motion_summary(self.failed_trial['trace'],aid)
            context['compact_scene']['other_trial_endpoints']={self.view.actor_aliases[a]:
                measured_rows(self.failed_trial['trace'],a,self.world.states['ego'])[-1]
                for a in self.failed_trial['plans'] if a!=aid}
            context['compact_scene']['image_note']="Red is the actual trial run of the rejected candidate; the state has been restored to this round's starting point. Blue is the measured EGO of that trial run, not the inevitable next trajectory."
        return context

    def _build_prompt(self, context):
        scene=dict(context['compact_scene'])
        config=self.world.config
        interval=float(config.get('decision_interval_s',1.))
        scene['execution_interval_s']=interval
        scene['remaining_time_s']=max(0.,float(config.get('max_simulation_s',6.))-float(scene.get('time_s',0.)))
        scene['ego_replan_interval_s']=float(config.get('ego_replan_interval_s',.1))
        system=POST_PLANNER_SYSTEM
        if interval!=1.:
            system=system.replace("Replan every 1 second and execute only the first 1 second",f'Replan every {interval:g} seconds and execute only the first {interval:g} seconds')
            system=system.replace("plan sparse keypoints for the next 1\u20133 seconds",f'plan sparse keypoints for the next {interval:g}–3 seconds')
            system=system.replace("last point must be >=1.0 second",f'last point must be >={interval:g} seconds')
            system=system.replace('"t_s":1.0,"x_m":12.0',f'"t_s":{interval},"x_m":{10+2*interval}')
        if scene['ego_replan_interval_s']!=.1:
            system=system.replace("EGO every 0.1 seconds",f"EGO every{scene['ego_replan_interval_s']:g}Seconds")
        if config.get('physical_waypoint_tracking'):
            system=system.replace("Local segment checks limit the speed change rate to 12 m/s\u00b2 and the heading change rate to 2 rad/s; the upper limits need not be fully used.",
                "Actually track waypoints according to the continuous vehicle model; points are for reference only: acceleration \u22124 to 3 m/s\u00b2, jerk \u22646 m/s\u00b3, steering rate \u22640.5 rad/s, steering acceleration \u22640.8 rad/s\u00b2, curvature \u22640.15/m; when stationary, the vehicle cannot turn in place and is corrected by measurements.")
        else:
            yaw_limit=float(config.get('validation_max_abs_yaw_rate_rps',2.))
            system=system.replace("heading change rate to 2 rad/s",f'heading change rate to {yaw_limit:g} rad/s')
        if not config.get('planner_send_bev',True):
            system='\n'.join(line for line in system.splitlines() if not line.startswith("In the image, the gray START"))
            system+="\nThis group receives only numerical values and road geometry; the centerline is not the complete vehicle body boundary, and road gaps must not be treated as crossable pavement."
        # Provider routing keeps its measured scalar outside the model text.
        bridge=getattr(self,'_client',None)
        if bridge is not None:
            bridge.planner_body_gap_m=(scene.get('current_body_geometry') or {}).get('body_gap_m')
            bridge.planner_request_meta=dict(time_s=scene.get('time_s'),control=scene.get('control'))
        return system,readable_planner_input(scene)

    def _query_plan_assessor(self,*args,**kwargs):
        # Keep the old local parser/replanning loop, defer all paid assessment.
        return dict(status='deferred_until_native_trial',passed=True)

    def _build_bev_data_url(self,obs,ctrl_idx,step_index,scene_index,ego_idx=None,overlay_mode='planner',rollout_summary=None):
        if overlay_mode=='assessor':return None,dict(bev_attached=False,status='deferred_until_native_trial')
        if not self.world.config.get('planner_send_bev',True):
            return None,dict(bev_attached=False,status='text_only_planner')
        return self.trial_image('planner')

    def trial_image(self, phase):
        participants=sorted(self.world.anchor_participants);ids=['ego']+participants
        trial=self.drawing_trial if phase=='assessor' else self.failed_trial
        history=self.episode.trace
        if phase=='assessor' and trial:
            history=[r for r in history if r['time_s']<=trial['start_s']+1e-8]
        layers=dict(phase=phase,initial={a:copy.deepcopy(self.episode.trace[0]['states'][a]) for a in ids},
            history=copy.deepcopy(history),trial=trial,
            plans={} if trial or phase=='assessor' else self.remaining_plans,ids=ids,
            now_s=self.world.time,origin_s=trial['start_s'] if trial else self.world.time,
            ego_forecast=None if trial else self.world.ego_current_plan())
        camera=self.round_origin
        positions=[s for r in history for a,s in r['states'].items() if a in ids]
        positions+=list(self.world.states[a] for a in ids)+list(layers['initial'].values())
        positions += [p['pose'] for poses in layers['plans'].values() for p in poses]
        if layers['ego_forecast']:
            positions += [dict(x=p['world_xy_m'][0],y=p['world_xy_m'][1]) for p in layers['ego_forecast']['points']]
        if trial:positions+=[s for r in trial['trace'] for a,s in r['states'].items() if a in ids]
        radius=max(25.,min(75.,max(math.hypot(*ego_xy([s['x'],s['y']],camera)) for s in positions)+8.))
        if trial:
            # Reserve the top legend and bottom key, including complete bodies.
            # A center inside the square can still hide under the top legend.
            local=[ego_xy(point,camera) for pose in positions for point in corners(pose)]
            radius=min(75.,max(radius,max(p[0] for p in local)/.58,
                -min(p[0] for p in local)/.78,max(abs(p[1]) for p in local)/.9))
        frame=self.view.view_scene(radius_m=radius,highlight_actor_ids=[self.view.actor_aliases[a] for a in participants],
            execution_layers=layers,camera_ego=camera)
        with Image.open(frame['image_path']) as im:
            im=im.convert('RGB');im.thumbnail((768,768));stream=io.BytesIO();im.save(stream,format='JPEG',quality=85)
        return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode(),dict(
            bev_attached=True,original_image=frame['image_name'],bev_overlay_mode=phase,
            trial_interval_s=[trial['start_s'],trial['end_s']] if trial else None,
            future_adv_plan_shown=phase!='assessor' and bool(layers['plans']),
            ego_trajectory_source='native_trial_execution' if trial else 'current_state_forecast',
            filled_vehicle_time_s=self.world.time,
            trial_endpoint_style='pale outlined bodies' if trial and phase=='planner' else 'current filled bodies',
            scene_initial_poses=True,accepted_history=True,coordinates='fixed EGO at this decision start')


def measured_rows(trace, aid, origin):
    return [dict(t_s=round(r['time_s']-trace[0]['time_s'],3),
        xy=ego_xy([r['states'][aid]['x'],r['states'][aid]['y']],origin),
        speed_mps=round(r['states'][aid]['speed'],2),
        heading_deg=round(math.degrees(wrap(r['states'][aid]['yaw']-origin['yaw'])),1))
        for i,r in enumerate(trace) if i%5==0 or i==len(trace)-1]


def restore_trial(toolkit,checkpoint,policy,policy_state,rng):
    world=toolkit.world;ep=toolkit.episode
    world.time=checkpoint['time_s'];world.states=copy.deepcopy(checkpoint['states'])
    world.controller=copy.deepcopy(checkpoint['controller']);world.bg_controller=copy.deepcopy(checkpoint['bg_controller'])
    world.background_fallbacks=copy.deepcopy(checkpoint['background_fallbacks']);world._ego_plan=copy.deepcopy(checkpoint['ego_plan'])
    world.restore_execution_state(checkpoint['simulation']);world._forecast_cache.clear();world._topology_cache.clear()
    ep.monitor.__dict__.update(copy.deepcopy(checkpoint['monitor']))
    ep.trace=ep.trace[:checkpoint['trace_length']];ep.action_history=ep.action_history[:checkpoint['action_length']]
    ep.stopped=False;ep.evaluation=None
    for key,value in policy_state.items():setattr(policy,key,copy.deepcopy(value))
    np.random.set_state(rng[0]);torch.set_rng_state(rng[1])


def candidate_plans(world,policy,decisions):
    plans={}
    for d in decisions:
        aid=policy.actor_names[d['ctrl_agent_index']];meta=d.get('parse_meta') or {}
        xy=meta.get('preview_positions_world',meta.get('trajectory_world_used',[]))
        yaw=meta.get('preview_yaws_world',meta.get('yaws_world_used',[]))
        if not xy:xy=[[s['x'],s['y']] for s in world.legacy_plans[aid]];yaw=[s['yaw'] for s in world.legacy_plans[aid]]
        actor=world.states[aid];poses=[dict(t_s=world.time,pose=copy.deepcopy(actor))]
        poses += [dict(t_s=round(world.time+(i+1)*world.dt,6),pose=dict(actor,x=p[0],y=p[1],yaw=yaw[i])) for i,p in enumerate(xy)]
        plans[aid]=poses
    return plans


def run_post_trial_window(toolkit,policy,dry_run=False,assess=None):
    """Joint candidate transaction: rejected branches never enter final trace."""
    world=toolkit.world;ep=toolkit.episode
    checkpoint=toolkit._checkpoint()
    keys=('_last_llm_output_by_agent','_last_plan_cache_by_agent','_last_failure_feedback_by_agent')
    policy_state={k:copy.deepcopy(getattr(policy,k)) for k in keys}
    rng=(np.random.get_state(),torch.get_rng_state())
    policy.round_origin=copy.deepcopy(world.states['ego']);policy.trial_feedback={};policy.failed_trial=None
    attempts=[]
    max_attempts=world.config.get('posttrial_assessor_max_replans',1)+1
    try:
        for attempt in range(max_attempts):
            start=world.time;count,decisions=world.prepare_legacy(policy)
            if not decisions:raise ValueError('No participant decision for native trial')
            plans=candidate_plans(world,policy,decisions)
            before_states=copy.deepcopy(world.states)
            action=dict(time_s=start,duration_s=count*world.dt,actions=[],plans_world={
                a:dict(created_at_s=start,points=[dict(t_s=p['t_s']-start,position_world_m=[p['pose']['x'],p['pose']['y']]) for p in points])
                for a,points in plans.items()},legacy_decisions=decisions,trajectory_method='post_trial_waypoint_assessor')
            ep.action_history.append(action);event_start=len(ep.monitor.events);rejection=None
            for _ in range(count):
                try:before,after,controls=world.step()
                except ForbiddenRearEnd as exc:rejection=str(exc);break
                contact=ep.monitor.update(before,after,controls,world.time,world.on_road)
                ep.trace.append(dict(time_s=world.time,states=after,controls=controls))
                if contact:break
            trial=dict(start_s=start,end_s=world.time,trace=copy.deepcopy(ep.trace[checkpoint['trace_length']-1:]),
                plans=plans,before_states=before_states,events=copy.deepcopy(ep.monitor.events[event_start:]),
                native_rear_end_rejection=rejection)
            road_evidence=trial_road_evidence(world,trial)
            if road_evidence.get('available'):
                road_evidence['actors']={policy.view.actor_aliases[a]:v for a,v in road_evidence['actors'].items()}
            safety=None
            check_road=world.config.get('posttrial_check_road',True)
            if world.config.get('posttrial_failure_policy')=='validated_fallback':
                from .evaluation_protocol import motion_audit,trial_road_gate
                from .feasible_waypoints import previous_executed_control
                from tbsim.utils.control_projection import DEFAULT_LIMITS
                previous={a:previous_executed_control(ep.trace[:checkpoint['trace_length']],a,start,world.dt)
                          for a in world.anchor_participants}
                safety=dict(motion=motion_audit(trial['trace'],world.anchor_participants,DEFAULT_LIMITS,previous),
                            road=trial_road_gate(road_evidence))
                safety['road']['enforced']=check_road
                safety['passed']=safety['motion']['passed'] and (not check_road or safety['road']['passed']) and not rejection
                trial['execution_safety']=safety
            elif world.config.get('posttrial_failure_policy')=='road_validated_fallback':
                from .evaluation_protocol import trial_road_gate
                if world.config.get('physical_waypoint_tracking'):
                    raise ValueError('Road-only trial gate is for the direct-interpolation ablation')
                safety=dict(road=trial_road_gate(road_evidence),motion=dict(checked=False,passed=None,
                    note='Direct waypoint interpolation: projected-unicycle control bounds are not an execution gate. '
                         'The Assessor still judges measured motion; final review is unchanged.'))
                safety['road']['enforced']=check_road
                safety['passed']=(not check_road or safety['road']['passed']) and not rejection
                trial['execution_safety']=safety
            policy.drawing_trial=trial
            verdicts=[]
            for d in decisions:
                aid=policy.actor_names[d['ctrl_agent_index']]
                payload=dict(control=policy.view.actor_aliases[aid],time_s=start,goal=policy.saved_goal['display_goal'],
                    trial_interval_s=[start,world.time],coordinates='EGO at trial start, x forward/y left meters',
                    actual_execution={policy.view.actor_aliases[a]:measured_rows(trial['trace'],a,policy.round_origin)
                        for a in ['ego']+sorted(world.anchor_participants)},
                    vehicle_dimensions={policy.view.actor_aliases[a]:dict(
                        length_m=before_states[a]['length'],width_m=before_states[a]['width'])
                        for a in ['ego']+sorted(world.anchor_participants)},
                    road_footprint=road_evidence,
                    assess_road_bounds=check_road,
                    relative_samples={policy.view.actor_aliases[a]:v for a,v in trial_relative_samples(trial).items()},
                    actual_body_geometry={policy.view.actor_aliases[a]:native_trial_geometry(trial,a,policy.round_origin)
                        for a in trial['plans']},
                    measured_motion={policy.view.actor_aliases[a]:measured_motion_summary(trial['trace'],a)
                        for a in trial['plans']},
                    actual_events=trial['events'],native_rear_end_rejection=rejection,
                    planned_adv_execution_interval=[dict(t_s=round(p['t_s']-start,3),xy=ego_xy([p['pose']['x'],p['pose']['y']],policy.round_origin))
                        for i,p in enumerate(plans[aid]) if p['t_s']<=world.time+1e-8 and (i%5==0 or i==len(plans[aid])-1)],
                    ego_replan_interval_s=world.config.get('ego_replan_interval_s',.1),ego_model=world.config.get('ego_model','strive'))
                payload['execution_safety']=safety
                user=readable_assessor_input(payload,policy.view.actor_aliases)
                assessor_enabled=world.config.get('posttrial_assessor_enabled',True)
                if not assessor_enabled:
                    image=None;meta=dict(bev_attached=False,status='assessor_disabled_ablation')
                elif safety is not None and not safety['passed']:
                    image=None;meta=dict(bev_attached=False,status='local_execution_safety_failed')
                else:image,meta=policy.trial_image('assessor')
                format_repairs=0
                if safety is not None and not safety['passed']:
                    verdict={'pass':False,'failure_feedback':"Physical test run failed independent physics/road checks:"+json.dumps(safety,ensure_ascii=False),
                             'revision_advice':"Correct the continuous actions according to the listed vehicles, time points, and out-of-bounds/motion values; it is not enough to only fix the JSON or claim reasonableness."}
                    raw=json.dumps(verdict,ensure_ascii=False)
                elif not assessor_enabled:
                    verdict={'pass':True,'reason':'Assessor disabled; local checks only.','failure_feedback':'','revision_advice':''}
                    raw=json.dumps(verdict)
                elif assess:verdict=assess(payload)
                elif dry_run:verdict=dict(pass_=True)
                else:
                    verdict,raw,format_repairs=call_trial_assessor(policy._client,user,image)
                if dry_run and not assess and (safety is None or safety['passed']):
                    verdict={'pass':True,'failure_feedback':'','revision_advice':''}
                passed=bool(verdict.get('pass')) and not rejection
                assessment=dict(status='post_trial_passed' if passed else 'post_trial_failed',passed=passed,
                    assessor_enabled=assessor_enabled,
                    parsed_response=verdict,response_text=json.dumps(verdict,ensure_ascii=False) if (dry_run or assess) else raw,
                    system_prompt=POST_ASSESSOR_SYSTEM,user_prompt=user,bev_meta=meta,
                    measured_evidence=payload,
                    format_repair_requests=format_repairs)
                d['assessor_meta']=assessment
                d['status']='post_trial_passed' if passed else 'post_trial_failed'
                verdicts.append(passed)
                policy.trial_feedback[aid]=dict(assessor_failure_feedback=str(verdict.get('failure_feedback','')),
                    assessor_revision_advice=str(verdict.get('revision_advice','')))
            passed=all(verdicts)
            trial['assessments']=[copy.deepcopy(d['assessor_meta']) for d in decisions]
            trial['assessor_enabled']=world.config.get('posttrial_assessor_enabled',True)
            final_attempt=attempt==max_attempts-1
            safe_final=final_attempt and (safety is None or safety['passed']) and world.config.get('posttrial_allow_rejected_fallback',True)
            trial['accepted']=passed;trial['committed']=passed or safe_final
            trial['attempt_index']=attempt;trial['decisions']=copy.deepcopy(decisions)
            name='candidate_trial_%03d_%02d.json'%(round(start/world.dt),attempt+1)
            ep.write(name,trial);attempts.append(dict(artifact=name,accepted=passed))
            if final_attempt and not passed and not safe_final:
                restore_trial(toolkit,checkpoint,policy,policy_state,rng)
                ep.stopped=True
                for d in decisions:d['status']='execution_safety_failed_budget_exhausted'
                return 0,decisions,'execution_safety_failed_budget_exhausted'
            if passed or safe_final:
                action['duration_s']=world.time-start;action['trial_attempts']=attempts
                if not passed:
                    for d in decisions:d['status']='post_trial_failed_final_used_soft'
                action['soft_fallback']=not passed
                if safety is not None:
                    action['execution_safety']=safety
                    action['fallback_policy']=(world.config['posttrial_failure_policy'] if not passed else
                        'assessed_and_validated' if check_road else 'assessed_road_check_disabled')
                policy.remaining_plans=copy.deepcopy(plans);policy.failed_trial=None
                ep.stopped=bool(rejection or ep.monitor.report()['metrics']['target_collision'] or world.time>=world.config['max_simulation_s']-1e-8)
                if world.time<=start and not rejection:raise ValueError('Native trial made no progress')
                return count,decisions,rejection
            policy.failed_trial=copy.deepcopy(trial)
            restore_trial(toolkit,checkpoint,policy,policy_state,rng)
        raise AssertionError('Unreachable candidate loop')
    except Exception:
        restore_trial(toolkit,checkpoint,policy,policy_state,rng)
        raise


def draw_execution_layers(draw,screen,layers,alias,font,line,size):
    """Shared world frame: initial ghosts, accepted history, trial and proposal."""
    ids=layers['ids']
    def label(pose,text,color,left=False,dy=12):
        x,y=screen([pose['x'],pose['y']]);width=draw.textbbox((0,0),text,font=font)[2]
        tx=max(16,x-width-12) if left else min(size-width-16,x+20)
        draw.text((tx,y+dy),text,font=font,fill=color,stroke_width=2,stroke_fill='white')
    def body(pose,color,dashed=False,fill=None):
        pts=[screen(c) for c in corners(pose)]
        if fill:draw.polygon(pts,fill=fill)
        for a,b in zip(pts,pts[1:]+pts[:1]):
            if not dashed:draw.line([a,b],fill=color,width=2);continue
            n=max(1,math.ceil(math.hypot(b[0]-a[0],b[1]-a[1])/7))
            for k in range(0,n,2):draw.line([(a[0]+(b[0]-a[0])*j/n,a[1]+(b[1]-a[1])*j/n) for j in (k,min(k+1,n))],fill=color,width=2)
        x,y=screen([pose['x'],pose['y']]);draw.ellipse((x-4,y-4,x+4,y+4),fill='white',outline=color,width=2)
        tip=[pose['x']+pose['length']/2*math.cos(pose['yaw']),pose['y']+pose['length']/2*math.sin(pose['yaw'])]
        line([[pose['x'],pose['y']],tip],color,2)
    for aid,pose in layers['initial'].items():
        body(pose,'#64748b',True);label(pose,alias(aid)+' START t=0.0s','#475569',aid=='ego',38)
    for aid in ids:
        rows=[r for r in layers['history'] if aid in r['states']]
        color=('#93a9c5' if aid=='ego' else '#c9aa90') if layers['trial'] else ('#1565c0' if aid=='ego' else '#d2691e')
        line([[r['states'][aid]['x'],r['states'][aid]['y']] for r in rows],color,3 if layers['trial'] else 4)
    trial=layers['trial']
    # Measured trial views never include unexecuted proposals, including feedback.
    for aid,poses in (layers['plans'] if layers['phase']=='planner' and not trial else {}).items():
        poses=[p for p in poses if p['t_s']>=layers['origin_s']-1e-8]
        line([[p['pose']['x'],p['pose']['y']] for p in poses],'#9333ea',3)
        for i,p in enumerate(poses):
            if i%5 and i!=len(poses)-1:continue
            body(p['pose'],'#a76bd1');label(p['pose'],alias(aid)+f" PLAN t={p['t_s']:.1f}s",'#6b21a8',dy=-18)
    if trial:
        for aid in ids:
            color='#1565c0' if aid=='ego' else '#dc2626' if layers['phase']=='planner' else '#d2691e'
            rows=trial['trace'];line([[r['states'][aid]['x'],r['states'][aid]['y']] for r in rows],color,6)
            for i,r in enumerate(rows):
                if i==0 or i==len(rows)-1:continue
                if i%5:continue
                pose=r['states'][aid]
                x,y=screen([pose['x'],pose['y']]);draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
                if i%10==0:body(pose,color)
            pose=trial['before_states'][aid];body(pose,'#075985',True)
            label(pose,alias(aid)+f" BEFORE t={trial['start_s']:.1f}s",'#075985',aid=='ego',dy=42)
            last=rows[-1];pose=last['states'][aid]
            fill=('#dbeafe' if aid=='ego' else '#fee2e2') if layers['phase']=='planner' else None
            body(pose,color,fill=fill)
            label(pose,alias(aid)+f" TRIAL END t={last['time_s']:.1f}s (+{last['time_s']-trial['start_s']:.1f}s)",color,aid=='ego',dy=-26)
    elif layers['ego_forecast']:
        forecast=layers['ego_forecast'];points=[p for p in forecast['points'] if p['t_s']>=layers['now_s']-1e-8]
        line([p['world_xy_m'] for p in points],'#0891b2',4)
        ego=layers['history'][-1]['states']['ego']
        for i,p in enumerate(points):
            if i%5 and i!=len(points)-1:continue
            pose=dict(ego,x=p['world_xy_m'][0],y=p['world_xy_m'][1],yaw=p.get('yaw_rad',ego['yaw']))
            body(pose,'#0891b2')
            label(pose,f"EGO FORECAST t={p['t_s']:.1f}s",'#0891b2',True)
    for y,text,color in [(87,'GRAY START: scene initial | filled: restored planning-start poses' if trial and layers['phase']=='planner' else
            'GRAY START: scene initial poses | solid vehicles: current measured poses','#475569'),
        (160,'BLUE: EGO | ORANGE: controlled | GRAY: background | solid paths: measured','#18364d'),
        (187,f"ASSESSOR: ACTUAL ONLY {trial['start_s']:.1f}-{trial['end_s']:.1f}s | BEFORE: trial start | no future plans" if layers['phase']=='assessor' and trial else
         f"PLANNER: rejected ACTUAL {trial['start_s']:.1f}-{trial['end_s']:.1f}s | pale END: trial end | no future plans" if trial else 'PLANNER: accepted history + remaining plan | CYAN: EGO forecast','#18364d')]:
        draw.rectangle((15,y,1100,y+25),fill='white');draw.text((24,y+2),text,font=font,fill=color)
