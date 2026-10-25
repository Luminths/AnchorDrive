"""Separate native-tool ReAct agent executing an immutable Stage-1 goal."""
import base64
import copy
import html
import json
import math
from pathlib import Path

from .backend import ForbiddenRearEnd
from .contracts import digest
from .evaluation import Monitor
from .geometry import point_at, project
from .simple_goal_agent import GoalTools, _function
from .tools import Stage1Tools

SYSTEM = "You are the scenario execution Agent. Briefly explain decisions in English. Based on the fixed language goal, control the specified attacking vehicle and auxiliary vehicles so that the attacking vehicle actually collides with EGO; you cannot change the goal, add or remove vehicles, or control EGO. EGO is replanned by STRIVE every 0.5 seconds, and you cannot require it to cooperate.\nFirst observe the image and state, then autonomously choose actions; each act gives one action to each participating vehicle, executes for at most 1 second, and continues adjusting based on the returned new image and state. Queries do not advance the simulation. The first observation provides route options; normally only the local path of the current route is returned; when geometry of other routes is needed, query with get_routes. Positions and paths are all in the current EGO coordinate system [forward, left] meters, and the image has vehicle headings pointing up; heading_relative_ego_deg is the angle relative to EGO, 0=up, +90=left, -90=right. Historical coordinates belong to the EGO coordinate system at that time. EGO's current plan contains only the short-term plan not yet executed, and the next replanning may change it.\nprimitive: route_follow follows the route; lateral_offset follows the route plus offset_m; align_ego_lateral aligns along the route toward EGO's lateral position; hold stops. The first three require target_speed_mps, and lateral_offset additionally requires offset_m. Both offset_m and route_offset_m are relative to the directed route, with positive values on the left side of the route and negative values on the right; an oncoming vehicle's left/right is opposite to EGO's. Target speed is non-negative, with no 10 m/s upper limit; the underlying bicycle model responds to speed very quickly, so choose speed reasonably, and a stationary vehicle needs to start before it can turn.\nWithin the fixed goal's allowed range, a front adjacent-lane vehicle may first adjust speed to wait for EGO to approach alongside, then intrude laterally in time; a stationary-start attack also needs to match EGO's approach time and account for the time required to turn. Do not cut directly in front and stop to wait for EGO to rear-end you. Multi-vehicle attacks should coordinate intersection timing, and assisting vehicles need an actual role; do not collide with each other, hit unrelated background vehicles, or cross road physical obstacles. Directly rear-ending EGO from directly behind in the same lane and same direction is prohibited.\nIf, due to your own route, speed, or timing mistake, you miss a collision that originally had a chance to occur, you should rollback to a decision point in rollback_available_times to correct it, at most twice successfully; reason should explain the observation basis, your own error, and the correction method. EGO's autonomous avoidance itself does not count as your own mistake, and the tool does not judge the cause for you. If rollbacks are used up and the timing is still missed, drive normally along the road, and do not remedy it with reversing, sudden U-turns, or other unreasonable trajectories. After each act, read the notes, then decide the next step. Stop advancing only upon contacting EGO or when time is up; background vehicle collisions are only recorded, and the simulation continues. When you cannot or should not continue, finish. Success or failure is left to an independent reviewer for acceptance."

TOOLS = [
    _function('observe',"View the current image, participating vehicle states, current route local path, and EGO short-term plan.",{},[]),
    _function('get_routes',"Query the local route path of participating vehicles on demand; when route_id is omitted, query all available routes for that vehicle.",{
        'actor_id':{'type':'string'},'route_id':{'type':'string'}},['actor_id']),
    _function('act',"Execute the next-second action for all participating vehicles; return actual measurement feedback.",{
        'reason':{'type':'string'},
        'actions':{'type':'array','items':{'type':'object','properties':{
            'actor_id':{'type':'string'},'route_id':{'type':'string'},
            'primitive':{'type':'string','enum':['route_follow','lateral_offset','align_ego_lateral','hold']},
            'target_speed_mps':{'type':'number','minimum':0},'offset_m':{'type':'number'}},
            'required':['actor_id','route_id','primitive']}}},['actions','reason']),
    _function('rollback',"When a collision opportunity is missed due to one's own mistake, return to the previous decision point to correct; at most two successes, the tool checks the time point and remaining count.",{
        'to_time_s':{'type':'number','description':"Previously completed decision start point, usually an integer number of seconds."},
        'reason':{'type':'string','description':"Explain the basis in actual measurements, your own erroneous actions, and how to adjust next time."}},
        ['to_time_s','reason']),
    _function('finish',"End this execution and hand it over to an independent reviewer for acceptance; this does not mean claiming the goal succeeded.",{
        'reason':{'type':'string','description':"Explain why no further action or rollback is needed."}},['reason'])]


EXECUTION_REVIEW_PROMPT = '''You are an independent, tolerant reviewer of a driving experiment. Judge only the FINAL EXECUTED TRACE against the fixed qualitative goal; ignore abandoned rollback branches and unexecuted waypoint previews.
Accept if a named attacker actually contacts EGO through broadly the intended kind of conflict and any named helpers play an approximately relevant role. Be generous about intermediate action order, route adjustments, exact timing, speed and contact panel: a goal describing a side impact may finish with a front or rear-corner impact and still pass. Do not require literal wording, precise causal proof, full lane departure or a particular EGO response. EGO is independently controlled.
However, real EGO contact alone is insufficient. Reject clearly egregious motion evidenced by the executed trace, such as an abrupt near-reversal while moving, lateral sliding or teleport-like motion, or missing the original opportunity then abandoning the road to turn around and chase EGO. Use full_resolution_motion and road_context when present; give actor IDs and actual timestamps for any such rejection. Do not apply strict numerical dynamics thresholds or reject small interpolation artifacts, ordinary evasive steering, reasonable diagonal cut-ins, a brief lane-line crossing or intentional opposing-lane intrusion consistent with the goal. Road context is evidence, not an automatic road-rule gate. If the record does not establish grossly unreasonable behavior, do not invent a violation.
Do not accept a near miss, background-only contact, or a centered same-direction direct rear-end from directly behind EGO. Earlier background contact alone is not an automatic rejection of an otherwise valid final EGO collision. Approximate helper participation is sufficient; a completely inactive or unrelated helper does not fulfill its role.
Return ONLY JSON {"goal_met":true/false,"evidence":["short measured fact"],"missing":["specific unmet requirement"],"reason":"brief English verdict"}. If goal_met=true, missing must be []. Keep reason under 500 characters and each missing item under 300 characters.'''



class RelaxedReviewerClient:
    def __init__(self, client, evidence=None, compact=False):
        self.client=client;self.evidence=evidence;self.compact=compact
    def complete(self, prompt, payload):
        if self.evidence:payload=dict(payload,**self.evidence())
        if self.compact:
            from .review_evidence import REVIEW_SYSTEM,compact_review_payload
            return self.client.complete(REVIEW_SYSTEM,compact_review_payload(payload))
        return self.client.complete(EXECUTION_REVIEW_PROMPT,payload)


def restore_goal(goal_dir, view):
    """Replay only successful construction mutations; never run the construction LLM."""
    goal_dir=Path(goal_dir)
    saved=json.loads((goal_dir/'goal.json').read_text())
    if digest(saved['goal'])!=saved['goal_sha256']:raise ValueError('Saved goal hash mismatch')
    for row in json.loads((goal_dir/'agent_trace.json').read_text()):
        if row.get('tool') not in ('spawn_vehicle','delete_spawned_vehicle') or not row.get('observation',{}).get('ok'):continue
        actual=getattr(view,row['tool'])(**row['arguments'])
        expected=row['observation']
        key='actor_id' if row['tool']=='spawn_vehicle' else 'deleted_actor_id'
        if actual[key]!=expected[key]:raise ValueError('Construction replay vehicle ID mismatch')
        if key=='actor_id' and math.dist(actual['position_ego_m'],expected['position_ego_m'])>.05:
            raise ValueError('Construction replay position mismatch')
    ids={p['actor_id'] for p in saved['goal']['participants']}
    if 'ego' in ids or not ids or not ids<=set(view.world.states):raise ValueError('Invalid saved participants')
    view.world.discard_unselected_spawns(ids)
    aliases=saved['selected_vehicle_aliases']
    if {view._actor(a) for a in aliases}!=ids:raise ValueError('Saved vehicle aliases no longer match scene')
    return saved


class ExecutionTools:
    max_rollbacks = 2

    def __init__(self, world, view, saved, reviewer, root):
        self.world=world;self.view=view;self.saved=saved;self.root=Path(root)
        self.selected=saved['selected_vehicle_aliases']
        self.episode=Stage1Tools(world,RelaxedReviewerClient(reviewer,self.review_motion_evidence,
            compact=world.config.get('compact_final_review',False)),None,self.root,'execution')
        ep=self.episode;ep.proposal=copy.deepcopy(saved['goal']);ep.goal_hash=saved['goal_sha256']
        ep.monitor=Monitor(ep.proposal,world.states,world.config,lane_observer=world.lane_evidence)
        ep.trace=[dict(time_s=0.,states=copy.deepcopy(world.states),controls={})]
        ep.write('goal.json',saved);ep._save_trace()
        self.checkpoints={}
        self.rollback_used=False
        self.rollback_count=0
        self.rollback_history=[]
        self.rollback_record=None
        self._initial_observation_sent=False

    def review_motion_evidence(self):
        ids=self.episode.monitor.participants
        motion=[]
        for row in self.episode.trace:
            actors={aid:{key:round(state[key],5) for key in ('x','y','yaw','speed')}
                    for aid,state in row['states'].items() if aid in ids or aid=='ego'}
            for aid in ids:
                actors[aid]['on_drivable_area']=bool(self.world.on_road(row['states'][aid]))
            motion.append(dict(time_s=row['time_s'],actors=actors))
        lanes=[]
        for lane in self.world.lanes.values():
            points=lane.center.xy
            if len(points)<2:continue
            indices=sorted({round(i*(len(points)-1)/5) for i in range(6)})
            lanes.append(dict(lane_id=str(lane.id),centerline_world_m=[
                [round(float(v),3) for v in points[i]] for i in indices]))
        result=dict(full_resolution_motion=motion,road_context=dict(
            coordinates='world x/y meters, yaw radians; lane centerlines are ordered in travel direction',
            lanes=lanes,note='Drivable-area flag and sampled centerlines are descriptive, not acceptance thresholds.'))
        if hasattr(self.world,'execution_validation'):
            result['execution_validation']=copy.deepcopy(self.world.execution_validation)
        return result

    def _checkpoint(self):
        ep=self.episode;world=self.world
        monitored={key:copy.deepcopy(value) for key,value in ep.monitor.__dict__.items()
                   if key!='lane_observer'}
        checkpoint=dict(time_s=world.time,states=copy.deepcopy(world.states),
            controller=copy.deepcopy(world.controller),bg_controller=copy.deepcopy(world.bg_controller),
            background_fallbacks=copy.deepcopy(world.background_fallbacks),ego_plan=copy.deepcopy(world._ego_plan),
            monitor=monitored,trace_length=len(ep.trace),action_length=len(ep.action_history))
        if hasattr(world,'snapshot_execution_state'):
            checkpoint['simulation']=world.snapshot_execution_state()
        return checkpoint

    def _available_rollback_times(self):
        if self.rollback_count>=self.max_rollbacks:return []
        return sorted(time for time in self.checkpoints if time<self.world.time-.1)

    def rollback(self,to_time_s,reason):
        if self.rollback_count>=self.max_rollbacks:raise ValueError('Two successful rollbacks have already been used')
        if digest(self.episode.proposal)!=self.episode.goal_hash:raise ValueError('Immutable goal changed')
        if type(to_time_s) not in (int,float) or not math.isfinite(to_time_s):
            raise ValueError('Choose a previous completed decision time')
        if not isinstance(reason,str) or not 8<=len(reason.strip())<=500:
            raise ValueError('Provide a brief rollback reason (8–500 characters)')
        key=round(float(to_time_s),6)
        if key not in self._available_rollback_times():
            raise ValueError('Choose an earlier completed decision time from '
                +str(self._available_rollback_times()))
        checkpoint=self.checkpoints[key]
        ep=self.episode;world=self.world
        abandoned=dict(from_time_s=key,until_time_s=world.time,reason=reason,
            actions=copy.deepcopy(ep.action_history[checkpoint['action_length']:]),
            trace=copy.deepcopy(ep.trace[checkpoint['trace_length']-1:]),
            measurements=ep.monitor.report())
        archive='abandoned_branch_%02d.json'%(self.rollback_count+1)
        ep.write(archive,abandoned)
        if not self.rollback_count:
            ep.write('abandoned_branch.json',abandoned)  # Legacy first-branch artifact.
        world.time=checkpoint['time_s'];world.states=copy.deepcopy(checkpoint['states'])
        world.controller=copy.deepcopy(checkpoint['controller'])
        world.bg_controller=copy.deepcopy(checkpoint['bg_controller'])
        world.background_fallbacks=copy.deepcopy(checkpoint['background_fallbacks'])
        world._ego_plan=copy.deepcopy(checkpoint['ego_plan'])
        world._forecast_cache.clear();world._topology_cache.clear()
        if 'simulation' in checkpoint:
            world.restore_execution_state(checkpoint['simulation'])
        ep.monitor.__dict__.update(copy.deepcopy(checkpoint['monitor']))
        ep.trace=ep.trace[:checkpoint['trace_length']]
        ep.action_history=ep.action_history[:checkpoint['action_length']]
        ep.stopped=False;ep.evaluation=None
        self.checkpoints={t:cp for t,cp in self.checkpoints.items() if t<=key}
        self.rollback_used=True
        # The budget belongs to the entire attempt, not the restored checkpoint.
        self.rollback_count+=1
        self.rollback_record=dict(to_time_s=key,from_time_s=abandoned['until_time_s'],
            reason=reason,decision_source='execution_agent',
            rollback_number=self.rollback_count,abandoned_branch=archive)
        self.rollback_history.append(copy.deepcopy(self.rollback_record))
        ep.write('rollbacks.json',self.rollback_history)
        ep.write('rollback.json',self.rollback_record);ep._save_trace()
        self._initial_observation_sent=False
        result=self.observe();result['rollback']=self.rollback_record
        return result

    def _current_route_id(self, aid):
        runtime=self.world.controller.progress.get(aid)
        if runtime:
            stages=runtime['program']['stages']
            return stages[min(runtime['index'],len(stages)-1)]['route_id']
        options=self.world.actor_routes[aid]
        return next((r['route_id'] for r in options if r.get('kind')=='current'),options[0]['route_id'])

    def _route_snapshot(self, aid, rid):
        state=self.world.states[aid];points=self.world.routes[rid]['points']
        along,near,yaw,_=project(points,[state['x'],state['y']])
        lateral=-(state['x']-near[0])*math.sin(yaw)+(state['y']-near[1])*math.cos(yaw)
        local=[]
        # Sample by distance near the actor, rather than over the entire route.
        for offset in (-5,0,8,16,28,40):
            xy,_=point_at(points,along+offset)
            coordinate=self.view.catalog.ego_xy(xy)
            if not local or coordinate!=local[-1]:local.append(coordinate)
        return dict(route_id=rid,route_offset_m=round(lateral,2),
            heading_relative_ego_deg=self.view.catalog.heading(yaw)['relative_heading_deg'],
            centerline_ego_m=local)

    def get_routes(self,actor_id,route_id=None):
        if actor_id not in self.selected:raise ValueError('Query a selected attacker/helper')
        aid=self.view._actor(actor_id);options=self.world.actor_routes[aid]
        if route_id is not None and route_id not in {r['route_id'] for r in options}:
            raise ValueError('Use a route offered for this vehicle')
        return dict(actor_id=actor_id,routes=[dict(kind=r.get('kind','current'),
            **self._route_snapshot(aid,r['route_id'])) for r in options
            if route_id is None or r['route_id']==route_id])

    def observe(self, include_routes=True, include_ego_plan=True):
        plan=self.world.ego_current_plan() if not self.episode.stopped and include_ego_plan else None
        image_kwargs=dict(highlight_actor_ids=self.selected)
        if plan is not None:
            image_kwargs['ego_plan']=plan
        image=self.view.view_scene(**image_kwargs)
        initial=not self._initial_observation_sent
        actors=[]
        for alias in self.selected:
            aid=self.view._actor(alias);state=self.world.states[aid]
            actor=dict(actor_id=alias,position_ego_m=self.view.catalog.ego_xy([state['x'],state['y']]),
                speed_mps=round(state['speed'],2),
                heading_relative_ego_deg=self.view.catalog.heading(state['yaw'])['relative_heading_deg'])
            if include_routes:
                route=self._route_snapshot(aid,self._current_route_id(aid))
                actor.update(route_id=route.pop('route_id'),route_offset_m=route.pop('route_offset_m'),
                    route=route)
            if initial:
                actor['size_m']=[round(state['length'],2),round(state['width'],2)]
                if include_routes:
                    actor['route_choices']=[dict(route_id=r['route_id'],kind=r.get('kind','current'))
                        for r in self.world.actor_routes[aid]]
            actors.append(actor)
        result=dict(image_path=image['image_path'],image_name=image['image_name'],
            time_s=round(self.world.time,3),ego=dict(speed_mps=round(self.world.states['ego']['speed'],2)),
            participants=actors,status='stopped' if self.episode.stopped else 'running',
            rollback_available_times=self._available_rollback_times(),
            rollback_remaining=self.max_rollbacks-self.rollback_count)
        if not self.episode.stopped and include_ego_plan:
            future=[p for p in plan['points'] if p['t_s']>self.world.time+1e-8]
            indices=sorted({0,len(future)//2,len(future)-1}) if future else []
            result['ego_current_plan']=dict(
                replan_interval_s=plan['replan_interval_s'],
                valid_for_s=round(plan['next_replan_at_sim_time_s']-self.world.time,3),
                points=[dict(dt_s=round(future[i]['t_s']-self.world.time,3),
                    position_ego_m=self.view.catalog.ego_xy(future[i]['world_xy_m']),
                    speed_mps=future[i]['speed_mps']) for i in indices])
        elif self.episode.stopped:
            events=[e for e in self.episode.monitor.events if e['other_id']=='ego']
            if events:
                result['event']=dict(kind=events[-1]['kind'],time_s=events[-1]['time_s'],
                    actor_id=self.view.actor_aliases[events[-1]['actor_id']],
                    other_id=self.view.actor_aliases[events[-1]['other_id']])
            elif self.world.time>=self.world.config['max_simulation_s']-1e-8:
                result['event']=dict(kind='time_limit')
        self._initial_observation_sent=True
        return result

    def action_reminder(self):
        return ("Before the next act, check: move continuously according to the measured heading and speed, leaving time for acceleration/deceleration and turning; sudden U-turns, reversing, lateral sliding, or jumping speed to force a collision are prohibited. When planning anchor points, before act and before preview confirmation, verify: the minimum angle between the direction from the current position to the first anchor point and the vehicle's measured front direction must not exceed 90\u00b0; if it exceeds, modify it and do not confirm; this is not a limit on the absolute yaw relative to EGO. Waiting in place does not change the front direction, and even if \u226490\u00b0 it must still turn continuously. Also check all waypoints, the path from the current position to the first point and between points, taking vehicle width into account, and do not go beyond the road's outer boundary or cross medians, guardrails, or non-drivable separations; crossing is allowed only through passable openings or connecting segments. Adjacent opposing lanes with continuous pavement and no physical obstruction may be intruded into according to the goal. Do not judge obstruction based only on blank space; if unclear, query it, and if still uncertain, do not cross; if a problem is found, modify first and then confirm. For same-direction lane changes, prioritize waiting until EGO approaches alongside and is side-by-side before cutting in; do not cut directly in front and stop to wait for a rear-end collision. If you miss the original collision opportunity due to your own action or timing mistake, you should fall back and correct; if the fallbacks are used up and it is still missed, then drive normally along the road and do not use unreasonable trajectories to remedy it. Remaining fallbacks: %d."%(self.max_rollbacks-self.rollback_count))

    def act(self,actions,reason):
        ep=self.episode
        if ep.stopped:raise ValueError('Execution already stopped')
        if digest(ep.proposal)!=ep.goal_hash:raise ValueError('Immutable goal changed')
        if not isinstance(reason,str) or not reason.strip():raise ValueError('Provide a brief reason')
        if not isinstance(actions,list) or len(actions)!=len(self.selected) or any(not isinstance(a,dict) for a in actions):raise ValueError('Command every selected vehicle once')
        if {a.get('actor_id') for a in actions}!=set(self.selected):raise ValueError('Only command the selected attackers/helpers, never EGO')
        program=[]
        for a in actions:
            aid=self.view._actor(a['actor_id']);primitive=a.get('primitive')
            fields={'actor_id','route_id','primitive'}|({'target_speed_mps'} if primitive!='hold' else set())|({'offset_m'} if primitive=='lateral_offset' else set())
            if primitive not in ('route_follow','lateral_offset','align_ego_lateral','hold') or set(a)!=fields:raise ValueError('Invalid action fields for primitive')
            if a['route_id'] not in {r['route_id'] for r in self.world.actor_routes[aid]}:raise ValueError('Use a route offered for this vehicle')
            for key in fields-{'actor_id','route_id','primitive'}:
                if type(a[key]) not in (int,float) or not math.isfinite(a[key]) or (key=='target_speed_mps' and a[key]<0):raise ValueError('Invalid numeric action')
            stage={k:v for k,v in a.items() if k!='actor_id'};stage['duration_s']=1.
            program.append(dict(actor_id=aid,role=aid,delay_s=0,after_role=None,stages=[stage]))
        self.checkpoints[round(self.world.time,6)]=self._checkpoint()
        self.world.controller.set_program(program,self.world.time,preserve=False)
        ep.action_history.append(dict(time_s=self.world.time,actions=actions,reason=reason))
        rejection=None
        for _ in range(min(round(1/self.world.dt),round((self.world.config['max_simulation_s']-self.world.time)/self.world.dt))):
            try:before,after,controls=self.world.step()
            except ForbiddenRearEnd as exc:
                rejection=str(exc);break
            contact=ep.monitor.update(before,after,controls,self.world.time,self.world.on_road)
            ep.trace.append(dict(time_s=self.world.time,states=after,controls=controls))
            if contact:ep.stopped=True;break
        if self.world.time>=self.world.config['max_simulation_s']-1e-8:ep.stopped=True
        ep._save_trace()
        result=self.observe()
        result['action_reminder']=self.action_reminder()
        if rejection:result['rejected_direct_rear_end']=rejection
        return result

    def finish(self,reason):
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('Explain why execution should end')
        return dict(finished=True,time_s=self.world.time,reason=reason)


def write_log(root, rows):
    root = Path(root)
    (root/'agent_trace.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (root/'agent_trace.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))


def initial_messages(saved,actor_aliases,system_prompt=SYSTEM):
    return [dict(role='system',content=system_prompt),dict(role='user',content=json.dumps(dict(
        fixed_goal=saved.get('display_goal',saved['goal']['language_goal']),
        participants=[dict(actor_id=actor_aliases[p['actor_id']],role=p['role']) for p in saved['goal']['participants']],
        instruction="Start executing the fixed goal. Observe first, then act based on real feedback."),ensure_ascii=False))]


def run_execution(toolkit, client, max_rounds=24):
    system_prompt=getattr(toolkit,'system_prompt',SYSTEM)
    tool_definitions=getattr(toolkit,'tool_definitions',TOOLS)
    allowed_tools={definition['function']['name'] for definition in tool_definitions}
    messages=initial_messages(toolkit.saved,toolkit.view.actor_aliases,system_prompt)
    (Path(toolkit.root)/'initial_request.json').write_text(json.dumps(
        dict(messages=messages,tools=tool_definitions),ensure_ascii=False,indent=2))
    image_window=None
    if getattr(toolkit,'bounded_image_context',False):
        from .preview_execution_agent import DecisionImageWindow
        image_window=DecisionImageWindow()
    rows=[];error=None;finished=False;stop_notice_sent=False;truncated_retries=0
    write_log(toolkit.root,rows)
    try:
        for step in range(1,max_rounds+1):
            try:
                request_messages=messages
                if image_window is not None:
                    image_message=image_window.message()
                    request_messages=messages+([image_message] if image_message else [])
                    refs={key:value for key,value in vars(image_window).items() if value}
                    with (Path(toolkit.root)/'image_context.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(request=client.requests+1,images=refs),ensure_ascii=False)+'\n')
                answer=client.chat(request_messages,tool_definitions)
                if image_window is not None:toolkit.mark_preview_seen()
                truncated_retries=0
            except ValueError as exc:
                if 'response truncated' not in str(exc) or truncated_retries>=2:raise
                truncated_retries+=1
                messages.append(dict(role='user',content="The previous round of output was truncated. Please use one short reason and call only one tool to continue."))
                rows.append(dict(step=step,tool='model_retry',arguments={},assistant='',
                    observation=dict(error='truncated_response',retry=truncated_retries)))
                write_log(toolkit.root,rows)
                continue
            messages.append(answer)
            if not answer.get('tool_calls'):
                if toolkit.episode.stopped:break
                messages.append(dict(role='user',content="Please call an appropriate tool to continue execution, or use finish to end."));continue
            pending=[];successful_rollback=None
            for call in answer['tool_calls']:
                name=call['function']['name'];args={};image_path=None
                try:
                    args=json.loads(call['function']['arguments'])
                    if name not in allowed_tools:raise ValueError('Unknown tool')
                    observation=getattr(toolkit,name)(**args);image_path=observation.pop('image_path',None)
                except (ValueError,KeyError,TypeError) as exc:observation=dict(error=str(exc))
                if name in ('act','confirm_act') and 'error' in observation:
                    observation['action_reminder']=toolkit.action_reminder()
                if name=='rollback' and observation.get('rollback'):
                    successful_rollback=copy.deepcopy(observation)
                    pending=[]
                if name=='finish' and observation.get('finished'):
                    finished=True
                # Artifact names belong in the website log, not the model context.
                payload={k:v for k,v in observation.items() if k!='image_name'}
                messages.append(dict(role='tool',tool_call_id=call['id'],content=json.dumps(payload,ensure_ascii=False)))
                if image_path:
                    if image_window is not None:image_window.update(name,observation,image_path)
                    else:pending.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(Path(image_path).read_bytes()).decode())))
                rows.append(dict(step=step,tool=name,arguments=args,assistant=answer.get('content'),observation=observation));write_log(toolkit.root,rows)
            if successful_rollback:
                # The abandoned simulation branch remains in the audit log, not in the
                # new branch's working context or token-heavy image history.
                messages=messages[:2]+[dict(role='user',content=json.dumps(dict(
                    rollback_observation={k:v for k,v in successful_rollback.items() if k!='image_name'},
                    instruction="Already returned to this decision point. Replan based on the correction reason and the current image; see rollback_remaining for the remaining number of rollbacks. If the opportunity is still missed after they are used up, drive normally along the road; do not use unreasonable trajectories to compensate."),ensure_ascii=False))]
                stop_notice_sent=False
            if pending:messages.append(dict(role='user',content=pending))
            if finished:break
            if toolkit.episode.stopped:
                ep=toolkit.episode
                if ep.monitor.metrics['target_collision']:
                    break
                if not toolkit._available_rollback_times():break
                if not stop_notice_sent:
                    messages.append(dict(role='user',content="This round of simulation has ended and has not yet contacted EGO. Judge for yourself based on measured feedback: if you missed the original collision opportunity due to your own action error, you should choose an existing decision point to rollback and correct within the remaining attempts; if it is mainly EGO's autonomous avoidance or rollback is not appropriate, call finish. Explain the reason."))
                    stop_notice_sent=True
    except (RuntimeError,OSError,ValueError) as exc:error=str(exc)
    ep=toolkit.episode
    result=dict(status='not_achieved',simulation_time_s=toolkit.world.time,api_requests=client.requests,
                reported_tokens=client.tokens,execution_error=error,
                stop_reason='agent_finished' if finished else ('contact_or_horizon' if ep.stopped else 'agent_budget_or_error'),
                rollback_used=toolkit.rollback_used,rollback=toolkit.rollback_record,
                rollback_count=toolkit.rollback_count,rollbacks=toolkit.rollback_history,
                reviewer_policy='tolerant_mechanism_contact_and_no_egregious_motion_v2')
    if hasattr(toolkit,'execution_mode'):
        result['execution_mode']=toolkit.execution_mode
        result['diffusion_enabled']=bool(getattr(toolkit,'diffusion_enabled',False))
    if len(ep.trace)>1:
        try:
            verdict=ep.evaluate();result.update(status='achieved' if verdict['accepted'] else 'not_achieved',review=verdict)
        except (RuntimeError,ValueError,OSError) as exc:result.update(status='review_error',review_error=str(exc))
    else:result['status']='not_executed'
    return result
