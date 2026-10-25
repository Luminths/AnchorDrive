"""Tools for one-goal, closed-loop Stage-1 scenario construction."""
import copy
import json
import math
import re
from pathlib import Path
from .contracts import digest,exact,number,text,validate_goal,validate_reactive_actions,requires_body_contact
from .geometry import clearance,overlap,project,relative,wrap
from .evaluation import Monitor
from .backend import ForbiddenRearEnd
from .topology import plausible_contact_routes


REVIEW_PROMPT="You are the sole independent reviewer of a Stage-1 driving scenario execution.\nDecide whether the FIXED English language goal was achieved in its main observable mechanism.\nJudge the maneuver and interaction at the level of a traffic scenario, not an exact impact pixel.\nUse only executed states, applied attacker controls, geometric measurements and detected contacts. The\nconstruction agent's intention and rationale are not evidence. The universal prohibition is\nNARROW: an attacker nearly aligned with EGO's heading and centerline must not strike EGO\ndirectly from behind. A diagonal cut-in, lateral body contact, or oblique rear-corner contact\nfrom an adjacent lane is not prohibited merely because the attacker's center is behind EGO.\nIf a directly following, same-heading attacker strikes EGO's rear,\nreturn goal_met=false and identify that forbidden event in missing. Judge this from\nactor_measurements.initial_relative_to_ego, the executed poses and contact\ngeometry; do not equate a genuine side or crossing impact with a rear-end.\nThe ban requires a centered follower approach, not every rear-corner contact. If the attacker\nbegan in an adjacent lane and its executed lateral motion carried it diagonally into EGO's\nrear corner, evaluate that actual cut-in even when its center remains behind EGO and its final\nheading is close to EGO's. The contact event records the offset and oriented corners. Read\nall corner coordinates carefully: one front corner can extend past EGO's rear x boundary\nwhile the other remains behind. Do not call the entire attacker body \"behind EGO\" in that case.\nThe event also reports narrow_rear_ban_geometry, computed at the exact swept-contact pose\nusing the same centered-follower geometry as the runtime guard. Its three booleans must ALL\nbe true for that narrow universal ban. If centerline_aligned=false, do not call the contact\na forbidden centered rear-end merely because headings match or the attacker center is behind.\nStill assess the fixed goal's attacker, maneuver and EGO interaction on executed evidence.\nFor an adjacent-lane cut-in goal, \"side collision\" and \"lateral body contact\" name the LATERAL\nCONFLICT MECHANISM, not the exact EGO body panel struck. Once the attacker really moves\nlaterally toward EGO's path and makes body contact with EGO, count the contact whether it\nlands on the flat side, front corner, front edge, rear corner, or rear edge. This also\napplies when the attacker center is ahead of or behind EGO at contact. Do not put\n\"wrong front/side/rear impact location\" in missing for such a goal. Only reject if the\nrequested lateral intrusion did not happen, there was no EGO body contact, or the event\nwas the prohibited centered same-heading direct rear-end.\nDo not assume that \"cut in EGO travel path\" requires same-direction traffic. An opposing or crossing\nvehicle can intrude into EGO's path and touch EGO's side. Opposing heading alone cannot disprove\n\"lateral body contact\"; use the oriented body geometry to identify which part of EGO was touched.\nCheck every named attacker, maneuver,\ndirection, ordering and requested interaction. EGO is controlled by a separate planner: its behavior is an\nobservation, not an attacker command, and temporal correlation does not prove the attacker caused it.\nInterpret \"deviate left/deviate right from the current route\" in the ATTACKER's own directed-route frame unless the goal\nexplicitly specifies EGO's frame. Positive lateral offset is left, negative is right in that frame.\nUse the executed lateral offset change and applied control to verify the requested direction;\nmoving toward EGO in EGO coordinates does not by itself establish an attacker-left maneuver.\nFor a multi-vehicle goal claiming that a support car blocked, occupied, narrowed, or limited an escape\ncorridor, verify the executed support-car position and timing actually occupied that corridor during the\nmain interaction. A spawned or stationary car far ahead is insufficient evidence of such a contribution.\nApply the same scrutiny to claims such as \"create a traffic-flow constraint/lateral traffic-flow constraint\". A support car crossing\nthe infinite extension of EGO's route far ahead does not establish a constraint during an almost\nimmediate main contact. Use contact_event.all_attackers_at_contact to compare its executed\nlongitudinal/lateral position and body clearance at the first contact with the claimed corridor.\nIf the support car remains far ahead when the main car hits EGO, its crossing direction or small\nlateral offset alone is NOT evidence it constrained EGO during that interaction. A support car\ncan satisfy a literal \"cross EGO's path\" action without satisfying the additional claim that\nit formed a constraint. Do not infer a causal contribution from the chosen route, a constant-speed\ncommand, or a later opportunity that simulation never reached.\nIf the goal only asks the support car to hold a position, assess that literal action without inferring a\ncausal effect on EGO. Do not count an attacker-attacker collision as the requested EGO interaction.\nDo not require any condition absent from the language goal or the universal rear-end ban.\nInterpret the FIXED goal's verbs literally: \"cut in/lane-change cut-in with contact\" can describe a collision WHILE\nthe lane change is underway. Unless the goal explicitly says \"complete the lane change\", \"fully enter the lane\", or an\nequivalent completed state, do not require the attacker to finish the lane change before contact.\nThe measurement departed_original_lane is true only when the entire vehicle footprint has LEFT its\noriginal lane. False does not mean the vehicle never began entering EGO's lane. Never use that\nboolean alone as a pass/fail condition. Instead check the executed lateral motion in the named\ndirection, whether the body entered the target-lane boundary during that motion, and the actual\ncontact geometry. A clear lane-change collision may satisfy \"cut in and make contact\" even while part of the\nattacker remains in its original lane. Do not accept a merely intended lane change, proximity as\ncollision, or contact without evidence of the requested maneuver. Words such as\n\"maintain\", \"continue\" or \"follow\" require a meaningful measured interval before contact; one instantaneous pose or\nan immediate collision is insufficient evidence.\nFor contact location or struck side, use attacker_relative_to_ego_at_contact in the contact event:\nlongitudinal_m positive means attacker center in front of EGO, negative means behind; lateral_m positive\nmeans attacker center to EGO's left, negative means to its right. Initial position and lane-change direction\nalone do not prove the side of contact. A slightly negative longitudinal CENTER position alone does not\nprove a rear-end: a vehicle alongside EGO can have its center just behind while its body touches EGO's\nside. Check the vehicle dimensions, lateral position, headings and executed poses to distinguish rear\ncontact from a lateral body contact. A center offset only a small fraction of a vehicle length behind\nEGO is not by itself a rear impact. Do not automatically reject every longitudinal_m < 0 contact.\nThe collision detector checks swept ORIENTED rectangles, so contact is possible even when center-to-center\nlateral separation exceeds the sum of the unrotated half-widths. The contact event includes\nbody_geometry_at_contact: attacker corners are [forward,left] in EGO coordinates and EGO occupies\n[-ego_half_length_m,+ego_half_length_m] by [-ego_half_width_m,+ego_half_width_m]. Use those corners to\nassess whether the bodies met at EGO's side, front, or rear. Do not override a detected collision by\nassuming axis-aligned rectangles or by comparing centers to vehicle widths alone. An attacker front\ncorner can make a valid side-body contact with EGO during a lane change; this is not automatically a\nforbidden rear-end.\nCheck numeric bounds accurately: a corner is inside EGO only when BOTH its forward and lateral\ncoordinates are within the respective bounds. An attacker edge can intersect an EGO corner even\nwhen none of the four attacker corners lies inside EGO. Do not falsely claim an outside corner\nis inside; describe edge/corner intersection when that is what the swept collision shows.\nbody_geometry_at_contact.attacker_front_edge_crosses_ego_rear_boundary and its intersection\nlateral coordinate make this distinction explicit. Cite the edge crossing, not an outside\ncorner, when explaining a rear-corner impact.\nFor initial \"left/right of EGO\", use\nactor_measurements.initial_relative_to_ego, which rotates world positions into EGO's heading. Never infer\nleft/right from world X or Y coordinates alone. Treat left/right in a maneuver as the\nrequested movement direction, but do not make a fine-grained struck-panel label a separate\nsuccess condition. If the same requested cut-in or crossing maneuver occurred and the\nattacker made real body contact with EGO, accept minor side-versus-corner or\nfront-versus-corner differences. For a real cut-in collision, even a full front-edge or\nrear-edge contact counts; the panel label is not an acceptance condition. Never substitute\nan unrelated head-on event, a centered\nstraight rear-end, or an attacker-background collision for the requested interaction.\nIf the goal merely says \"cut in and make body contact\", verify the cut-in maneuver and body contact;\ndo not add an unstated requirement for a side impact or an ahead-of-EGO contact. Return ONLY JSON\n{\"goal_met\":true/false,\"evidence\":[\"specific measured fact\"],\"missing\":[\"missing or contradictory goal element\"],\"reason\":\"brief English verdict\"}.\nEach missing item must be a specific unmet condition under 300 characters, not deliberation or a\nself-correction. Keep reason under 500 characters and consistent with goal_met. If the main\nactor maneuver and EGO interaction are supported, return goal_met=true and missing=[].\nEvidence should identify the actual maneuver and contact or measured near miss."


def review_verdict(client, payload):
    """Ask the independent reviewer to repair an internally contradictory verdict."""
    for attempt in range(2):
        verdict=client.complete(REVIEW_PROMPT,payload)
        try:
            exact(verdict,{'goal_met','evidence','missing','reason'},'semantic verdict')
            if type(verdict['goal_met']) is not bool or not isinstance(verdict['evidence'],list) or not isinstance(verdict['missing'],list):
                raise ValueError('Malformed semantic verdict')
            if any(not isinstance(v,str) for v in verdict['evidence']+verdict['missing']):
                raise ValueError('Reviewer evidence and missing entries must be strings')
            text(verdict['reason'],'verdict reason',2000)
        except (ValueError,TypeError,KeyError) as exc:
            payload=dict(payload,format_error=str(exc),format_instruction='Return exactly goal_met(boolean), evidence(list of strings), missing(list of strings), reason(string), with no other fields. Reassess the same factual evidence.')
            continue
        if (verdict['goal_met'] == (len(verdict['missing']) == 0) and
            all(0<len(v)<=300 for v in verdict['missing']) and len(verdict['reason'])<=500):
            return verdict
        payload=dict(payload,previous_inconsistent_verdict=dict(goal_met=verdict['goal_met'],
            missing=[v[:300] for v in verdict['missing'][:2]],reason=verdict['reason'][:300]),
            consistency_error=('goal_met=true requires missing=[]; goal_met=false requires a short '
                               'specific unmet condition. Prior missing/reason was contradictory or '
                               'rambling. Reassess independently. No deliberation in JSON fields.'))
    raise ValueError('Independent reviewer returned an inconsistent or non-concise verdict twice')


def validate_first_lateral_action(goal, actions, snapshot):
    """Catch a clear left/right contradiction before executing an immutable goal."""
    if len(goal['participants'])!=1:return
    language=goal['language_goal']
    match=re.search("\u5411([\u5de6\u53f3])(?:\u504f\u79bb|\u53d8\u9053|\u5207\u5165|\u5e76\u7ebf)|([\u5de6\u53f3])\u53d8\u9053",language)
    if not match:return
    expected='left' if (match.group(1) or match.group(2))=="\u5de6" else 'right'
    action=actions[0]
    attacker=next(a for a in snapshot['actors'] if a['id']==action['actor_id'])
    rel=attacker['relative_to_ego']
    if (action['primitive'] in ('side_overlap','staged_cut_in') and rel.get('longitudinal_m',0)<0 and
            rel.get('heading_delta_rad',math.inf)<.35 and
            abs(rel.get('lateral_m',0))>1.5):
        original_side='left' if rel['lateral_m']>0 else 'right'
        if action['side']!=original_side:
            raise ValueError(action['primitive']+'.side means the side of EGO where the attacker should '
                'remain, NOT the direction of the lane change. This attacker starts on EGO\'s '+
                original_side+'; for an adjacent-behind side contact choose side='+original_side+
                ' and a nonzero lateral separation, then use speed to time the contact.')
    options={o['route_id']:o for a in snapshot['actors'] if a['id']==action['actor_id'] for o in a['route_options']}
    kind=options[action['route_id']]['kind']
    actual=None
    if action['primitive']=='lane_change':actual=kind
    elif (action['primitive']=='lateral_offset' and kind=='current' and
          options[action['route_id']].get('distance_to_route_m',0)<.75):
        offset=action['offset_m']
        if abs(offset)>.25:actual='left' if offset>0 else 'right'
    if actual and actual!=expected:
        raise ValueError('First lateral action contradicts the fixed language goal: goal says attacker-'+
            expected+' but command targets attacker-'+actual+'. Left/right is relative to the attacker\'s '
            'directed route, including for opposing traffic. Choose a matching offered route or offset.')


class Stage1Tools:
    def __init__(self,world,client,memory,root,episode_id,coverage_brief=None):
        self.world=world;self.client=client;self.memory=memory;self.root=Path(root);self.episode_id=episode_id
        self.proposal=None;self.goal_hash=None;self.monitor=None;self.trace=[];self.evaluation=None
        self.stopped=False;self.inspected=False;self.topology_checked=set();self.occupancy_checked=set();self.action_history=[];self.last_control_feedback=[]
        self.last_action_rejection=None
        self.construction_findings={};self.spawn_shortlist=[];self.seen_queries=set()
        self.duplicate_query_counts={};self.query_suppressed_at={}
        self.spawn_search_revision=None
        self.coverage_brief=coverage_brief
        self.root.mkdir(parents=True,exist_ok=True)

    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))

    def context(self):
        scene=self.world.public_snapshot()
        required={'ego'}|({p['actor_id'] for p in self.proposal['participants']} if self.proposal else set())
        retained=set(required)
        if self.proposal is None:
            retained.update(row['actor_id'] for row in
                (self.coverage_brief or {}).get('existing_route_shortlist',[])[:5])
            retained.update(self.topology_checked)
            retained.update(getattr(self.world,'spawned',[]))
        ranked=sorted(scene['actors'],key=lambda a:(a['id'] not in retained,
            a['relative_to_ego'].get('distance_m',0)))
        limit=max(len(retained),12) if self.proposal is None else max(len(required),8)
        scene=dict(scene,actors=ranked[:limit],omitted_distant_actors=max(0,len(ranked)-limit))
        goal=None if self.proposal is None else copy.deepcopy(self.proposal)
        if self.stopped:
            return dict(scene=scene,goal=goal,goal_sha256=self.goal_hash,stopped=True,
                feedback=self.monitor.report() if self.monitor else None,
                reviewer_verdict=self.evaluation['llm_verdict'] if self.evaluation else None,
                recent_attacker_actions=self.action_history[-3:],available_tools=self.available_tools())
        current_plan=self.world.ego_current_plan() if hasattr(self.world,'ego_current_plan') else None
        forecast=self.world.forecast_ego() if hasattr(self.world,'forecast_ego') else None
        interaction=None
        if self.proposal and hasattr(self.world,'interaction_opportunities'):
            interaction=self.world.interaction_opportunities(sorted(required-{'ego'}))
        occupancy=None
        if self.proposal and hasattr(self.world,'route_occupancy'):
            occupancy=self.world.route_occupancy(sorted(required-{'ego'}))
        action_affordances=[]
        for aid in sorted(required-{'ego'}):
            actor=self.world.states[aid]
            neighbors=[]
            for other_id,other in self.world.states.items():
                if other_id==aid:continue
                rel=relative(other,actor)
                if rel['distance_m']>35:continue
                neighbors.append(dict(actor_id=other_id,body_clearance_m=round(clearance(actor,other),2),
                                      relative_to_attacker=rel))
            neighbors.sort(key=lambda item:item['body_clearance_m'])
            options=next((a['route_options'] for a in scene['actors'] if a['id']==aid),[])
            ego=self.world.states['ego'];route_geometry=[]
            for option in options:
                if not hasattr(self.world,'routes') or option['route_id'] not in self.world.routes:
                    continue
                points=self.world.routes[option['route_id']]['points']
                attacker_s,attacker_nearest,attacker_heading,_=project(points,[actor['x'],actor['y']])
                ego_s,ego_nearest,ego_heading,_=project(points,[ego['x'],ego['y']])
                route_end_s,_,_,_=project(points,points[-1])
                forward_gap=ego_s-attacker_s
                body_reach=max(0.,forward_gap-.5*(actor['length']+ego['length']))
                def signed_offset(vehicle,nearest,heading):
                    return round(-(vehicle['x']-nearest[0])*math.sin(heading)+
                                  (vehicle['y']-nearest[1])*math.cos(heading),2)
                route_geometry.append(dict(route_id=option['route_id'],
                    remaining_route_m=round(max(0.,route_end_s-attacker_s),2),
                    ego_forward_from_attacker_m=round(forward_gap,2),
                    ego_lateral_offset_m=signed_offset(ego,ego_nearest,ego_heading),
                    attacker_lateral_offset_m=signed_offset(actor,attacker_nearest,attacker_heading),
                    estimated_speed_to_body_reach_in_1s_mps=(round(ego['speed']+body_reach,2)
                        if forward_gap>0 else None)))
            action_affordances.append(dict(actor_id=aid,
                offered_routes=[dict(route_id=o['route_id'],kind=o['kind']) for o in options],
                ego_route_geometry=route_geometry,
                closest_vehicles=neighbors[:5],
                side_overlap_hint=(dict(current_side_of_ego='left' if relative(actor,ego)['lateral_m']>0 else 'right',
                    suggested_partial_overlap_separation_m=round((actor['width']+ego['width'])/2-.4,2),
                    meaning='side is the desired side of EGO, not steering direction')
                    if relative(actor,ego)['longitudinal_m']<0 and
                       relative(actor,ego)['heading_delta_rad']<.35 and
                       abs(relative(actor,ego)['lateral_m'])>1.5 else None)))
        result=dict(scene=scene,goal=goal,goal_sha256=self.goal_hash,
            remaining_simulation_s=round(self.world.config['max_simulation_s']-self.world.time,3),
            stopped=self.stopped,feedback=self.monitor.report() if self.monitor else None,
            reviewer_verdict=self.evaluation['llm_verdict'] if self.evaluation else None,
            ego_current_plan=current_plan,ego_forecast=forecast,interaction_opportunities=interaction,
            route_occupancy=occupancy,action_affordances=action_affordances,
            recent_attacker_actions=self.action_history[-3:],last_control_feedback=self.last_control_feedback,
            last_action_rejection=self.last_action_rejection,
            available_tools=self.available_tools())
        if self.proposal is None:
            result['construction_findings']=list(self.construction_findings.values())
            result['spawn_shortlist']=self.spawn_shortlist
            if self.coverage_brief:result['coverage_brief']=self.coverage_brief
        return result

    def available_tools(self):
        if not self.inspected:return ['inspect_scene','finish']
        # A duplicate query is rejected by its own signature; other actors must
        # remain inspectable when the agent is still searching for a valid goal.
        queries=[q for q in ('forecast_ego','inspect_topology',
            'inspect_interaction_opportunities','inspect_route_occupancy')
            if self.query_suppressed_at.get(q)!=round(getattr(self.world,'time',0.),3)]
        if self.proposal is None:
            spawn_capacity=len(getattr(self.world,'spawned',[]))<getattr(
                self.world,'config',{}).get('max_spawned_vehicles',2)
            search=(['search_spawn_opportunities'] if spawn_capacity and
                    self.spawn_search_revision!=len(getattr(self.world,'spawned',[])) else [])
            spawn=['spawn_actor'] if spawn_capacity else []
            return queries+search+spawn+['define_goal','finish']
        if self.stopped:return ['finish'] if self.evaluation else ['evaluate','finish']
        return [q for q in queries if q!='inspect_topology']+['act','evaluate','finish']

    def inspect_scene(self):
        self.inspected=True
        current_plan=self.world.ego_current_plan() if hasattr(self.world,'ego_current_plan') else None
        return dict(ok=True,observation=self.world.public_snapshot(),
            ego_current_plan=current_plan,
            ego_forecast=self.world.forecast_ego() if hasattr(self.world,'forecast_ego') else None,
            construction_policy='Choose one goal freely. Existing, spawned, and mixed attacker teams are supported; use a supporting actor only if it has a meaningful feasible role.')

    def forecast_ego(self,horizon_s):
        if not self.inspected:raise ValueError('Inspect the scene first')
        number(horizon_s,.5,self.world.config.get('ego_forecast_horizon_s',4.),'horizon_s')
        current_plan=self.world.ego_current_plan() if hasattr(self.world,'ego_current_plan') else None
        return dict(ok=True,ego_current_plan=current_plan,forecast=self.world.forecast_ego(horizon_s))

    def _require_actor_ids(self,actor_ids):
        if not isinstance(actor_ids,list) or not 1<=len(actor_ids)<=4 or len(set(actor_ids))!=len(actor_ids):
            raise ValueError('Query 1..4 distinct actor IDs')
        invalid=[aid for aid in actor_ids if aid not in self.world.states or aid=='ego']
        if invalid:
            valid=[a['id'] for a in self.world.public_snapshot()['actors'] if a['id']!='ego' and a['controllable']]
            raise ValueError('Unknown non-EGO actor ID(s): '+', '.join(invalid)+'. Copy exact ID(s) from current state: '+', '.join(valid[:20]))

    def _duplicate_query(self,name,actor_ids,horizon_s):
        key=(name,tuple(sorted(actor_ids)),round(float(horizon_s),3),round(self.world.time,3))
        covered_by=next((prior for prior in self.seen_queries
            if prior[0]==name and set(key[1]).issubset(prior[1]) and
               key[2]<=prior[2] and key[3]==prior[3]),None)
        if covered_by:
            repeats=self.duplicate_query_counts.get(covered_by,0)+1
            self.duplicate_query_counts[covered_by]=repeats
            if repeats>=2:
                self.query_suppressed_at['inspect_'+name+'_opportunities' if name=='interaction'
                    else 'inspect_route_occupancy' if name=='occupancy'
                    else 'inspect_topology']=round(self.world.time,3)
            return dict(ok=False,error='This actor query was already covered at the unchanged '
                'simulation time by an earlier call. Its conclusion is preserved in construction_findings. '
                'Inspect a different actor with another available tool or choose a supported goal. '
                'Do not repeat this query again.')
        self.seen_queries.add(key)
        return None

    def inspect_interaction_opportunities(self,actor_ids,horizon_s):
        if not self.inspected:raise ValueError('Inspect the scene first')
        self._require_actor_ids(actor_ids)
        number(horizon_s,.5,self.world.config.get('ego_forecast_horizon_s',4.),'horizon_s')
        duplicate=self._duplicate_query('interaction',actor_ids,horizon_s)
        if duplicate:return duplicate
        result=dict(ok=True,interaction_opportunities=self.world.interaction_opportunities(actor_ids,horizon_s))
        self.topology_checked.update(actor_ids)
        for item in result['interaction_opportunities']['actors']:
            finding=self.construction_findings.setdefault(item['actor_id'],dict(actor_id=item['actor_id']))
            finding.update(current_relation=item['current_relation'],
                initial_body_clearance_m=item.get('initial_body_clearance_m'),
                lead_merge_contact_weak=item.get('lead_merge_contact_weak'),
                long_chase_contact_weak=item.get('long_chase_contact_weak'),
                nominal_current_route_min_body_clearance_m=item.get('nominal_current_route_min_body_clearance_m'),
                interaction_routes=[{k:route[k] for k in ('route_id','kind',
                    'usable_spatiotemporal_conflict','estimated_body_clearance_at_route_match_m',
                    'initial_distance_to_route_center_m','actor_forward_distance_m') if k in route}
                    for route in item['opportunities'][:2]])
        if hasattr(self.world,'route_occupancy'):
            occupancy_horizon=min(float(horizon_s),3.)
            result['route_occupancy']=self.world.route_occupancy(actor_ids,occupancy_horizon)
            self.seen_queries.add(('occupancy',tuple(sorted(actor_ids)),
                round(occupancy_horizon,3),round(self.world.time,3)))
            self.occupancy_checked.update(actor_ids)
            self._remember_occupancy(result['route_occupancy'])
            self.write('last_route_occupancy.json',dict(ok=True,route_occupancy=result['route_occupancy']))
        self.write('last_interaction_query.json',result)
        return result

    def inspect_route_occupancy(self,actor_ids,horizon_s):
        if not self.inspected:raise ValueError('Inspect the scene first')
        if actor_ids==[]:
            raise ValueError('An empty actor_ids list cannot inspect a spawn slot. '
                'search_spawn_opportunities already includes traffic_preview for each slot. '
                'Choose a slot, call spawn_actor, then pass its returned actor.id to '
                'inspect_route_occupancy.')
        self._require_actor_ids(actor_ids)
        number(horizon_s,.5,4.,'horizon_s')
        duplicate=self._duplicate_query('occupancy',actor_ids,horizon_s)
        if duplicate:return duplicate
        result=dict(ok=True,route_occupancy=self.world.route_occupancy(actor_ids,horizon_s))
        self.occupancy_checked.update(actor_ids)
        self._remember_occupancy(result['route_occupancy'])
        self.write('last_route_occupancy.json',result)
        return result

    def inspect_topology(self,actor_ids,horizon_s):
        if not self.inspected:raise ValueError('Inspect the scene first')
        self._require_actor_ids(actor_ids)
        number(horizon_s,.5,self.world.config['max_simulation_s'],'horizon_s')
        duplicate=self._duplicate_query('topology',actor_ids,horizon_s)
        if duplicate:return duplicate
        result=dict(ok=True,topology=[self.world.route_topology(aid,horizon_s,True) for aid in actor_ids])
        for item in result['topology']:
            self.construction_findings.setdefault(item['actor_id'],dict(actor_id=item['actor_id']))[
                'directed_route_conflict_possible']=item['potential_conflict']
        self.topology_checked.update(actor_ids);self.write('last_topology_query.json',result)
        return result

    def _remember_occupancy(self,occupancy):
        for item in occupancy['actors']:
            hazards=[]
            center_blockers=[]
            for route in item['routes']:
                for corridor in route['lateral_corridors']:
                    for hazard in corridor['non_target_hazards']:
                        hazards.append(dict(route_id=route['route_id'],offset_m=corridor['offset_m'],
                            other_actor_id=hazard['other_actor_id'],time_s=hazard['time_s'],
                            forward_distance_m=hazard['forward_distance_m'],
                            nominal_body_clearance_m=hazard['nominal_body_clearance_m']))
                        if (route['kind']=='current' and abs(corridor['offset_m'])<.5
                            and 0<=hazard['forward_distance_m']<=12
                            and hazard['time_s']<=3.0
                            and hazard['nominal_body_clearance_m']<=.5):
                            center_blockers.append(dict(route_id=route['route_id'],
                                other_actor_id=hazard['other_actor_id'],time_s=hazard['time_s'],
                                forward_distance_m=hazard['forward_distance_m']))
            finding=self.construction_findings.setdefault(item['actor_id'],dict(actor_id=item['actor_id']))
            entry_distance={route['route_id']:route.get('initial_distance_to_route_center_m',5.)
                for route in finding.get('interaction_routes',[])}
            hazards.sort(key=lambda h:(abs(h['offset_m'])>
                entry_distance.get(h['route_id'],5.)+.5,
                h['nominal_body_clearance_m'],h['time_s'],h['forward_distance_m']))
            distinct=[];seen=set()
            for hazard in hazards:
                key=(hazard['route_id'],hazard['other_actor_id'])
                if key in seen:continue
                seen.add(key);distinct.append(hazard)
            center_blockers.sort(key=lambda h:(h['forward_distance_m'],h['time_s']))
            finding['nearest_non_target_hazards']=distinct[:6]
            finding['current_route_center_blockers']=center_blockers[:2]

    def search_spawn_slots(self,relation,reason):
        if not self.inspected:raise ValueError('Inspect existing actors first')
        if self.proposal is not None:raise ValueError('Spawn before fixing the goal')
        text(reason,'spawn reason')
        revision=len(getattr(self.world,'spawned',[]))
        if self.spawn_search_revision==revision:
            return dict(ok=False,error='The complete spawn-slot search was already returned for this '
                'unchanged traffic state. Use spawn_shortlist in context, define a supported goal, '
                'or finish; after a successful spawn you may search again.')
        if relation not in ('any','ahead','behind','left_adjacent','right_adjacent','crossing','opposing'):
            raise ValueError('Unknown relative relation')
        matches=[]
        for actor in self.world.public_snapshot()['actors']:
            if actor['controllable'] and (relation=='any' or actor['relative_to_ego']['relation']==relation):
                if not any(other['id']!=actor['id'] and overlap(actor,other) for other in self.world.states.values()):
                    matches.append(actor['id'])
        slots=self.world.search_spawn_slots(relation)
        self.spawn_search_revision=revision
        self.spawn_shortlist=[dict(slot_id=s['slot_id'],relative_to_ego=s['relative_to_ego'],
            route_id=s['route_options'][0]['route_id'],
            potential_conflict=s['topology']['potential_conflict'],
            contact_setup_priority=s['contact_setup_priority'],
            traffic_risk_level=s['traffic_preview']['risk_level'],
            lead_merge_contact_weak=s['forecast_interaction'].get('lead_merge_contact_weak'),
            long_chase_contact_weak=s['forecast_interaction'].get('long_chase_contact_weak'),
            nearest_traffic=s['traffic_preview']['safest_current_route'],
            nominal_current_route_min_body_clearance_m=s['forecast_interaction'].get('nominal_current_route_min_body_clearance_m'))
            for s in slots]
        return dict(ok=True,slots=slots,existing_actor_ids=matches,reason=reason)

    def search_spawn_opportunities(self,reason):
        """Offer all legal spatial relations in one unbiased construction query."""
        return self.search_spawn_slots('any',reason)

    def spawn_actor(self,slot_id,speed_mps,reason):
        if not self.inspected or self.proposal is not None:raise ValueError('Inspect and search before fixing the goal')
        text(reason,'reason');number(speed_mps,0,10,'speed_mps')
        slot=next((s for s in self.spawn_shortlist if s['slot_id']==slot_id),None)
        if slot is None:
            raise ValueError('Choose a slot ID from the latest search_spawn_opportunities result')
        if not slot['potential_conflict']:
            raise ValueError('This slot has no directed forward route interaction with EGO. '
                'Choose a returned slot with potential_conflict=true; do not spawn a disconnected '
                'vehicle and repeatedly define an impossible contact goal.')
        if slot['traffic_risk_level']=='likely_blocked':
            raise ValueError('This spawn slot has a predicted near-term background-car collision '
                'on its available route. Choose a clearer slot or an existing vehicle.')
        try:actor=self.world.spawn(slot_id,speed_mps)
        except ValueError as exc:
            if 'now occupied' in str(exc):
                return dict(ok=False,error='That slot was occupied by the previous spawn. '
                    'Call search_spawn_opportunities again for the refreshed traffic state; '
                    'do not reuse the old slot list.')
            raise
        self.seen_queries.clear()
        self.duplicate_query_counts.clear();self.query_suppressed_at.clear()
        self.write('spawned_actors.json',[a for a in self.world.states.values() if a['source']=='spawned'])
        current_plan=self.world.ego_current_plan() if hasattr(self.world,'ego_current_plan') else None
        forecast=self.world.forecast_ego() if hasattr(self.world,'forecast_ego') else None
        inspection=self.inspect_interaction_opportunities([actor['id']],
            self.world.config.get('ego_forecast_horizon_s',4.))
        interaction=inspection['interaction_opportunities']
        actor_interaction=interaction['actors'][0]
        close_hazards=[]
        for occupancy_actor in (inspection.get('route_occupancy') or {}).get('actors',[]):
            for route in occupancy_actor['routes']:
                for corridor in route['lateral_corridors']:
                    for hazard in corridor['non_target_hazards']:
                        if (hazard['forward_distance_m']<=12 and hazard['time_s']<=2.5 and
                                hazard['nominal_body_clearance_m']<=1.5):
                            close_hazards.append(dict(route_id=route['route_id'],
                                lateral_offset_m=corridor['offset_m'],
                                other_actor_id=hazard['other_actor_id'],
                                forward_distance_m=hazard['forward_distance_m'],
                                time_s=hazard['time_s'],
                                nominal_body_clearance_m=hazard['nominal_body_clearance_m']))
        close_hazards.sort(key=lambda h:(h['forward_distance_m'],h['time_s']))
        diagnostic=None
        if forecast and forecast.get('samples'):
            ego=self.world.states['ego'];minimum=float('inf')
            for sample in forecast['samples']:
                xy=sample['world_xy_m']
                future=dict(ego,x=xy[0],y=xy[1],yaw=wrap(ego['yaw']+sample['yaw_change_rad']))
                minimum=min(minimum,clearance(actor,future))
            rel=relative(actor,ego)
            stopped=forecast['samples'][-1]['speed_mps']<.5
            lead_block=(speed_mps<=.5 and rel['longitudinal_m']>0 and
                        abs(rel['lateral_m'])<2.5 and rel['heading_delta_rad']<.7 and
                        stopped and minimum>1.)
            ahead_merge_warning=(rel['longitudinal_m']>5 and abs(rel['lateral_m'])>1.5 and
                                 rel['heading_delta_rad']<.7 and stopped and minimum>1.)
            diagnostic=dict(nominal_min_body_clearance_m=round(minimum,2),
                ego_stops_in_nominal_forecast=stopped,
                stationary_lead_contact_unlikely=lead_block,
                ahead_adjacent_contact_unlikely=ahead_merge_warning,
                same_direction_lead_in_ego_corridor=actor_interaction.get('same_direction_lead_in_ego_corridor',False),
                lead_merge_contact_weak=actor_interaction.get('lead_merge_contact_weak',False),
                long_chase_contact_weak=actor_interaction.get('long_chase_contact_weak',False),
                nominal_current_route_min_body_clearance_m=actor_interaction.get('nominal_current_route_min_body_clearance_m'),
                nearby_non_target_hazards=close_hazards[:4],
                guidance=('A stationary lead remains separated from a stopped EGO in the refreshed forecast. '
                          'This vehicle cannot reverse; inspect another spawn or choose an observable non-contact goal before fixing it.'
                          if lead_block else
                          'This actor is already AHEAD in the same EGO corridor, not in an adjacent lane. '
                          'Do not call it an adjacent cut-in or assume the route-match overlap proves body contact; '
                          'compare a real side/crossing actor or choose a different observable goal.'
                          if actor_interaction.get('same_direction_lead_in_ego_corridor') else
                          'This attacker is ahead on a roughly same-direction road and the nominal current-route '
                          'body gap stays over 3 m. EGO can replan and brake behind a lead cut-in; route-match '
                          'overlap alone is weak evidence for a contact goal. Compare a close crossing or '
                          'adjacent-behind setup, or choose a measurable non-contact interaction.'
                          if actor_interaction.get('lead_merge_contact_weak') else
                          'This is a long same-direction chase from behind. The initial body gap exceeds 15 m; '
                          'several EGO replans can occur before the attacker reaches it. Prefer a nearer '
                          'side/crossing setup for a short-window contact goal.'
                          if actor_interaction.get('long_chase_contact_weak') else
                          'This spawn has a near-term non-target occupancy hazard. Check the listed route, '
                          'lateral corridor, actor and time before setting a goal or braking into that car. '
                          'A valid spawn footprint does not make the later route clear.'
                          if close_hazards else
                          'This adjacent attacker is already ahead and EGO stops behind it in the refreshed forecast. '
                          'A merge-and-brake contact goal is weak; compare adjacent-behind slots before fixing the goal.'
                          if ahead_merge_warning else
                          'Forecast is advisory; verify route and goal against the refreshed state.'))
        return dict(ok=True,actor=actor,observation=self.world.public_snapshot(),
            ego_current_plan=current_plan,ego_forecast=forecast,spawn_diagnostic=diagnostic,
            interaction_opportunities=interaction,route_occupancy=inspection.get('route_occupancy'),
            next='Spawned actor topology, forecast interaction, and route occupancy were inspected automatically. '
                 'Use this refreshed evidence to choose one achievable goal; repeating inspect_topology is optional.')

    def define_goal(self,language_goal,rationale,participants):
        if self.proposal is not None:raise ValueError('The goal is already fixed')
        if not self.inspected:raise ValueError('Inspect the scene first')
        goal=validate_goal(dict(language_goal=language_goal,rationale=rationale,participants=participants),
                           self.world.public_snapshot(),self.world.config)
        unchecked=[p['actor_id'] for p in participants if p['actor_id'] not in self.topology_checked]
        if unchecked:raise ValueError('Inspect topology before fixing these attackers: '+', '.join(unchecked))
        contact_goal=requires_body_contact(language_goal)
        if contact_goal:
            rejected=[p['actor_id'] for p in participants if
                self.construction_findings.get(p['actor_id'],{}).get('contact_route_rejected')]
            if len(rejected)==len(participants):
                raise ValueError('These same actors were already ruled out for a body-contact goal: '+
                    ', '.join(rejected)+'. Choose a different evidenced actor, a measurable near-miss '
                    'target, or finish unsupported; changing the contact wording cannot fix the route.')
            unchecked_occupancy=[p['actor_id'] for p in participants if p['actor_id'] not in self.occupancy_checked]
            if unchecked_occupancy:
                raise ValueError('Inspect interaction opportunities or route occupancy for every contact attacker before fixing its goal: '+', '.join(unchecked_occupancy))
            weak_leads=[p['actor_id'] for p in participants if
                self.construction_findings.get(p['actor_id'],{}).get('lead_merge_contact_weak')]
            if len(weak_leads)==len(participants):
                raise ValueError('Every selected attacker is an ahead lead whose nominal route stays clear of EGO. '
                    'A cut-in can make EGO stop without body contact. Inspect another existing actor or search '
                    'spawn opportunities for a side/crossing encounter; alternatively choose a measurable '
                    'non-contact goal supported by this geometry.')
            if hasattr(self.world,'interaction_opportunities'):
                interactions=self.world.interaction_opportunities([p['actor_id'] for p in participants])['actors']
                plausible=[];best_clearance=float('inf')
                for item in interactions:
                    speed=self.world.states[item['actor_id']]['speed']
                    plausible.extend((item['actor_id'],route) for route in
                        plausible_contact_routes(item,speed))
                    clearances=[r['estimated_body_clearance_at_route_match_m']
                        for r in item['opportunities']]
                    if clearances:best_clearance=min(best_clearance,*clearances)
                if not plausible:
                    for item in interactions:
                        aid=item['actor_id']
                        self.construction_findings.setdefault(aid,dict(actor_id=aid))[
                            'contact_route_rejected']=True
                    raise ValueError('No selected attacker has a route with near-term contact geometry '
                        'and enough time to enter it laterally (best route-match body clearance {:.2f} m). '
                        'The route-intersection flag alone is insufficient. Choose another actor/slot or '
                        'a specific observable near-miss goal; do not claim body contact without evidence.'
                        .format(best_clearance))
                def early_background_blocker(aid,route):
                    initial_lateral=route.get('initial_distance_to_route_center_m',0.)
                    return next((hazard for hazard in
                        self.construction_findings.get(aid,{}).get('nearest_non_target_hazards',[])
                        if hazard['route_id']==route['route_id'] and
                        hazard['time_s']<=1.0 and 0<=hazard['forward_distance_m']<=20 and
                        hazard['nominal_body_clearance_m']<=.5 and
                        abs(hazard['offset_m'])<=initial_lateral+.5),None)
                blockers=[early_background_blocker(aid,route) for aid,route in plausible]
                if blockers and all(blockers):
                    raise ValueError('Every plausible contact route has a background vehicle in '
                        'the first second of the attacker lateral transition (for example '+
                        blockers[0]['other_actor_id']+'). A nominal EGO route intersection does '
                        'not make this cut-in corridor clear. Choose a different attacker, spawn '
                        'slot or supported non-contact goal.')
            if len(participants)==1 and re.search("(?:\u4fdd\u6301\u5f53\u524d\u901f\u5ea6|\u6cbf\u5f53\u524d\u8def\u7ebf|\u6cbf\\s*route_[A-Za-z0-9]+\\s*(?:\u884c\u9a76|\u76f4\u884c))",language_goal):
                aid=participants[0]['actor_id']
                blocker=(self.construction_findings.get(aid,{}).get('current_route_center_blockers') or [None])[0]
                if blocker:
                    raise ValueError('A background vehicle '+blocker['other_actor_id']+
                        ' is forecast on the attacker current route '+blocker['route_id']+
                        ' within '+str(round(blocker['forward_distance_m'],1))+' m and '+
                        str(round(blocker['time_s'],1))+' s. A simple keep-speed/route-follow '
                        'contact goal is likely to hit that vehicle before EGO. Choose a clear '
                        'route, different attacker, or a concrete maneuver around the blocker.')
        disconnected=[p['actor_id'] for p in participants if not self.world.route_topology(p['actor_id'])['potential_conflict']]
        if disconnected:raise ValueError('No directed forward interaction is possible for: '+', '.join(disconnected))
        removed=self.world.discard_unselected_spawns({p['actor_id'] for p in participants}) if hasattr(self.world,'discard_unselected_spawns') else []
        if removed:
            self.write('spawned_actors.json',[a for a in self.world.states.values() if a['source']=='spawned'])
        self.proposal=goal;self.goal_hash=digest(goal)
        self.monitor=Monitor(goal,self.world.states,self.world.config,lane_observer=getattr(self.world,'lane_evidence',None))
        self.trace=[dict(time_s=0.,states=copy.deepcopy(self.world.states),controls={})]
        self.write('goal.json',dict(goal=goal,goal_sha256=self.goal_hash))
        self._save_trace()
        return dict(ok=True,goal=goal,goal_sha256=self.goal_hash,
            removed_unselected_spawns=removed,
            next='Use act repeatedly; choose every attacker behavior from each new measured state.')

    def _save_trace(self):
        if not self.monitor:return
        with (self.root/'trial_01.jsonl').open('w') as stream:
            for row in self.trace:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        self.write('latest_metrics.json',self.monitor.report())
        self.write('attacker_actions.json',self.action_history)

    def act(self,actions,reason):
        if not self.proposal:raise ValueError('Fix one language goal first')
        if self.stopped:raise ValueError('Execution has stopped; evaluate it')
        if digest(self.proposal)!=self.goal_hash:raise RuntimeError('Immutable goal changed')
        text(reason,'action reason',2000);duration_s=float(self.world.config.get('decision_interval_s',1.0))
        if duration_s!=1.0:raise ValueError('decision_interval_s must be exactly 1.0')
        snapshot=self.world.public_snapshot()
        actions=validate_reactive_actions(actions,self.proposal['participants'],snapshot)
        if not self.action_history:
            validate_first_lateral_action(self.proposal,actions,snapshot)
        program=[]
        roles={p['actor_id']:p['role'] for p in self.proposal['participants']}
        for action in actions:
            stage=dict(action);stage.pop('actor_id');stage['duration_s']=duration_s
            if action['primitive']=='intercept':
                route=self.world.routes[action['route_id']]['points']
                start,_,_,_=project(route,[self.world.states[action['actor_id']]['x'],self.world.states[action['actor_id']]['y']])
                route_length=sum(math.dist(a,b) for a,b in zip(route,route[1:]))
                target=start+action['target_forward_distance_m']
                if target>route_length+1e-6:
                    raise ValueError('Intercept target exceeds remaining offered route by {:.1f} m'.format(target-route_length))
                stage['target_route_s_m']=target
                stage['deadline_s']=self.world.time+action['target_arrival_time_s']
            program.append(dict(actor_id=action['actor_id'],role=roles[action['actor_id']],delay_s=0,
                                after_role=None,stages=[stage]))
        self.world.controller.set_program(program,self.world.time,preserve=False)
        self.action_history.append(dict(time_s=self.world.time,duration_s=duration_s,actions=actions,reason=reason))
        self.last_action_rejection=None
        remaining=self.world.config['max_simulation_s']-self.world.time
        steps=min(int(round(duration_s/self.world.dt)),int(round(remaining/self.world.dt)))
        if steps<=0:raise ValueError('Simulation time budget exhausted')
        self.evaluation=None
        blocked=None
        try:
            for _ in range(steps):
                try:before,after,controls=self.world.step()
                except ForbiddenRearEnd as exc:
                    attacker=self.world.states[exc.actor_id];ego=self.world.states['ego']
                    pose=relative(attacker,ego)
                    blocked=dict(actor_id=exc.actor_id,time_s=round(exc.time_s,3),
                        contact_relation=exc.relation,reason=str(exc),
                        recovery_hint=dict(attacker_speed_mps=round(attacker['speed'],2),
                            ego_speed_mps=round(ego['speed'],2),
                            attacker_lateral_m=round(pose['lateral_m'],2),
                            suggested_slow_target_mps=round(min(attacker['speed'],ego['speed']),2),
                            instruction='At the unchanged measured pose, first slow to no faster '
                                'than EGO and move laterally away from its centerline; only close '
                                'again after the offset is real. This is advisory, not a controller limit.'))
                    self.last_action_rejection=blocked
                    self.action_history[-1]['rejected_rear_end']=blocked
                    break
                contact=self.monitor.update(before,after,controls,self.world.time,self.world.on_road)
                self.trace.append(dict(time_s=self.world.time,states=after,controls=controls))
                if contact:self.stopped=True;break
        finally:self._save_trace()
        if blocked is not None:
            return dict(ok=False,error=blocked['reason'],time_s=self.world.time,
                forbidden_rear_end=blocked,stopped=False,measured=self.monitor.report(),
                current=self.world.public_snapshot())
        if self.world.time>=self.world.config['max_simulation_s']-1e-8:self.stopped=True
        control_feedback=[]
        for action in actions:
            if action['primitive']!='intercept':continue
            aid=action['actor_id'];route=self.world.routes[action['route_id']]['points']
            progress,_,_,lateral=project(route,[self.world.states[aid]['x'],self.world.states[aid]['y']])
            stage=next(p['stages'][0] for p in program if p['actor_id']==aid)
            remaining=max(0.,stage['target_route_s_m']-progress)
            time_left=stage['deadline_s']-self.world.time
            control_feedback.append(dict(actor_id=aid,remaining_route_distance_m=round(remaining,2),
                target_time_remaining_s=round(time_left,2),route_center_distance_m=round(lateral,2),
                required_average_speed_mps=round(remaining/max(time_left,self.world.dt),2) if time_left>0 else None,
                target_reached=remaining<.5 and lateral<2.))
        self.last_control_feedback=control_feedback
        return dict(ok=True,time_s=self.world.time,stopped=self.stopped,
            measured=self.monitor.report(),control_feedback=control_feedback,current=self.world.public_snapshot())

    def evaluate(self):
        if not self.monitor or len(self.trace)<2:raise ValueError('No executed trajectory to assess')
        if digest(self.proposal)!=self.goal_hash:raise RuntimeError('Goal hash mismatch')
        if self.evaluation and self.evaluation['trace_length']==len(self.trace):
            return dict(ok=True,**self.evaluation)
        ids=self.monitor.participants|{'ego'};stride=max(1,len(self.trace)//45)
        frames=self.trace[::stride]
        if frames[-1] is not self.trace[-1]:frames=frames+[self.trace[-1]]
        samples=[]
        for row in frames:
            samples.append(dict(time_s=row['time_s'],
                actors=[state for aid,state in row['states'].items() if aid in ids],
                applied_controls={aid:value for aid,value in row.get('controls',{}).items() if aid in ids}))
        verdict=review_verdict(self.client,dict(language_goal=self.proposal['language_goal'],
            fixed_participants=self.proposal['participants'],goal_sha256=self.goal_hash,
            descriptive_measurements=self.monitor.report(),executed_trace=samples))
        accepted=bool(verdict['goal_met'] and not verdict['missing'])
        self.evaluation=dict(accepted=accepted,llm_verdict=verdict,
            descriptive_measurements=self.monitor.report(),goal_sha256=self.goal_hash,trace_length=len(self.trace))
        self.write('trial_01_acceptance.json',self.evaluation)
        return dict(ok=True,**self.evaluation)

    def finish(self,status,reason):
        if status not in ('achieved','not_achieved','unsupported'):raise ValueError('Unknown final status')
        text(reason,'finish reason')
        if status=='unsupported' and self.proposal is None and self.coverage_brief:
            unchecked=[row['actor_id'] for row in
                self.coverage_brief.get('existing_route_shortlist',[])[:2]
                if row['actor_id'] not in self.topology_checked]
            if unchecked:
                raise ValueError('Before declaring this scene unsupported, inspect the '
                    'shortlisted existing actor(s) that passed the nominal route/time screen: '+
                    ', '.join(unchecked)+'. A failed goal for another actor does not rule these out.')
        if self.monitor and len(self.trace)>1 and (not self.evaluation or self.evaluation['trace_length']!=len(self.trace)):
            self.evaluate()
        if status=='achieved' and (not self.evaluation or not self.evaluation['accepted']):
            missing=(self.evaluation or {}).get('llm_verdict',{}).get('missing',[])
            raise ValueError('Only the independent Qwen reviewer may accept the fixed language goal; reviewer accepted=false; missing='+str(missing))
        return dict(ok=True,finished=True,status=status,reason=reason,acceptance=self.evaluation)
