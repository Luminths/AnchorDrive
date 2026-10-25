#!/usr/bin/env python3
"""Run original offline Stage 2 on a saved native ReAct Stage 1 scene."""
import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.run_stage2_anchor_extractor import _build_prompts, _extract_json_object, _validate_anchor_response
from tbsim.scene_agent.client import QwenClient
from tbsim.scene_agent.execution_agent import ExecutionTools, restore_goal
from tbsim.scene_agent.export import export_episode
from tbsim.scene_agent.simple_goal_agent import GoalTools
from tbsim.scene_agent.two_stage_backend import NuscTwoStageWorld
from tbsim.scene_agent.backend import ForbiddenRearEnd


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def restore_initial_roundoff(world, saved):
    """Use saved scalar values after checking cross-runtime floating roundoff only."""
    if world.states.keys()!=saved.keys():
        raise ValueError('Restored vehicle IDs differ from selected Stage 1')
    changes=[]
    for aid,state in world.states.items():
        if state.keys()!=saved[aid].keys():
            raise ValueError('Restored vehicle fields differ from selected Stage 1')
        for key,value in state.items():
            expected=saved[aid][key]
            if value==expected:continue
            numeric=isinstance(value,(int,float)) and isinstance(expected,(int,float))
            if key not in ('x','y','yaw','speed','length','width') or not numeric or not (
                    math.isfinite(value) and math.isfinite(expected) and
                    math.isclose(value,expected,rel_tol=1e-14,abs_tol=1e-12)):
                raise ValueError('Restored initial state differs beyond numeric roundoff: '+aid+'/'+key)
            changes.append(dict(actor_id=aid,field=key,restored=value,saved=expected,delta=value-expected))
    if changes:world.states=copy.deepcopy(saved)
    return changes


def build_cases(source, destination):
    saved=json.loads((source/'goal.json').read_text())
    handoff=json.loads((source/'handoff.json').read_text())
    frames=[json.loads(line) for line in (source/'trial_01.jsonl').read_text().splitlines()]
    actions=json.loads((source/'attacker_actions.json').read_text())
    measured=json.loads((source/'latest_metrics.json').read_text())
    track_ids={aid:i for i,aid in enumerate(handoff['initial_states'])}
    trajectories={aid:[dict(step=round(row['time_s']/.1),position_world=[row['states'][aid]['x'],row['states'][aid]['y']],
        yaw_world=row['states'][aid]['yaw'],speed_mps=row['states'][aid]['speed']) for row in frames]
        for aid in track_ids}
    cases=[]
    for participant in saved['goal']['participants']:
        aid=participant['actor_id'];alias=next(k for k,v in handoff['aliases'].items() if v==aid)
        case_id=f"scene_{saved['scene_index']:03d}_{alias}"
        waypoints=[]
        for index,action in enumerate(actions):
            plan=(action.get('plans_world') or {}).get(aid)
            if not plan:continue
            for point in plan['points'][1:]:
                step=round((action['time_s']+point['t_s'])/.1)
                submitted_end=action['time_s']+action.get('duration_s',frames[-1]['time_s']-action['time_s'])
                if step<=min(trajectories[aid][-1]['step'],round(submitted_end/.1)):
                    waypoints.append(dict(step=step,stage_id=index,position_world=point['position_world_m']))
        events=[]
        for event in measured.get('events',[]):
            if event['actor_id']!=aid and event['other_id']!=aid:continue
            # The original extraction prompt defines these as EGO/ADV contact
            # events. Do not relabel a Stage 1 background collision as EGO contact.
            if {event['actor_id'],event['other_id']}!={aid,'ego'}:continue
            step=min(round(frames[-1]['time_s']/.1),max(1,math.ceil(event['time_s']/.1)))
            state=next(p for p in trajectories[aid] if p['step']==step)
            events.append(dict(step=step,position_world=state['position_world'],
                reason='measured_stage1_contact',other_id=event['other_id'],time_s=event['time_s']))
        case=dict(schema_version='stage2_case_v1',case_id=case_id,scene_id=source.name,
            scene_index=saved['scene_index'],ctrl_track_id=track_ids[aid],ego_track_id=track_ids['ego'],
            actor_id=aid,actor_alias=alias,participant_role=participant['role'],
            scenario_goal=saved.get('display_goal',saved['goal']['language_goal']),
            step_dt_s=.1,stage_plan=[dict(stage_id=i,description=a.get('reason') or '；'.join(
                (d.get('llm_output') or {}).get('reason','') for d in a.get('legacy_decisions',[])
                if d.get('ctrl_track_id')==track_ids[aid]) or "Submitted closed-loop trajectory intervals") for i,a in enumerate(actions)],
            executed_waypoint_plan=waypoints,collision_events=events,
            adv_rollout_traj=trajectories[aid],ego_rollout_traj=trajectories['ego'],
            source_stage1=handoff.get('source_stage1_web_path',str(source)),goal_sha256=saved['goal_sha256'])
        path=destination/(case_id+'.json');write_json(path,case);cases.append((path,case))
    write_json(destination/'index.json',dict(schema_version='stage2_case_v1',cases=[str(p) for p,_ in cases],
        actor_track_ids=track_ids,source_stage1=str(source)))
    return saved,handoff,cases,track_ids


def extract(cases, config, root):
    client=QwenClient(config,root/'anchor_extractor_calls.jsonl')
    for case_path,case in cases:
        anchor_path=case_path.with_name(case_path.stem+'.adv_anchor_plan.json')
        dense_options=config.get('dense_anchor_extraction')
        sparse_options=config.get('sparse_anchor_extraction')
        if dense_options and sparse_options:
            raise ValueError('Choose either dense or sparse anchor extraction')
        selection_validator=None
        if sparse_options:
            from tbsim.scene_agent.sparse_anchors import sparse_anchor_prompts,validate_sparse_anchors
            prompts=sparse_anchor_prompts(case,sparse_options)
            selection_validator=validate_sparse_anchors
        elif dense_options:
            from tbsim.scene_agent.dense_anchors import dense_anchor_prompts,validate_dense_anchors
            prompts=dense_anchor_prompts(case,dense_options)
            selection_validator=validate_dense_anchors
        else:
            prompts=_build_prompts(case)
        if anchor_path.exists():
            stored=json.loads(anchor_path.read_text())
            _validate_anchor_response(case,stored)
            if selection_validator:
                checked=selection_validator(case,dict(anchor_extraction_reason=stored['anchor_extraction_reason'],
                    selected_steps=[p['step'] for p in stored['adv_anchor_plan']]),prompts['extraction_contract'])
                if [p['position_world'] for p in checked['adv_anchor_plan']]!=[p['position_world'] for p in stored['adv_anchor_plan']]:
                    raise ValueError('Saved selected anchor coordinates no longer match measured frames')
            continue
        write_json(case_path.with_name(case_path.stem+'.anchor_prompt.json'),prompts)
        messages=[dict(role='system',content=prompts['system_prompt']),dict(role='user',content=prompts['user_prompt'])]
        for attempt in range(3 if selection_validator else 2):
            response=''
            try:
                response=client.chat(messages)['content']
                case_path.with_name(case_path.stem+f'.anchor_attempt_{attempt+1}.raw.txt').write_text(response)
                parsed=_extract_json_object(response)
                if parsed is None:raise ValueError('Anchor extractor returned no JSON object')
                normalized=(selection_validator(case,parsed,prompts['extraction_contract'])
                            if selection_validator else _validate_anchor_response(case,parsed))
                if not normalized['adv_anchor_plan']:raise ValueError('Anchor normalization removed every point')
                write_json(anchor_path,normalized)
                print('ANCHORS',case['actor_alias'],len(normalized['adv_anchor_plan']),flush=True)
                break
            except (ValueError,RuntimeError) as exc:
                if attempt==(2 if selection_validator else 1):raise
                print('ANCHOR_RETRY',case['actor_alias'],str(exc),flush=True)
                if response:messages.append(dict(role='assistant',content=response))
                messages.append(dict(role='user',content="Fix output: "+str(exc)+". Still select according to the original measured frames and contract; do not fabricate coordinates."))
    return client.requests


def save_scene(root, ep, result, records):
    export_episode(ep, result)
    write_json(root/'diffusion_plans.json', records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-batch',required=True,type=Path)
    parser.add_argument('--scene',required=True,type=int)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/scene_agent/two_stage_reviewer_dense50_joint.json')
    parser.add_argument('--extractor-config',type=Path,default=ROOT/'configs/scene_agent/openai_extractor.json')
    parser.add_argument('--reviewer-config',type=Path,default=ROOT/'configs/scene_agent/openai_reviewer.json')
    parser.add_argument('--anchor-source-scene',type=Path,help='Reuse validated anchors from an identical saved Stage 2 case')
    parser.add_argument('--defer-review',action='store_true',help='Save and publish the simulation for offline metrics without making a final LLM review call')
    parser.add_argument('--initial-anchor-scene',type=Path,help='Use initial-state-only predicted cases; do not read Stage 1 future artifacts')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    source=args.source_batch.resolve()/f'execute_{args.scene:03d}/scene_{args.scene:03d}'
    options=json.loads(args.config.read_text())
    extractor_config=json.loads(args.extractor_config.read_text());extractor_config['temperature']=.1
    if args.initial_anchor_scene:
        if args.anchor_source_scene:raise ValueError('Initial and measured anchors are mutually exclusive')
        saved=json.loads((source/'goal.json').read_text());handoff=json.loads((source/'handoff.json').read_text())
        track_ids={aid:i for i,aid in enumerate(handoff['initial_states'])};cases=[]
        prepared=args.initial_anchor_scene/'stage2_cases'
        for old in sorted(prepared.glob('scene_*.json')):
            if old.name.count('.')!=1:continue
            case=json.loads(old.read_text())
            assert case['reference_kind']=='initial_prediction_not_executed'
            assert case['goal_sha256']==saved['goal_sha256'] and case['scene_index']==args.scene
            assert case['ctrl_track_id']==track_ids[case['actor_id']]
            p=root/'stage2_cases'/old.name;write_json(p,case);cases.append((p,case))
            for suffix in ['.adv_anchor_plan.json','.anchor_prompt.json']:
                shutil.copyfile(old.with_name(old.stem+suffix),p.with_name(p.stem+suffix))
            _validate_anchor_response(case,json.loads(p.with_name(p.stem+'.adv_anchor_plan.json').read_text()))
        assert {case['actor_id'] for _,case in cases}=={p['actor_id'] for p in saved['goal']['participants']}
        write_json(root/'initial_anchor_provenance.json',dict(source=str(args.initial_anchor_scene),reads_stage1_future=False,reference_kind='initial_prediction_not_executed'))
        (root/'anchor_extractor_calls.jsonl').touch()
    else:
        saved,handoff,cases,track_ids=build_cases(source,root/'stage2_cases')
    if args.anchor_source_scene:
        hashes={}
        for p,case in cases:
            old=args.anchor_source_scene.resolve()/'stage2_cases'/p.name
            if json.loads(old.read_text())!=case:
                raise ValueError('Cannot reuse anchors from a different Stage 1 case: '+str(old))
            for suffix in ['.adv_anchor_plan.json','.anchor_prompt.json']:
                src=old.with_name(old.stem+suffix)
                dst=p.with_name(p.stem+suffix)
                shutil.copyfile(str(src),str(dst))
                hashes[dst.name]=hashlib.sha256(dst.read_bytes()).hexdigest()
        write_json(root/'reused_anchors.json',dict(source=str(args.anchor_source_scene.resolve()),sha256=hashes))
        (root/'anchor_extractor_calls.jsonl').touch(exist_ok=True)
    start=time.perf_counter();ep=None
    write_json(root/'source_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(),ROOT/'tbsim/scene_agent/two_stage_backend.py',ROOT/'tbsim/policies/stage2_diffusion.py',
        ROOT/'scripts/run_stage2_anchor_extractor.py',ROOT/'tbsim/models/diffusion.py',ROOT/'tbsim/utils/guidance_utils.py',
        ROOT/'tbsim/utils/control_projection.py',ROOT/'tbsim/utils/control_projection_triton.py',
        ROOT/'tbsim/models/RasterizedDiffusionModel.py',ROOT/'tbsim/scene_agent/backend.py',
        ROOT/'tbsim/scene_agent/anchor_backend.py',ROOT/'tbsim/scene_agent/initialization.py',ROOT/'tbsim/scene_agent/adaptive_anchors.py',
        ROOT/'tbsim/scene_agent/execution_agent.py',ROOT/'tbsim/scene_agent/review_evidence.py',
        ROOT/'tbsim/scene_agent/stage2_review.py',ROOT/'tbsim/scene_agent/guidance_weights.py',
        ROOT/'tbsim/scene_agent/dense_anchors.py',ROOT/'tbsim/scene_agent/sparse_anchors.py',ROOT/'tbsim/scene_agent/evaluation_protocol.py',
        ROOT/'tbsim/scene_agent/road_evidence.py',ROOT/'tbsim/utils/road_body_guidance.py',
        ROOT/'tbsim/utils/joint_candidate_selection.py',ROOT/'tbsim/algos/algos.py',
        ROOT/'tbsim/utils/body_gap_guidance.py']})
    write_json(root/'config.json',options)
    try:
        config=json.loads(args.extractor_config.read_text())
        config.update(dt=.1,max_simulation_s=options['max_simulation_s'],ego_replan_interval_s=options['ego_replan_interval_s'],
            ego_plan_horizon_s=3.2,two_stage=options,diffusion_device=args.device,
            ego_model=options.get('ego_model',handoff.get('ego_model','idm')),
            lightweight_native_steps=options.get('lightweight_native_steps',False),
            compact_final_review=options.get('compact_final_review',False),
            baseline_existing_actor_ids=[aid for aid,state in handoff['initial_states'].items() if state['source']=='existing'])
        stage1_end_steps={case['ego_rollout_traj'][-1]['step'] for _,case in cases}
        assert len(stage1_end_steps)==1, 'Stage 1 cases must share an executed end frame'
        config['stage1_end_step']=stage1_end_steps.pop()
        world=NuscTwoStageWorld(args.scene,config,str(ROOT/'nuscenes'),handoff['start_frame'])
        view=GoalTools(world,None,str(ROOT/'nuscenes'),root)
        goal_source=Path(handoff['source'])
        if not goal_source.is_absolute():goal_source=ROOT/goal_source
        goal=restore_goal(goal_source,view)
        assert view.alias_actors==handoff['aliases'], 'Stage 2 aliases differ from selected Stage 1'
        roundoff=restore_initial_roundoff(world,handoff['initial_states'])
        assert world.states==handoff['initial_states'], 'Stage 2 initialization differs from selected Stage 1'
        write_json(root/'initialization_check.json',dict(all_initial_states_equal=True,aliases_equal=True,
            goal_sha256_equal=goal['goal_sha256']==saved['goal_sha256'],existing_actor_ids=config['baseline_existing_actor_ids'],
            cross_runtime_roundoff_restored=roundoff))
        assert goal['goal_sha256']==saved['goal_sha256']
        if not args.initial_anchor_scene:extract(cases,extractor_config,root)
        world.begin_two_stage(goal['goal'],root/'stage2_cases',root,track_ids)
        write_json(root/'resolved_guidance.json',world.resolved_guidance)
        write_json(root/'handoff.json',dict(handoff,source_stage1_execution=str(source),
            execution_mode='original_offline_stage2',diffusion_enabled=True,same_initial_layout=True,
            ego_model=config['ego_model'],background_mode='diffusion',sampling_contract=world.sampling_contract,
            compatibility_adaptations=['native ReAct artifact format and spawned vehicle history',
                'multiple attacker-EGO collision pairs','helpers anchored with BG avoidance, no collision attraction'],
            preserved_simulation_settings=dict(ego_replan_interval_s=options['ego_replan_interval_s'],max_simulation_s=options['max_simulation_s'])))
        if options.get('post_anchor_collision_factor') is not None:
            write_json(root/'temporal_guidance_method.json',dict(factor=options['post_anchor_collision_factor'],
                decay_duration_s=options.get('post_anchor_collision_decay_s'),
                start_source=options.get('post_anchor_collision_start','final_anchor'),
                stage1_end_time_s=config['stage1_end_step']*.1,
                collision_decay_start_time_s_by_actor={a:step*.1 for a,step in world.collision_decay_start_steps.items()},
                activation='predicted step >= cutoff step' if options.get('post_anchor_collision_decay_s') else 'predicted step > cutoff step',
                distance_term='separate minima for before/after cutoff, excluding zero-weight times',
                sample_filter='original causecollision-only filter, with the same temporal loss',
                diffusion_replan_steps=options['diffusion_replan_steps']))
        reviewer=QwenClient(json.loads(args.reviewer_config.read_text()),root/'reviewer_calls.jsonl')
        toolkit=ExecutionTools(world,view,goal,reviewer,root);ep=toolkit.episode
        static_plans={}
        for p,case in cases:
            anchors=json.loads(p.with_name(p.stem+'.adv_anchor_plan.json').read_text())['adv_anchor_plan']
            static_plans[case['actor_id']]=dict(created_at_s=0.,points=[dict(t_s=a['step']*.1,
                position_world_m=a['position_world']) for a in anchors])
        stop_reason='time_limit'
        while world.time<options['max_simulation_s']-1e-8 and not ep.stopped:
            print('STAGE2_SAMPLE',world.time,flush=True)
            record=world.prepare_stage2()
            ep.action_history.append(dict(time_s=world.time,duration_s=record['execution_interval_s'],
                plans_world=static_plans,reason="Offline fixed anchor, original Stage2DiffusionPolicy guidance",diffusion=record))
            for _ in range(min(options['diffusion_replan_steps'],round((options['max_simulation_s']-world.time)/.1))):
                try:before,after,controls=world.step()
                except ForbiddenRearEnd as exc:
                    ep.stopped=True;stop_reason='forbidden_rear_end';write_json(root/'rejected_contact.json',dict(reason=str(exc)));break
                contact=ep.monitor.update(before,after,controls,world.time,world.on_road)
                ep.trace.append(dict(time_s=world.time,states=after,controls=controls))
                if contact:ep.stopped=True;stop_reason='contact';break
            ep._save_trace()
            write_json(root/'diffusion_plans.json',world.diffusion_records)
            print('STAGE2_DONE',world.time,'sampling_s',record['sampling_wall_s'],flush=True)
        result=dict(scene_index=args.scene,status='awaiting_review',
            simulation_time_s=world.time,execution_mode='original_offline_stage2',diffusion_enabled=True,
            stop_reason=stop_reason,rollback_used=False,source_stage1=str(source),
            goal_sha256=saved['goal_sha256'],wall_time_s=time.perf_counter()-start,
            sampling_wall_total_s=sum(r['sampling_wall_s'] for r in world.diffusion_records),
            sampling_rounds=len(world.diffusion_records),sampling_contract=world.sampling_contract,
            ego_model=config['ego_model'],background_mode='diffusion',
            control_projection_enabled=world.projection_enabled)
        if args.initial_anchor_scene:result['reference_kind']='initial_prediction_not_executed'
        write_json(root/'anchor_plans.json',[dict(time_s=0.,plans=static_plans)])
        from tbsim.scene_agent.stage2_review import save_simulation_checkpoint,required_contact_review,make_review_request
        validation=None
        if options.get('evaluation_protocol')=='reviewer_v1_20261006':
            from tbsim.scene_agent.evaluation_protocol import motion_audit,trial_road_gate,interaction_category,designated_contact_origin
            from tbsim.scene_agent.road_evidence import trial_road_evidence
            from tbsim.utils.control_projection import DEFAULT_LIMITS
            road=trial_road_evidence(world,dict(start_s=0.,end_s=world.time,trace=ep.trace,
                                               plans={a:[] for a in world.anchor_participants}))
            validation=dict(protocol_version=options['evaluation_protocol'],
                motion=motion_audit(ep.trace,world.anchor_participants,dict(DEFAULT_LIMITS,**options.get('control_projection_limits',{}))),
                road=trial_road_gate(road),road_measurements=road,
                initial_interaction_category=interaction_category(goal['goal'],ep.trace[0]['states']))
            validation['passed']=validation['motion']['passed'] and validation['road']['passed']
            validation['contact_origin']=designated_contact_origin(ep.trace,world.attackers)
            if validation['contact_origin']['initial_contact_actors'] and not validation['contact_origin']['has_new_designated_contact']:
                validation['passed']=False
            world.execution_validation=validation
            write_json(root/'execution_validation.json',validation)
        evidence=dict(descriptive_measurements=ep.monitor.report(),**toolkit.review_motion_evidence())
        geometry=dict(name=world.name,dt=world.dt,lanes=[dict(center=lane.center.xy.tolist(),
            left_edge=lane.left_edge.xy.tolist() if lane.left_edge is not None else None,
            right_edge=lane.right_edge.xy.tolist() if lane.right_edge is not None else None)
            for lane in world.lanes.values()])
        save_simulation_checkpoint(root,result,evidence,
            make_review_request(saved,ep.trace,evidence,options.get('compact_final_review',False)),geometry)
        if args.defer_review:
            result.update(status='pending_review',wall_time_s=time.perf_counter()-start,
                          review=dict(accepted=None,review_source='deferred',
                                      reason='Simulation complete; final semantic review has not been performed.',
                                      descriptive_measurements=ep.monitor.report()))
            (root/'reviewer_calls.jsonl').touch(exist_ok=True)
            save_scene(root,ep,result,world.diffusion_records)
            write_json(root/'result.json',result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            return 0
        review=None
        if validation is not None and not validation['passed']:
            review=dict(ok=True,accepted=False,review_source='measured_motion_or_road_gate',semantic_review_called=False,
                reason="The actual trajectory did not pass the pre-fixed motion or complete vehicle body road checks.",execution_validation=validation,
                descriptive_measurements=ep.monitor.report(),goal_sha256=saved['goal_sha256'],trace_length=len(ep.trace))
            (root/'reviewer_calls.jsonl').touch(exist_ok=True)
        if review is None and options.get('require_contact_before_review',False):
            review=required_contact_review(ep.monitor.report(),world.attackers,saved['goal_sha256'],len(ep.trace))
            if review is not None:(root/'reviewer_calls.jsonl').touch(exist_ok=True)
        if review is None:
            for review_attempt in range(2):
                try:
                    review=ep.evaluate()
                    break
                except ValueError as exc:
                    if review_attempt or not str(exc).startswith('Qwen format error:'):raise
                    # Retry malformed JSON only; a valid negative verdict is final.
                    write_json(root/'reviewer_format_retry.json',dict(error=str(exc),simulation_rerun=False))
        result.update(status='achieved' if review['accepted'] else 'not_achieved',review=review,
                      wall_time_s=time.perf_counter()-start)
        save_scene(root,ep,result,world.diffusion_records)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result=dict(scene_index=args.scene,status='error',error=str(exc),wall_time_s=time.perf_counter()-start)
    write_json(root/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 2 if result['status']=='error' else 0


if __name__=='__main__':raise SystemExit(main())
