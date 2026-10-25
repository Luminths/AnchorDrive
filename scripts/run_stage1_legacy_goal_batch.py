#!/usr/bin/env python3
"""Stage 1 planner and Assessor execution."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SCENES = ','.join(map(str, range(150)))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def legacy_config(policy_mode='posttrial', overrides=None):
    config = json.loads((ROOT/'configs/scene_agent/openai_stage1.json').read_text())
    if overrides:
        extra=json.loads(Path(overrides).read_text()) if not isinstance(overrides,dict) else overrides
        config.update(extra)
    return config


def run_scene(source, output, scene, dry_run=False, policy_mode='posttrial', overrides=None):
    from tbsim.scene_agent.client import QwenClient
    from tbsim.scene_agent.simple_goal_agent import GoalTools
    from tbsim.scene_agent.execution_agent import restore_goal, write_log
    from tbsim.scene_agent.anchor_execution_agent import AnchorExecutionTools
    from tbsim.scene_agent.legacy_waypoint_execution import (
        NuscLegacyWaypointWorld, LegacySavedGoalPolicy, LegacyChatBridge)
    from tbsim.scene_agent.backend import ForbiddenRearEnd
    from tbsim.scene_agent.export import export_episode

    root = output/('scene_%03d' % scene)
    root.mkdir(parents=True, exist_ok=False)
    config = legacy_config(policy_mode,overrides)
    lesson=config.pop('prior_attempt_lessons_by_scene',{}).get(str(scene))
    if lesson:config['prior_attempt_lessons']=lesson
    reviewer_path = Path(config.get('reviewer_config', ROOT/'configs/scene_agent/openai_reviewer.json'))
    if not reviewer_path.is_absolute():
        reviewer_path = ROOT/reviewer_path
    reviewer_config = json.loads(reviewer_path.read_text())
    write_json(root/'config.json', config)
    started = time.monotonic()
    toolkit = None
    rows = []
    result = dict(scene_index=scene, status='error', execution_mode=config['execution_mode'])
    try:
        baseline=None
        if config.get('baseline_batch'):
            baseline_dir=Path(config['baseline_batch'])/('execute_%03d'%scene)/('scene_%03d'%scene)
            if json.loads((baseline_dir/'goal.json').read_text())!=json.loads((source/'goal.json').read_text()):
                raise ValueError('Saved goal differs from requested baseline')
            baseline=json.loads((baseline_dir/'handoff.json').read_text())['initial_states']
            config['baseline_existing_actor_ids']=[aid for aid,state in baseline.items() if state['source']=='existing']
            write_json(root/'config.json',config)
        revised_initial=None
        if config.get('goal_revision_baseline_batch'):
            if baseline is not None:
                raise ValueError('Choose strict old-goal baseline or explicit goal revision baseline')
            from tbsim.scene_agent.goal_regeneration import existing_states
            baseline_dir=Path(config['goal_revision_baseline_batch'])/('execute_%03d'%scene)/('scene_%03d'%scene)
            baseline=json.loads((baseline_dir/'handoff.json').read_text())['initial_states']
            revised_initial=json.loads((source/'goal_initial_states.json').read_text())
            if existing_states(revised_initial)!=existing_states(baseline):
                raise ValueError('Revised goal changed original existing initial states; API not called')
            config['baseline_existing_actor_ids']=list(existing_states(baseline))
            write_json(root/'config.json',config)
        world = NuscLegacyWaypointWorld(scene, config, str(ROOT/'nuscenes'), 11)
        client = QwenClient(config, root/'llm_calls.jsonl')
        reviewer = QwenClient(reviewer_config, root/'reviewer_calls.jsonl')
        view = GoalTools(world, client, str(ROOT/'nuscenes'), root)
        saved = restore_goal(source, view)
        if baseline is not None:
            expected=revised_initial if revised_initial is not None else baseline
            if world.states!=expected:
                raise ValueError('Restored complete initial state differs from baseline; API not called')
            result['baseline_initial_state_checked']=True
            if revised_initial is not None:result['goal_revision_initial_state_checked']=True
        toolkit_class=AnchorExecutionTools
        policy_class=LegacySavedGoalPolicy
        if policy_mode in ('optimized','posttrial'):
            from tbsim.scene_agent.compact_waypoint_execution import CompactWaypointPolicy,CompactLegacyExecutionTools
            toolkit_class=CompactLegacyExecutionTools;policy_class=CompactWaypointPolicy
            world.trajectory_method='compact_relative_waypoint_planner_assessor'
        if policy_mode=='posttrial':
            from tbsim.scene_agent.trial_waypoint_execution import PostTrialWaypointPolicy
            policy_class=PostTrialWaypointPolicy
            world.trajectory_method='native_trial_waypoint_assessor'
        toolkit = toolkit_class(world, view, saved, reviewer, root)
        bridge = LegacyChatBridge(client, root)
        if config['background_mode'] != 'recorded':
            raise ValueError('Stage 1 requires recorded background trajectories')
        background_policy = None
        policy = policy_class(config, saved, root, bridge, background_policy)
        if policy_mode in ('optimized','posttrial'):policy.bind_scene(world,view,toolkit.episode)
        write_json(root/'legacy_policy_config.json', policy.eval_config.to_dict())
        toolkit.episode.write('handoff.json', dict(source=str(source),
            goal_sha256=saved['goal_sha256'], start_frame=11,
            initial_states=world.states, aliases=view.alias_actors,
            execution_model=config['model'], reviewer_model=reviewer_config['model'],
            planner_model=config.get('planner_model',config['model']),
            planner_near_model=config.get('planner_near_model'),
            planner_near_distance_m=config.get('planner_near_distance_m'),
            assessor_model=config['model'],planner_send_bev=config.get('planner_send_bev',True),
            execution_mode=config['execution_mode'], background_mode=config['background_mode'],
            ego_model=config.get('ego_model','idm'), diffusion_enabled=False))
        observation = toolkit.observe()
        observation.pop('image_path', None)
        rows.append(dict(step=0, tool='observe', arguments={}, assistant='', observation=observation))
        write_log(root, rows)
        if dry_run:
            # Exercise the full old get_action loop without spending API tokens.
            def manual(step_index):
                names = list(world._safesim_env.current_agent_names)
                actor = world.states[next(a for a in names if a in world.anchor_participants)]
                if policy_mode in ('optimized','posttrial'):
                    from tbsim.scene_agent.compact_waypoint_execution import ego_xy
                    import math
                    points=[]
                    for t in (1.,2.,3.):
                        xy=ego_xy([actor['x']+actor['speed']*t*math.cos(actor['yaw']),
                                   actor['y']+actor['speed']*t*math.sin(actor['yaw'])],world.states['ego'])
                        points.append(dict(t_s=t,x_m=xy[0],y_m=xy[1]))
                    return json.dumps(dict(reason="Offline verification continuous forward driving",waypoints=points))
                return json.dumps(dict(stage_plan=[dict(stage_id=1, stage_name="Proceed along the current direction",
                    goal_description="Offline interface verification", completion_condition="Complete current window")],
                    waypoint_plan=[dict(step=k, stage_id=1, position_world=[
                        actor['x']+actor['speed']*.1*k*__import__('math').cos(actor['yaw']),
                        actor['y']+actor['speed']*.1*k*__import__('math').sin(actor['yaw'])])
                        for k in (5,10,15,20,25,30)]))
            policy._load_manual_response_text = manual
            def offline_assessor(**kw):
                system,user=policy._build_plan_assessor_prompt(kw['planner_context'],kw['rollout_summary'])
                return dict(status='offline_mock_assessor',passed=True,system_prompt=system,user_prompt=user)
            policy._query_plan_assessor = offline_assessor
        window = 0
        while not toolkit.episode.stopped:
            window += 1
            before_time = world.time
            if policy_mode=='posttrial':
                from tbsim.scene_agent.trial_waypoint_execution import run_post_trial_window
                count,decisions,rejection=run_post_trial_window(toolkit,policy,dry_run)
            else:
                count, decisions = world.prepare_legacy(policy)
            if not decisions:
                raise ValueError('Legacy policy produced no participant decisions')
            if window == 1:
                first = bridge.exchanges[0] if policy_mode=='posttrial' and bridge.exchanges else decisions[0]
                write_json(root/'initial_request.json', dict(messages=[
                    dict(role='system', content=first['system_prompt']),
                    dict(role='user', content=first['user_prompt'])], tools=[]))
            if policy_mode!='posttrial':
                plans = {}
                for decision in decisions:
                    aid = policy.actor_names[decision['ctrl_agent_index']]
                    meta = decision.get('parse_meta') or {}
                    points = [dict(t_s=0., position_world_m=[world.states[aid]['x'],world.states[aid]['y']])]
                    points += [dict(t_s=p['step']*.1, position_world_m=p['position_world'])
                               for p in meta.get('waypoint_plan') or []]
                    plans[aid] = dict(created_at_s=before_time, points=points)
                toolkit.episode.action_history.append(dict(time_s=before_time, duration_s=count*.1,
                    actions=[], plans_world=plans, legacy_decisions=decisions,
                    trajectory_method=world.trajectory_method, reason="Legacy planner/assessor window execution"))
                rejection = None
                for _ in range(count):
                    try:
                        before, after, controls = world.step()
                    except ForbiddenRearEnd as exc:
                        rejection = str(exc)
                        toolkit.episode.stopped = True
                        break
                    contact = toolkit.episode.monitor.update(before, after, controls, world.time, world.on_road)
                    toolkit.episode.trace.append(dict(time_s=world.time, states=after, controls=controls))
                    if contact:
                        toolkit.episode.stopped = True
                        break
            if world.time >= config['max_simulation_s']-1e-8:
                toolkit.episode.stopped = True
            if world.time <= before_time and not rejection:
                raise ValueError('No native simulation progress')
            toolkit.episode._save_trace()
            observation = toolkit.observe()
            observation.pop('image_path', None)
            observation['legacy_decisions'] = decisions
            observation['local_rear_end_rejection'] = rejection
            rows.append(dict(step=window, tool='legacy_plan_assess_execute',
                arguments=dict(planning_horizon_s=3., execution_interval_s=config['decision_interval_s']),
                assistant="Legacy structured waypoint planning with independent review; see the full call record for this round.",
                observation=observation))
            write_log(root, rows)
            if dry_run:
                break
        result.update(status='not_achieved', simulation_time_s=world.time,
            api_requests=client.requests, reported_tokens=client.tokens, execution_error=None,
            rollback_used=False, rollback_count=0, windows=window, diffusion_enabled=False,
            dry_run=dry_run, background_mode=config['background_mode'], ego_model=config.get('ego_model','idm'),
            stop_reason=('execution_safety_failed_budget_exhausted' if rejection=='execution_safety_failed_budget_exhausted'
                         else 'direct_rear_end_rejection' if rejection else 'contact_or_horizon'))
        if policy_mode=='posttrial':
            result.update(trial_attempts_total=sum(len(a.get('trial_attempts',[])) for a in toolkit.episode.action_history),
                trial_restore_count=sum(max(0,len(a.get('trial_attempts',[]))-1) for a in toolkit.episode.action_history),
                posttrial_assessor_max_replans=config['posttrial_assessor_max_replans'])
        if not dry_run and len(toolkit.episode.trace)>1:
            review = toolkit.episode.evaluate()
            result.update(status='achieved' if review['accepted'] else 'not_achieved', review=review,
                reviewer_requests=reviewer.requests, reviewer_reported_tokens=reviewer.tokens)
        elif dry_run:
            result['status'] = 'offline_validated'
        else:
            result.update(review=dict(accepted=False,review_source='execution_safety_no_committed_motion'),
                          reviewer_requests=0,reviewer_reported_tokens=0)
    except Exception as exc:
        result.update(status='error', execution_error=str(exc), simulation_time_s=toolkit.world.time if toolkit else 0.)
        import traceback
        (root/'execution_error.txt').write_text(traceback.format_exc())
    # Include incurred usage even when execution or final review stopped early.
    if 'client' in locals():
        result.update(api_requests=client.requests,reported_tokens=client.tokens,
                      api_http_requests=client.http_requests)
    if 'reviewer' in locals():
        result.update(reviewer_requests=reviewer.requests,reviewer_reported_tokens=reviewer.tokens,
                      reviewer_http_requests=reviewer.http_requests)
    result['wall_s'] = round(time.monotonic()-started, 3)
    if toolkit:
        export_episode(toolkit.episode, result)
        write_log(root, rows)
    write_json(root/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result
