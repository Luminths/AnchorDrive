#!/usr/bin/env python3
"""Run image-first Stage-1 construction through define_goal only."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tbsim.scene_agent.backend import NuscStage1World
from tbsim.scene_agent.client import QwenClient
from tbsim.scene_agent.simple_goal_agent import GoalTools,SYSTEM,TOOLS,run_goal_agent
from tbsim.scene_agent.goal_map import render_scene


def render_goal_selection(toolkit):
    """Presentation snapshot only; never claim this was an Agent observation."""
    if not toolkit.goal:return
    selected={p['actor_id'] for p in toolkit.goal['participants']}
    render_scene(toolkit.world,toolkit.catalog,toolkit.root/'goal_selection.png',
                 toolkit.actor_aliases.__getitem__,55,selected)








def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',default='0')
    p.add_argument('--output',default=None)
    p.add_argument('--start-frame',type=int,default=11)
    p.add_argument('--max-steps',type=int,default=16)
    p.add_argument('--dataset-path',default=str(ROOT/'nuscenes'))
    p.add_argument('--config',default=str(ROOT/'configs/scene_agent/openai_goal.json'))
    p.add_argument('--render-only',action='store_true',help='Render the initial image without an API call')
    args=p.parse_args()
    scenes=[int(item) for item in args.scenes.split(',')]
    if not scenes or len(set(scenes))!=len(scenes) or any(x<0 or x>=150 for x in scenes):
        p.error('Use distinct scene indices 0..149')
    if not 1<=args.max_steps<=50:p.error('Invalid max steps')
    output=Path(args.output or ROOT/'outputs/stage1_agent'/('goal_only_'+time.strftime('%Y%m%d_%H%M%S'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.config).read_text())
    (output/'config.json').write_text(json.dumps(config,indent=2))
    sources=[ROOT/'tbsim/scene_agent'/name for name in ('simple_goal_agent.py','goal_regeneration.py','construction_cues.py','goal_quality_review.py','initialization.py','critical_conflict_guide.md','goal_map.py','backend.py','client.py','contracts.py')]
    (output/'source_manifest.json').write_text(json.dumps({str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},indent=2))
    results=[]
    for scene_index in scenes:
        root=output/f'scene_{scene_index:03d}';root.mkdir()
        toolkit=None
        try:
            active=copy.deepcopy(config)
            baseline=None
            if active.get('goal_revision_baseline_batch'):
                baseline=json.loads((Path(active['goal_revision_baseline_batch'])/f'execute_{scene_index:03d}/scene_{scene_index:03d}/handoff.json').read_text())['initial_states']
                from tbsim.scene_agent.goal_regeneration import existing_states
                active['baseline_existing_actor_ids']=list(existing_states(baseline))
            world=NuscStage1World(scene_index,active,args.dataset_path,args.start_frame)
            if baseline is not None and world.states!=existing_states(baseline):
                raise ValueError('Goal construction changed original existing actors; API not called')
            client=QwenClient(active,root/'llm_calls.jsonl')
            toolkit=GoalTools(world,client,args.dataset_path,root)
            (root/'original_existing_states.json').write_text(json.dumps(world.states,ensure_ascii=False,indent=2))
            if args.render_only:
                observation=toolkit.view_scene();observation.pop('image_path')
                toolkit.trace.append(dict(step=0,tool='view_scene',observation=observation))
                (root/'agent_trace.json').write_text(json.dumps(toolkit.trace,ensure_ascii=False,indent=2))
                result=dict(status='rendered',display_goal=None,api_requests=0,reported_tokens=0)
            else:
                result=run_goal_agent(toolkit,args.max_steps)
                render_goal_selection(toolkit)
                if toolkit.goal:
                    (root/'goal_initial_states.json').write_text(json.dumps(world.states,ensure_ascii=False,indent=2))
                    if active.get('goal_regeneration'):
                        previous=active.get('goal_failure_context_by_scene',{}).get(str(scene_index),{})
                        (root/'goal_revision.json').write_text(json.dumps(dict(
                            scene_index=scene_index,original_goal_source=previous.get('original_goal_source'),
                            original_goal_sha256=previous.get('original_goal_sha256'),
                            failed_execution_source=previous.get('failed_execution_source'),
                            revised_goal_sha256=json.loads((root/'goal.json').read_text())['goal_sha256'],
                            existing_initial_states_preserved=baseline is not None,
                            generation_mode=('fresh_initial_state' if active.get('goal_from_initial_state') else 'failure_revision'),
                            note=("This batch uniformly rebuilds the goal from the initial state and does not provide past experiment results; the original EGO/BG states are retained, and participating vehicles and generated vehicles may change."
                                  if active.get('goal_from_initial_state') else
                                  "The user authorizes refactoring the failed goal; the original EGO/BG state is retained, and the participating vehicle selection and generated vehicles may change.")),ensure_ascii=False,indent=2))
            result.update(scene_index=scene_index,scene_name=world.name,model=config['model'],enable_thinking=config.get('enable_thinking',False),agent_steps=len(toolkit.trace)-1,
                          simulation_time_s=world.time,output=str(root))
            if world.time!=0:raise RuntimeError('Goal-only run advanced simulation time')
        except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
            result=dict(scene_index=scene_index,status='error',reason=str(exc),output=str(root),
                        simulation_time_s=0,agent_steps=len(toolkit.trace)-1 if toolkit else 0)
        if toolkit and hasattr(toolkit,'goal_quality_client'):
            result.update(goal_quality_requests=toolkit.goal_quality_client.requests,
                goal_quality_tokens=toolkit.goal_quality_client.tokens)
        (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        results.append(result)
        (output/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    print('Output:',output,flush=True)
    return 0 if all(item['status'] in ('goal_defined','rendered') for item in results) else 2


if __name__=='__main__':raise SystemExit(main())
