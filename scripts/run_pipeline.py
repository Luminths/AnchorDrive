"""Run scenario design, closed-loop rollout and diffusion regeneration."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'goal', 'stage1', 'stage2'], default='all')
    parser.add_argument('--scenes', default='0', help='Comma-separated scene indices (0–149)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--goal-config', type=Path, default=ROOT/'configs/scene_agent/openai_goal.json')
    parser.add_argument('--stage1-config', type=Path, default=ROOT/'configs/scene_agent/openai_stage1.json')
    parser.add_argument('--stage2-config', type=Path, default=ROOT/'configs/scene_agent/two_stage_reviewer_dense50_joint.json')
    parser.add_argument('--extractor-config', type=Path, default=ROOT/'configs/scene_agent/openai_extractor.json')
    parser.add_argument('--reviewer-config', type=Path, default=ROOT/'configs/scene_agent/openai_reviewer.json')
    args = parser.parse_args()
    scenes = [int(s) for s in args.scenes.split(',')]
    if len(set(scenes)) != len(scenes) or any(s < 0 or s >= 150 for s in scenes):
        parser.error('Use unique scene indices between 0 and 149')
    output = args.output.resolve()
    configs = {k: getattr(args, k).resolve() for k in ['goal_config','stage1_config','stage2_config','extractor_config','reviewer_config']}
    output.mkdir(parents=True, exist_ok=True)
    def run(module, *options):
        subprocess.run([sys.executable, '-m', module, *map(str, options)], cwd=ROOT, check=True)
    if args.stage in ('all', 'goal'):
        run('scripts.run_stage1_goal_agent', '--scenes', args.scenes, '--output', output/'goals', '--config', configs['goal_config'])
    if args.stage in ('all', 'stage1'):
        from scripts.run_stage1_legacy_goal_batch import run_scene
        config = json.loads(configs['stage1_config'].read_text())
        config['reviewer_config'] = str(configs['reviewer_config'])
        for scene in scenes:
            source = output/'goals'/f'scene_{scene:03d}'
            if not (source/'goal.json').exists():
                raise FileNotFoundError('No generated goal: '+str(source))
            parent = output/'stage1'/f'execute_{scene:03d}'
            parent.mkdir(parents=True, exist_ok=True)
            result = run_scene(source, parent, scene, policy_mode='posttrial', overrides=config)
            if result['status'] == 'error':
                raise RuntimeError('Stage 1 failed; inspect '+str(parent))
    if args.stage in ('all', 'stage2'):
        for scene in scenes:
            target = output/'stage2'/f'execute_{scene:03d}'/f'scene_{scene:03d}'
            if target.exists():
                raise FileExistsError('Choose a new output directory: '+str(target))
            run('scripts.run_stage2_react_scene', '--source-batch', output/'stage1', '--scene', scene,
                '--output', target, '--device', args.device, '--config', configs['stage2_config'],
                '--extractor-config', configs['extractor_config'], '--reviewer-config', configs['reviewer_config'])


if __name__ == '__main__':
    main()
