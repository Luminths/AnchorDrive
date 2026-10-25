"""Review a completed Stage 2 simulation without rerunning diffusion."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/scene_agent/openai_reviewer.json')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file for the independent verdict')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from tbsim.scene_agent.stage2_review import read_simulation_checkpoint, required_contact_review
    from tbsim.scene_agent.client import QwenClient
    from tbsim.scene_agent.roles import is_helper_role
    from tbsim.scene_agent.tools import review_verdict
    saved = read_simulation_checkpoint(args.scene_dir)
    goal = json.loads((args.scene_dir/'goal.json').read_text())
    attackers = [p['actor_id'] for p in goal['goal']['participants'] if not is_helper_role(p['role'])]
    request = saved['review_request']
    if not request:
        raise ValueError('Checkpoint has no complete reviewer evidence')
    request['payload'].pop('execution_validation', None)
    report = saved['evidence']['descriptive_measurements']
    verdict = required_contact_review(report, attackers, goal['goal_sha256'], sum(bool(line.strip()) for line in (args.scene_dir/'trial_01.jsonl').read_text().splitlines()))
    if verdict is None:
        client = QwenClient(json.loads(args.config.read_text()), args.output.with_suffix('.calls.jsonl'))
        class Reviewer:
            def complete(self, unused_system, payload):
                return client.complete(request['system'], payload)
        verdict = review_verdict(Reviewer(), request['payload'])
        verdict['accepted'] = verdict['goal_met']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2)+'\n')
    print(json.dumps(verdict))


if __name__ == '__main__':
    main()
