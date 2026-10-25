"""Preserve completed simulations and skip semantic calls for proven misses."""
import hashlib
import json
import copy
from pathlib import Path


def required_contact_review(report, attackers, goal_sha256, trace_length):
    """Return a measured failure, or None when semantic review is still needed.

    This never accepts a run or constrains an action. Helpers and background
    contacts cannot replace contact by a designated attacker.
    """
    if not isinstance(report.get('events'), list) or 'target_collision' not in report.get('metrics', {}):
        raise ValueError('Missing complete contact measurements')
    contacts=[e for e in report['events'] if e.get('kind')=='collision' and e.get('other_id')=='ego']
    if bool(contacts)!=bool(report['metrics']['target_collision']):
        raise ValueError('Inconsistent measured EGO contact flags')
    if any(e.get('actor_id') in attackers for e in contacts):
        return None
    return dict(ok=True,accepted=False,review_source='required_contact_precheck',
        semantic_review_called=False,
        reason="The designated attacking vehicle did not actually contact EGO, so the necessary condition was not met; the semantic Reviewer was not called. Contact by a background vehicle or only an auxiliary vehicle does not substitute for contact by the attacking vehicle.",
        descriptive_measurements=report,goal_sha256=goal_sha256,trace_length=trace_length)


CHECKPOINT_FILES=('trial_01.jsonl','diffusion_plans.json','anchor_plans.json',
                  'handoff.json','goal.json','source_manifest.json')


def make_review_request(goal, trace, evidence, compact=False):
    """Use the same final-branch facts and sampling as Stage1Tools.evaluate."""
    ids={p['actor_id'] for p in goal['goal']['participants']}|{'ego'}
    frames=trace[::max(1,len(trace)//45)]
    if frames[-1] is not trace[-1]:frames=frames+[trace[-1]]
    samples=[dict(time_s=r['time_s'],actors=[s for aid,s in r['states'].items() if aid in ids],
        applied_controls={aid:c for aid,c in r.get('controls',{}).items() if aid in ids}) for r in frames]
    payload=dict(language_goal=goal['goal']['language_goal'],fixed_participants=goal['goal']['participants'],
        goal_sha256=goal['goal_sha256'],executed_trace=samples,**copy.deepcopy(evidence))
    if compact:
        from .review_evidence import REVIEW_SYSTEM,compact_review_payload
        return dict(system=REVIEW_SYSTEM,payload=compact_review_payload(payload))
    from .execution_agent import EXECUTION_REVIEW_PROMPT
    return dict(system=EXECUTION_REVIEW_PROMPT,payload=payload)


def save_simulation_checkpoint(root, result, evidence, review_request=None, playback_geometry=None):
    """Save before the paid final call; an API failure must not lose the run."""
    root=Path(root)
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CHECKPOINT_FILES}
    value=dict(execution=result,evidence=evidence,files_sha256=hashes,
               review_request=review_request,playback_geometry=playback_geometry,
               scope='Completed simulation only; no semantic acceptance is implied.')
    tmp=root/'simulation_checkpoint.json.tmp'
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(root/'simulation_checkpoint.json')
    return value


def read_simulation_checkpoint(root):
    root=Path(root)
    value=json.loads((root/'simulation_checkpoint.json').read_text())
    if set(value['files_sha256'])!=set(CHECKPOINT_FILES):
        raise ValueError('Incomplete simulation checkpoint')
    for name,expected in value['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Simulation evidence changed: '+name)
    request=value.get('review_request')
    if request and request['payload']['goal_sha256']!=value['execution']['goal_sha256']:
        raise ValueError('Review goal differs from saved execution')
    return value
