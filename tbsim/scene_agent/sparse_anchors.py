"""Bounded LLM selection of actual Stage 1 frames, without invented positions."""
import json
import math


def sparse_anchor_prompts(case, options):
    frames = [p for p in case['adv_rollout_traj'] if p['step'] > 0]
    if not frames:
        raise ValueError('No measured future frames to select')
    dt = case['step_dt_s']
    cap = int(options.get('max_anchors', 16))
    spacing = float(options.get('nominal_spacing_s', .4))
    max_gap = float(options.get('max_gap_s', 1.))
    slack = int(options.get('count_slack', 2))
    if cap < 2 or spacing <= 0 or max_gap < dt or slack < 0:
        raise ValueError('Invalid sparse anchor budget or temporal coverage')
    measured_steps = {p['step'] for p in frames}
    mandatory = {frames[0]['step'], frames[-1]['step']}
    mandatory.update(e['step'] for e in case.get('collision_events', [])
                     if e['step'] in measured_steps)
    if len(mandatory) > cap:
        raise ValueError('Contact frames exceed the sparse anchor budget')
    span_s = (frames[-1]['step'] - frames[0]['step']) * dt
    target = min(len(frames), cap, max(len(mandatory), math.ceil(span_s / spacing) + 1))
    minimum = max(len(mandatory), target - slack)
    # Short traces have smaller budgets. The maximum is never above 16 by default.
    contract = dict(target_count=target, min_anchors=minimum, max_anchors=target,
                    total_anchor_cap=cap, nominal_spacing_s=spacing,
                    max_gap_steps=int(math.floor(max_gap / dt + 1e-8)),
                    mandatory_steps=sorted(mandatory))
    payload = dict(goal=case['scenario_goal'], role=case.get('participant_role'),
                   dt_s=dt, contract=contract,
                   measured_frames=[dict(step=p['step'], xy=p['position_world'],
                       yaw=p.get('yaw_world'), speed=p.get('speed_mps')) for p in frames])
    system = ("You are a trajectory key anchor extractor. Select a small number of key frames from the executed trajectory; only select from the given steps, do not create coordinates, and do not change the target. Output only JSON: anchor_extraction_reason in brief English, selected_steps as a strictly increasing integer array.")
    user = ("Select min_anchors to max_anchors points for the entire trajectory, as close to target_count as possible. Retain the first and last points in mandatory_steps and the real contact frame; prioritize retaining key positions at the end of waiting, starting, turning, entering the interaction area, and before and after contact. Select fewer points for smooth straight segments; do not output frame by frame. The step difference between adjacent points must not exceed max_gap_steps, so as to cover the entire motion. Coordinates are taken by the program from the corresponding measured positions.\n" +
            json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    return dict(system_prompt=system, user_prompt=user, extraction_contract=contract)


def validate_sparse_anchors(case, payload, contract):
    reason = payload.get('anchor_extraction_reason')
    selected = payload.get('selected_steps')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Provide a short anchor_extraction_reason')
    if not isinstance(selected, list) or not selected or any(type(s) is not int for s in selected):
        raise ValueError('selected_steps must contain measured integer steps')
    if any(b <= a for a, b in zip(selected, selected[1:])):
        raise ValueError('selected_steps must be unique and increasing')
    frames = {p['step']: p for p in case['adv_rollout_traj'] if p['step'] > 0}
    if set(selected) - set(frames):
        raise ValueError('Select only measured future frames')
    if not contract['min_anchors'] <= len(selected) <= contract['max_anchors']:
        raise ValueError('Expected %d..%d anchors for the whole trace, got %d' %
                         (contract['min_anchors'], contract['max_anchors'], len(selected)))
    if set(contract['mandatory_steps']) - set(selected):
        raise ValueError('Missing mandatory first, last, or measured contact frame')
    if any(b - a > contract['max_gap_steps'] for a, b in zip(selected, selected[1:])):
        raise ValueError('Selected frames leave a gap exceeding the coverage limit')
    return dict(anchor_extraction_reason=reason.strip(),
                adv_anchor_plan=[dict(step=s, position_world=frames[s]['position_world'],
                                     reason='llm_selected_measured_keyframe') for s in selected],
                extraction_method='llm_sparse_measured_frame_selection',
                extraction_contract=contract, measured_frame_count=len(frames),
                selected_frame_fraction=len(selected) / len(frames))
