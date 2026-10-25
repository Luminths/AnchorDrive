"""LLM selection of measured frames, with an explicit dense coverage contract."""
import json
import math


def dense_anchor_prompts(case, options):
    frames = [p for p in case['adv_rollout_traj'] if p['step'] > 0]
    contract = dict(max_gap_steps=int(options.get('max_gap_steps', 3)),
                    min_density=float(options.get('min_density', .65)),
                    max_anchors=int(options.get('max_anchors', 96)))
    if not frames or len(frames) > contract['max_anchors'] / contract['min_density']:
        raise ValueError('Measured trajectory exceeds the dense extraction budget')
    mandatory = {frames[0]['step'], frames[-1]['step']}
    steps = {p['step'] for p in frames}
    for event in case.get('collision_events', []):
        mandatory.update(s for s in steps if abs(s-event['step']) <= 3)
    contract['mandatory_steps'] = sorted(mandatory)
    payload = dict(goal=case['scenario_goal'], role=case.get('participant_role'),
                   dt_s=case['step_dt_s'], contract=contract,
                   measured_frames=[dict(step=p['step'], xy=p['position_world'],
                       yaw=p.get('yaw_world'), speed=p.get('speed_mps')) for p in frames])
    system = ("You are a measured-trajectory anchor extractor. Only select the given steps; do not create coordinates, do not rewrite motion or goals. Retain trajectory shape densely, especially near waiting, starting, turning, intrusion, and contact; do not select only a few events. Output only JSON: anchor_extraction_reason in brief English, selected_steps a strictly increasing integer array.")
    user = ("Select at least ceil(min_density\u00d7frame count) anchors and at most max_anchors anchors; the interval between adjacent steps must not exceed max_gap_steps, and mandatory_steps must all be retained. A small number of redundant frames may be omitted in ordinary smooth segments; near contact and in motion-change segments, retain every frame as much as possible. Coordinates are taken by the program from the corresponding measured positions; anchors cannot be moved.\n" +
            json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    return dict(system_prompt=system, user_prompt=user, extraction_contract=contract)


def validate_dense_anchors(case, payload, contract):
    reason = payload.get('anchor_extraction_reason')
    selected = payload.get('selected_steps')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Provide a short anchor_extraction_reason')
    if not isinstance(selected, list) or not selected or any(type(s) is not int for s in selected):
        raise ValueError('selected_steps must contain integers')
    if any(b <= a for a, b in zip(selected, selected[1:])):
        raise ValueError('selected_steps must be unique and increasing')
    frames = {p['step']:p for p in case['adv_rollout_traj'] if p['step'] > 0}
    if set(selected) - set(frames):
        raise ValueError('Select only measured future frames')
    minimum = math.ceil(contract['min_density'] * len(frames))
    if not minimum <= len(selected) <= contract['max_anchors']:
        raise ValueError('Expected %d..%d selected frames, got %d' %
                         (minimum, contract['max_anchors'], len(selected)))
    if set(contract['mandatory_steps']) - set(selected):
        raise ValueError('Missing mandatory start/end/contact-neighborhood frames')
    if any(b-a > contract['max_gap_steps'] for a,b in zip(selected, selected[1:])):
        raise ValueError('Dense frame coverage contains an excessive gap')
    return dict(anchor_extraction_reason=reason.strip(),
                adv_anchor_plan=[dict(step=s, position_world=frames[s]['position_world'],
                                     reason='llm_selected_measured_frame') for s in selected],
                extraction_method='llm_dense_measured_frame_selection',
                extraction_contract=contract, measured_frame_count=len(frames),
                selected_frame_fraction=len(selected)/len(frames))
