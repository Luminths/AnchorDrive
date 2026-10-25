"""Readable measured-trial input; detailed geometry remains in local artifacts."""
def _number(value):
    return f'{value:.2f}' if isinstance(value, (int, float)) else "unknown"


def _position(value):
    if not isinstance(value, (int, float)):
        return "unknown"
    if value == 0:
        return "longitudinally aligned"
    return ("ahead" if value > 0 else "behind") + _number(abs(value)) + "meters"


def readable_assessor_input(payload, actor_aliases=None):
    """Summarize facts without computing an automatic opportunity verdict."""
    start, end = payload['trial_interval_s']
    lines = [
        f"Scenario goal:{payload['goal']}",
        f"Review vehicle:{payload['control']}; trial run interval: simulation{_number(start)}–{_number(end)}seconds.",
        "The following times are counted from the start of this round. Positions (x,y) are in meters, fixed to the EGO at the start of this round: x forward, y left; heading 0\u00b0 is the starting EGO direction, and positive angles are to the left.",
        f"EGO closed-loop decision interval:{_number(payload['ego_replan_interval_s'])}seconds; the response of the next trial run may change.",
        '', "Vehicle motion (speed in m/s)",
    ]
    dimensions = payload.get('vehicle_dimensions', {})
    for actor, rows in payload['actual_execution'].items():
        size = dimensions.get(actor)
        suffix = f" (length{_number(size['length_m'])}meters, width{_number(size['width_m'])}meters)" if size else ''
        lines.append(f'{actor}{suffix}：')
        for row in rows:
            x, y = row['xy']
            lines.append(f"  +{_number(row['t_s'])}seconds: position({_number(x)}, {_number(y)}), speed{_number(row['speed_mps'])}, heading{_number(row['heading_deg'])}°。")
    lines += ['', "II. Actual relative position to EGO (calculated based on EGO heading at the same moment)"]
    for actor, rows in payload.get('relative_samples', {}).items():
        for row in rows:
            lateral = row['lateral_m']
            side = ("left side" if lateral >= 0 else "right side") + _number(abs(lateral)) + "meters"
            lines.append(f"{actor} +{_number(row['t_s'])}seconds:{_position(row['longitudinal_m'])}，{side}。")
        geometry = payload.get('actual_body_geometry', {}).get(actor, {})
        closest, last = geometry.get('closest'), geometry.get('end')
        if closest and last:
            lines.append(f"{actor}Closest body gap to EGO: +{_number(closest['t_s'])}seconds{_number(closest['body_gap_m'])}meters; endpoint gap{_number(last['body_gap_m'])}meters. A gap is not a collision event.")
        motion = payload.get('measured_motion', {}).get(actor, {})
        if 'actual_path_length_m' in motion:
            lines.append(f"{actor}Actual distance traveled this round{_number(motion['actual_path_length_m'])}meters.")
    safety = payload.get('execution_safety') or {}
    if payload.get('assess_road_bounds', True):
        lines += ['', "III. Road and local checks"]
        road = payload.get('road_footprint') or {}
        road_gate = safety.get('road') or {}
        if type(road_gate.get('passed')) is bool:
            lines.append("Local full-body road check:" + ("Passed." if road_gate['passed'] else "Not passed."))
        else:
            lines.append("No local road check conclusion was provided; it cannot be considered passed.")
        if road.get('available'):
            lines.append(f"The following is the proportion of the vehicle body area outside the annotated road, already accounted for{_number(road.get('tolerance_m'))}meter edge tolerance; outside the annotation is not necessarily a physical obstacle.")
            for actor, evidence in road.get('actors', {}).items():
                def fraction(row):
                    value = row.get('tolerance_outside_fraction')
                    return _number(100 * value) + '%' if isinstance(value, (int, float)) else "unknown"
                worst = evidence.get('actual_tolerance_worst', evidence.get('actual_worst', {}))
                lines.append(f"{actor}: scenario initial{fraction(evidence.get('initial', {}))}; this round start{fraction(evidence.get('actual_start', {}))}, endpoint{fraction(evidence.get('actual_end', {}))}; maximum this round{fraction(worst)}（+{_number(worst.get('t_s'))}seconds).")
        else:
            lines.append("Road measurement is missing; it cannot be determined that the vehicle remained on the road surface at all times.")
    else:
        lines += ['', "III. Local motion check", "This review does not check road boundary violations and will not reject candidates due to boundary violations or road check failures."]
    motion_gate = safety.get('motion') or {}
    if type(motion_gate.get('passed')) is bool:
        lines.append("Local actual motion check:" + ("Passed." if motion_gate['passed'] else "Not passed."))
    else:
        lines.append("No actual motion-pass conclusion is provided locally; review is still required based on actual measurements and images.")
    lines += ['', "IV. Events of this round"]
    events = payload.get('actual_events') or []
    aliases = actor_aliases or {}
    for event in events:
        actor = aliases.get(event.get('actor_id'), event.get('actor_id', "Unknown vehicle"))
        other = aliases.get(event.get('other_id'), event.get('other_id', "Unknown vehicle"))
        kind = "collision" if event.get('kind') == 'collision' else str(event.get('kind', "Unknown event"))
        lines.append(f"simulation{_number(event.get('time_s'))}seconds:{actor}and{other}occurred{kind}。")
    if not events:
        lines.append("No recorded collision events.")
    if payload.get('native_rear_end_rejection'):
        lines.append("Local no-rear-end-collision check:" + str(payload['native_rear_end_rejection']))
    return '\n'.join(lines)
