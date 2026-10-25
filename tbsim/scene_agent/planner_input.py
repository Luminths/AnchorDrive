"""Compact planner instructions and readable input without neighboring vehicles."""

PLANNER_SYSTEM = "You control the specified attacker or assistant vehicle and plan motion under the constraint of scenario goal g. Control only this vehicle; do not change the goal, initial state, or EGO.\nReplan every 1 second and execute only the first 1 second; plan sparse keypoints for the next 1\u20133 seconds.\nCoordinates are fixed at this round's start with EGO as origin: x forward, y left, in meters; heading 0\u00b0 is the same direction, and positive angles turn left. A waypoint is this vehicle's future position, starting from this vehicle's current position, without subtracting EGO's displacement. To compare future spacing, use both vehicles' positions at the same time.\nEGO responds to the environment in closed loop every 0.1 seconds; neither the current prediction nor the last measurement is guaranteed to happen again.\nThe goal is to make contact with EGO in the future, not to drive toward EGO's current position. Combine EGO's prediction or the measured data from a rejected trial run to judge where EGO will be in the future and where this vehicle can be at the same time, then choose waypoints and interaction timing; do not judge the opportunity only by the current spacing, and do not chase a position EGO has already passed.\nStrictly follow the action order and trigger conditions in GOAL such as \"first ... then ...\" and \"when ...\"; do not skip prerequisite steps early just to collide as soon as possible. For example, if it requires \"change lanes only after driving side by side,\" first keep approaching or waiting in the adjacent lane, and start cutting into EGO's lane only after the longitudinal body ranges of the two vehicles overlap; before being side by side, do not cut in ahead of it early. Judge conditions using the executed history, current relative position, and both vehicles' positions at the same future time; do not treat predictions as facts that have already occurred. Preparation and lane changing may span planning windows; the reason should briefly state which step is current and its basis, and the waypoints must reflect that order.\nFirst compare both vehicles' directions and arrival times at the interaction area, then decide to accelerate, decelerate, wait, or turn. When approaching from behind in the same direction, a speed advantage must be established; for crossing, consider decelerating if arriving early, and consider accelerating only if arriving late. Being side by side does not require the centers to be exactly aligned, but if GOAL has more specific conditions, they must be followed. For oncoming passing, leave time for lateral movement. A collision is not required this round, but necessary actions must not be repeatedly postponed.\nCheck speed against waypoint distance/time: the reason cannot say \"accelerate\" while the points still indicate slow driving. A constant-speed prediction showing no meeting does not mean no meeting is possible after acceleration. Review comments may be wrong; verify against measured data.\nMotion must be continuous with the current speed and heading, and the first segment's direction must make an angle of no more than 90\u00b0 with the heading; do not compensate for a collision by sudden U-turns, reversing, lateral sliding, or speed jumps. Local segment checks limit the speed change rate to 12 m/s\u00b2 and the heading change rate to 2 rad/s; the upper limits need not be fully used.\nSudden U-turns are strictly prohibited, as is splitting a U-turn into multiple segments to chase EGO. If the interaction opportunity has already been lost, simply drive normally in the forward direction permitted by this vehicle's road, and accept that the scenario goal is unachieved; do not ignore the road travel direction to reverse the heading, drive against traffic, reverse, or turn back to force a chase and collision. Judge whether this vehicle has reversed using the road alignment and the measurements before and after; normal steering along a curve is not a U-turn, and you must not give up normal driving merely because this vehicle's heading differs from EGO's.\nKeep the complete vehicle body within the passable road surface and do not pass through road separation areas; missing map data does not prove a physical obstacle. Rear-ending directly from straight behind in the same lane is prohibited, as is cutting directly in front of EGO and decelerating or stopping to cause a rear-end collision. Background collisions are allowed, but they cannot substitute for the target EGO contact.\nIn the image, the gray START dashed box is the scenario initial position; blue is EGO, orange is the controlled vehicle, solid gray vehicles are background vehicles, and arrows are headings. Cyan is EGO's prediction, and purple is this vehicle's remaining plan. Rejected-feedback image: solid vehicles are the positions before the restored plan (BEFORE); the thick blue/red lines are that run's actual EGO/this-vehicle trajectories, and the light vehicle boxes TRIAL END are the trial-run endpoints. Only this round's actually executed interval is drawn; lighter thin lines are submitted history; the trial run has been revoked, so its endpoint cannot be treated as the current state, and EGO is not guaranteed to repeat that response after the plan is modified.\nOutput only JSON; the reason should briefly state this round's actions and basis, and waypoints should contain 1\u20138 points. Each point uses numeric fields t_s, x_m, y_m, not x/y, xy, array coordinates, or string values.\nt_s is seconds relative to now, range 0.1\u20133.0, in integer multiples of 0.1 seconds and strictly increasing, not required to be equally spaced; a 0-second point is prohibited, and the last point must be >=1.0 second. Before outputting, check the point count, field names, and times.\nLegal format example (format only; do not copy the coordinates): {\"reason\":\"this round's actions and basis\",\"waypoints\":[{\"t_s\":0.2,\"x_m\":10.4,\"y_m\":3.0},{\"t_s\":0.5,\"x_m\":11.0,\"y_m\":3.0},{\"t_s\":1.0,\"x_m\":12.0,\"y_m\":3.0}]}"


def number(value):
    return f'{value:.2f}' if isinstance(value, (int, float)) else "unknown"


def xy(point):
    return '(' + ', '.join(number(v) for v in point) + ')'


def state_line(label, state):
    size=state.get('size_m')
    return (f"{label}: position{xy(state.get('xy', []))}, speed{number(state.get('speed_mps'))}m/s, heading{number(state.get('heading_deg'))}°" +
            (f', dimensions{number(size[0])}×{number(size[1])}meters' if size else '') +
            f", road{state.get('lane') or 'No match'}。")


def executed_lines(label, rows, start=None):
    lines=[]
    for row in rows:
        time=(f"Trial run +{number(row['t_s'])}Seconds" if start is None else
              f"simulation{number(start + row['t_s'])}Seconds")
        lines.append(f"{label} {time}: position{xy(row['xy'])}, speed{number(row['speed_mps'])}m/s, heading{number(row['heading_deg'])}°。")
    return lines


def readable_planner_input(scene):
    """Select only planning essentials; never serialize nearby/peer vehicles."""
    current=scene.get('controlled_vehicle', {})
    control=scene.get('control', current.get('id', "unknown"))
    lines=[f"Scenario target g:{scene.get('goal', '')}",
           f"Control vehicle:{control}（{current.get('role', 'Controlled vehicle')}）。",
           f"current simulation{number(scene.get('time_s'))}seconds, remaining{number(scene.get('remaining_time_s'))}seconds; this round executes{number(scene.get('execution_interval_s', 1.0))}seconds, EGO every{number(scene.get('ego_replan_interval_s', 0.1))}second decision.",
           '', "Current state:"]
    if scene.get('ego'):lines.append(state_line('EGO', scene['ego']))
    if current:lines.append(state_line(control, current))
    gap=(scene.get('current_body_geometry') or {}).get('body_gap_m')
    if gap is not None:lines.append(f'Current closest body gap between the two vehicles{number(gap)}meters.')
    lines+=['', "Road reference (the centerline does not represent vehicle body boundaries; see the image for the complete road surface):"]
    for lane in scene.get('lanes', []):
        points=lane.get('center_xy', [])
        indices=sorted({0, len(points)//2, len(points)-1}) if points else []
        lines.append(f"{lane['id']}: centerline" + '、'.join(xy(points[i]) for i in indices) +
                     "; connect" + ('、'.join(lane.get('next', [])) or "Not provided") + '。')
    failed_ego=scene.get('ego_trial_execution')
    if not failed_ego:
        lines+=['', "EGO current prediction (closed-loop response may change it):"]
        forecast=scene.get('ego_plan') or {}
        if forecast.get('available'):
            for row in forecast.get('points', []):
                lines.append(f"Future +{number(row['t_s'])}seconds: position{xy(row['xy'])}, speed{number(row['speed_mps'])}m/s, heading{number(row['heading_deg'])}°。")
        else:lines.append("No predictions available.")
    if scene.get('remaining') and not failed_ego:
        lines+=['', "This vehicle's remaining plan from the previous round (not executed, may be modified):"]
        for row in scene['remaining']:
            lines.append(f"Future +{number(row['t_s'])}seconds: position{xy(row['xy'])}。")
    history=scene.get('latest_accepted_execution') or {}
    if history:
        interval=history['interval_sim_s']
        lines+=['', f"The previous segment has submitted measured data (simulation{number(interval[0])}–{number(interval[1])}seconds, position has been moved to the current round's coordinates):"]
        for label, rows in [('EGO', history.get('ego', [])), (control, history.get('controlled', []))]:
            lines+=executed_lines(label, [rows[i] for i in sorted({0,len(rows)-1})] if rows else [],interval[0])
        previous=scene.get('last_interval_assessment') or {}
        if previous.get('passed') is False:
            lines.append("The previous segment did not pass the Assessor; submitted per fallback:"+str(previous.get('failure_feedback') or previous.get('reason') or "No reason provided"))
    if failed_ego:
        lines+=['', "Rejected trial run actual measurements (state has been restored to this round's starting point; EGO is not guaranteed to repeat this response):"]
        lines+=executed_lines('EGO',failed_ego)
        lines+=executed_lines(control,scene.get('controlled_trial_execution', []))
        geometry=scene.get('rejected_trial_geometry') or {}
        if geometry.get('closest') and geometry.get('end'):
            lines.append(f"Minimum vehicle body clearance for this trial run{number(geometry['closest']['body_gap_m'])}meters, endpoint{number(geometry['end']['body_gap_m'])}meters.")
    feedback=scene.get('feedback') or {}
    lines+=['', "Feedback for this round:"]
    reason=feedback.get('validation_failure_reason') or feedback.get('assessor_failure_feedback')
    advice=feedback.get('revision_advice') or feedback.get('assessor_revision_advice')
    if reason:lines.append("Rejection reason:"+str(reason))
    if advice:lines.append("Modification suggestions:"+str(advice))
    if not reason and not advice:lines.append("none.")
    lesson=scene.get('previous_experiment') or {}
    if lesson.get('previous_reason') or lesson.get('preparatory_advice'):
        lines.append("Past experimental feedback (for reference only, not measured in this round):"+str(lesson.get('previous_reason', ''))+' '+str(lesson.get('preparatory_advice', '')))
    return '\n'.join(lines)
