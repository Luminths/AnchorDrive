"""Brief prior-experiment reference poses; never replayed as actions."""
import math
from .geometry import relative,wrap


def prior_gap_reference(measurements,actor_id,ego):
    past=measurements.get(actor_id)
    if not past:return None
    dx,dy=past['missing_translation_world_m']
    if not all(math.isfinite(v) for v in (dx,dy,past['time_s'])):return None
    c,s=math.cos(ego['yaw']),math.sin(ego['yaw'])
    return dict(past_time_s=past['time_s'],past_body_gap_m=round(math.hypot(dx,dy),3),
        missing_translation_xy_m=[round(c*dx+s*dy,3),round(-s*dx+c*dy,3)],
        note="When the past reference trial run was closest, the difference vector from the closest point on the controlled vehicle body to the closest point on the EGO body, converted to the current fixed coordinates. This only explains whether that near miss lacked front-rear timing or lateral intrusion; it is not the displacement or action required for this round, and the current point cannot be translated as a whole and copied directly.")


def prior_execution_reference(rows, actor_id, ego, now_s):
    points=[]
    for offset in (.2,.5,1.,2.,3.):
        target=now_s+offset
        row=next((r for r in rows if abs(r['time_s']-target)<1e-6),None)
        if row is None or actor_id not in row['states']:
            continue
        pose=row['states'][actor_id];r=relative(pose,ego)
        points.append(dict(t_s=offset,xy=[r['longitudinal_m'],r['lateral_m']],
            speed_mps=round(pose['speed'],2),
            heading_deg=round(math.degrees(wrap(pose['yaw']-ego['yaw'])),1)))
    return dict(available=bool(points),points=points,
        note="The position actually executed in the previous experiment with the same goal, converted to the current fixed EGO coordinates for this round; it is only a historical reference, not this round's future, an action command, or the dataset future. The current state may already differ; it must be checked for first-segment continuity, road, and this round's EGO response before modification, and must not be copied directly.")
