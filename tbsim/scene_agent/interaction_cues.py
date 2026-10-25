"""Body projection arithmetic for a tentative EGO forecast, never action control."""
import math
from .geometry import relative


def forecast_interaction(actor, ego, forecast):
    if not forecast.get('available'):
        return dict(available=False)
    relation=relative(actor,ego)
    heading=actor['yaw']-ego['yaw']
    velocity=[actor['speed']*math.cos(heading),actor['speed']*math.sin(heading)]
    samples=[]
    for point in forecast.get('points',[]):
        t=point['t_s'];angle=math.radians(point['heading_deg'])
        dx=relation['longitudinal_m']+velocity[0]*t-point['xy'][0]
        dy=relation['lateral_m']+velocity[1]*t-point['xy'][1]
        longitudinal=math.cos(angle)*dx+math.sin(angle)*dy
        lateral=-math.sin(angle)*dx+math.cos(angle)*dy
        delta=heading-angle
        half_long=(abs(math.cos(delta))*actor['length']+abs(math.sin(delta))*actor['width']+ego['length'])/2
        half_lateral=(abs(math.sin(delta))*actor['length']+abs(math.cos(delta))*actor['width']+ego['width'])/2
        samples.append(dict(t_s=t,longitudinal_center_m=round(longitudinal,2),
            lateral_center_m=round(lateral,2),
            longitudinal_body_gap_m=round(abs(longitudinal)-half_long,2),
            lateral_body_gap_m=round(abs(lateral)-half_lateral,2)))
    return dict(available=bool(samples),reference_only=True,samples=samples,
        note="The controlled vehicle uses its current heading at constant speed, and EGO uses the current tentative prediction; positions are relative to that second's predicted EGO heading. body_gap>0 still means separated, <=0 only indicates projection overlap on that axis and does not prove contact. The table does not include the steering/acceleration/deceleration you prepared, nor does it guarantee EGO's actual response. waypoints still use fixed coordinates from this round's starting point.")
