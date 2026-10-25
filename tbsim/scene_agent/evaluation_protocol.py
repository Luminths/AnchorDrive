"""Explicit measurement gates; independent of an LLM's semantic verdict."""
from .roles import is_helper_role
import math
import numpy as np

PROTOCOL_VERSION = 'reviewer_v1_20261006'


def motion_audit(trace, actors, limits, previous_controls=None, tolerance=1e-4):
    """Audit the actual unicycle update, including the first control transition."""
    times=np.array([r['time_s'] for r in trace],dtype=float)
    if len(times)<2 or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        return dict(passed=False,reason='No consecutive measured motion',actors={})
    dt=np.diff(times);results={}
    for aid in actors:
        states=[r['states'][aid] for r in trace]
        speed=np.array([s['speed'] for s in states],dtype=float)
        yaw=np.unwrap([s['yaw'] for s in states])
        xy=np.array([[s['x'],s['y']] for s in states],dtype=float)
        acc=np.diff(speed)/dt;rate=np.diff(yaw)/dt
        previous=(previous_controls or {}).get(aid,[0.,0.])
        jerk=np.diff(np.r_[previous[0],acc])/dt
        yaw_acc=np.diff(np.r_[previous[1],rate])/dt
        expected=xy[:-1]+dt[:,None]*np.column_stack((speed[1:]*np.cos(yaw[1:]),speed[1:]*np.sin(yaw[1:])))
        checks=dict(finite=bool(np.isfinite(np.r_[speed,yaw,xy.ravel(),acc,rate,jerk,yaw_acc]).all()),
            speed=bool(speed.min()>=-tolerance and speed.max()<=limits['speed_max']+tolerance),
            acceleration=bool(acc.min()>=limits['acceleration_min']-tolerance and acc.max()<=limits['acceleration_max']+tolerance),
            jerk=bool(abs(jerk).max()<=limits['jerk_max']+tolerance),
            yaw_rate=bool(abs(rate).max()<=limits['yaw_rate_max']+tolerance),
            yaw_acceleration=bool(abs(yaw_acc).max()<=limits['yaw_acceleration_max']+tolerance),
            lateral_acceleration=bool(abs(speed[1:]*rate).max()<=limits['lateral_acceleration_max']+tolerance),
            curvature=bool((abs(rate)-limits['curvature_max']*speed[1:]).max()<=tolerance),
            integration=bool(np.allclose(xy[1:],expected,rtol=0,atol=1e-6)))
        results[aid]=dict(passed=all(checks.values()),checks=checks,
            max_longitudinal_jerk_mps3=float(abs(jerk).max()),max_yaw_rate_radps=float(abs(rate).max()),
            max_position_update_error_m=float(np.linalg.norm(xy[1:]-expected,axis=1).max()))
    return dict(protocol_version=PROTOCOL_VERSION,passed=bool(results) and all(r['passed'] for r in results.values()),
                actors=results,limits=dict(limits),tolerance=tolerance,
                note='Longitudinal jerk is distinct from the vector-jerk realism metric.')


def trial_road_gate(road):
    if not road.get('available'):
        return dict(passed=False,reason='Mapped road footprint evidence is unavailable')
    failures=[]
    for aid,value in road['actors'].items():
        initial=value['actual_start']['tolerance_outside_fraction']
        worst_row=value.get('actual_tolerance_worst',value['actual_worst'])
        worst=worst_row['tolerance_outside_fraction']
        # Only defects of the original recorded layout are inherited. A newly
        # spawned car is responsible for its complete initial body footprint.
        inherited=value.get('initial_source','existing')=='existing'
        allowed=max(initial,.001) if inherited else .001
        if worst>allowed+.001:
            failures.append(dict(actor_id=aid,time_s=worst_row['t_s'],
                                 start_outside_fraction=initial,worst_outside_fraction=worst,
                                 inherited_initial_defect=inherited))
    return dict(passed=not failures,failures=failures,tolerance_m=road['tolerance_m'],
                note='Reject newly worsening full-body departures; inherited initial defects remain disclosed.')


def interaction_category(goal,states):
    """Initial geometric category, not a claim that the requested maneuver occurred."""
    categories=[];ego=states['ego']
    for p in goal['participants']:
        if is_helper_role(p['role']):continue
        actor=states[p['actor_id']]
        angle=abs(math.atan2(math.sin(actor['yaw']-ego['yaw']),math.cos(actor['yaw']-ego['yaw'])))
        if angle>math.radians(120):category='opposing_encroachment'
        elif angle>math.radians(45):category='crossing_or_turning'
        else:
            lateral=-(actor['x']-ego['x'])*math.sin(ego['yaw'])+(actor['y']-ego['y'])*math.cos(ego['yaw'])
            category='same_direction_adjacent' if abs(lateral)>(actor['width']+ego['width'])/2 else 'same_direction_aligned'
        categories.append(category)
    return categories[0] if len(set(categories))==1 else 'mixed'


def designated_contact_origin(trace,attackers):
    from shapely.geometry import Polygon
    from .geometry import corners
    def contact(row,aid):
        return Polygon(corners(row['states'][aid])).intersects(Polygon(corners(row['states']['ego'])))
    initial={a for a in attackers if contact(trace[0],a)}
    eligible=set(attackers)-initial;new={}
    for row in trace[1:]:
        for aid in attackers:
            touching=contact(row,aid)
            if not touching:eligible.add(aid)
            elif aid in eligible and aid not in new:new[aid]=row['time_s']
    return dict(initial_contact_actors=sorted(initial),new_contact_time_s_by_actor=new,
                has_new_designated_contact=bool(new),
                note='Initial overlap alone never counts; an initially touching pair must separate before a new contact can qualify.')
