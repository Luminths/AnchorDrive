"""Measured-state arithmetic for the planner; suggestions never override actions."""
import math
from .geometry import relative


def straight_speed_references(actor, ego, previous_acceleration=0., dt=.1):
    """Numerical examples under execution limits, not selected actions."""
    from tbsim.utils.control_projection import DEFAULT_LIMITS
    cfg=DEFAULT_LIMITS;r=relative(actor,ego);heading=actor['yaw']-ego['yaw']
    options={}
    for label,target in [('maintain',0.),('accelerate_limit',cfg['acceleration_max']),('brake_limit',cfg['acceleration_min'])]:
        speed=actor['speed'];acceleration=previous_acceleration;distance=0.;points=[]
        for step in range(1,round(2/dt)+1):
            acceleration=max(acceleration-cfg['jerk_max']*dt,min(acceleration+cfg['jerk_max']*dt,target))
            acceleration=max(cfg['acceleration_min'],min(cfg['acceleration_max'],acceleration))
            speed=max(0.,min(cfg['speed_max'],speed+acceleration*dt));distance+=speed*dt
            if step in [round(t/dt) for t in (.5,1.,2.)]:
                points.append(dict(t_s=round(step*dt,3),speed_mps=round(speed,2),
                    xy=[round(r['longitudinal_m']+distance*math.cos(heading),2),
                        round(r['lateral_m']+distance*math.sin(heading),2)]))
        options[label]=points
    return dict(options=options,note="Arithmetic reference for speed/displacement along the current heading only, with no turning; it is not a mandatory action; when moving laterally, adjust according to the road and curvature. The hold branch includes the previous actual acceleration gradually returning to zero.")


def measured_motion_summary(rows, actor_id):
    """Expose measured speed and separation arithmetic, never a verdict."""
    if len(rows)<2:return None
    first,last=rows[0],rows[-1]
    a,b=first['states'][actor_id],last['states'][actor_id]
    e,f=first['states']['ego'],last['states']['ego']
    start,end=relative(a,e),relative(b,f)
    distance=sum(math.hypot(y['states'][actor_id]['x']-x['states'][actor_id]['x'],
                            y['states'][actor_id]['y']-x['states'][actor_id]['y'])
                 for x,y in zip(rows,rows[1:]))
    duration=last['time_s']-first['time_s']
    if duration<=0:raise ValueError('Measured interval must have positive duration')
    return dict(interval_s=round(duration,3),
        speed_start_end_mps=[round(a['speed'],2),round(b['speed'],2)],
        ego_speed_start_end_mps=[round(e['speed'],2),round(f['speed'],2)],
        speed_change_mps=round(b['speed']-a['speed'],2),
        actual_path_length_m=round(distance,2),average_path_speed_mps=round(distance/duration,2),
        relative_longitudinal_start_end_m=[start['longitudinal_m'],end['longitudinal_m']],
        absolute_longitudinal_separation_change_m=round(abs(end['longitudinal_m'])-abs(start['longitudinal_m']),3),
        center_distance_change_m=round(end['distance_m']-start['distance_m'],3),
        note="A positive spacing change indicates an increase, and a negative one indicates a decrease; only a positive speed change is actual acceleration. The longitudinal axis changes with the measured EGO heading; on curves, combine the vehicle body/road. This only provides arithmetic, and does not judge opportunities or actions.")


def motion_cues(actor, ego):
    relation=relative(actor,ego)
    heading=actor['yaw']-ego['yaw']
    vx=actor['speed']*math.cos(heading);vy=actor['speed']*math.sin(heading)
    half=(ego['length']+abs(math.cos(heading))*actor['length']+abs(math.sin(heading))*actor['width'])/2
    gap=relation['longitudinal_m'];separation_velocity=vx-ego['speed']
    result=dict(position_frame='fixed EGO pose at request time; ego motion is NOT subtracted from waypoint coordinates',
        measured_velocity_xy_mps=[round(vx,2),round(vy,2)],
        constant_velocity_reference=[dict(t_s=t,xy=[round(relation['longitudinal_m']+vx*t,2),round(relation['lateral_m']+vy*t,2)]) for t in (.5,1.)],
        reference_note="Only verify the first segment's speed and direction, not the target action; modify this reference when acceleration/deceleration/steering is needed",
        current_relative=relation)
    window=None
    if abs(separation_velocity)>.05:
        first,last=sorted(((-half-gap)/separation_velocity,(half-gap)/separation_velocity))
        if last>=0:window=dict(start_s=round(max(0.,first),2),end_s=round(last,2))
    elif abs(gap)<=half:
        window=dict(start_s=0.,end_s=None)
    result['longitudinal_projection']=dict(
        current_overlap=abs(gap)<=half,combined_half_extent_m=round(half,2),
        relative_velocity_mps=round(separation_velocity,2),constant_velocity_overlap_window_s=window,
        note="Only the longitudinal projection overlap period extrapolated from both parties' current speed and heading; it does not include lateral contact judgment or closed-loop response, and is not an action constraint. end_s=None means there is currently approximately no longitudinal separation speed; the entire window being null means there is currently no future overlap under constant-velocity extrapolation. A positive relative speed means this vehicle is faster than EGO along EGO's forward direction.")
    if abs(math.atan2(math.sin(heading),math.cos(heading)))<math.pi/4:
        closing=ego['speed']-vx
        result['same_direction_longitudinal_overlap_now']=abs(gap)<=half
        if closing>.2 and gap>half:
            result['estimated_overlap_in_s']=round((gap-half)/closing,2)
            result['estimate_note']="Start of longitudinal body overlap extrapolated at the current speed/on a straight road; EGO will change speed in a closed loop, so the lane in the figure must be considered"
    return result


def ego_forecast_cues(world):
    """Expose the existing online EGO plan, never the recorded dataset future."""
    plan=getattr(world,'_ego_plan',None)
    if not plan:return dict(available=False)
    ego=world.states['ego'];dt=world.dt;created=plan.get('created_at_s',world.time)
    points=[]
    for t in (.5,1.,2.,3.):
        index=int(round((world.time+t-created)/dt))
        if not 0<=index<len(plan['states']):continue
        a=plan['states'][index];r=relative(a,ego)
        points.append(dict(t_s=t,xy=[r['longitudinal_m'],r['lateral_m']],speed_mps=round(a['speed'],2),
                           heading_deg=round(math.degrees(math.atan2(math.sin(a['yaw']-ego['yaw']),math.cos(a['yaw']-ego['yaw']))),1)))
    return dict(available=bool(points),points=points,replan_interval_s=.1,
        note="A tentative prediction generated by EGO from the current state, which changes every 0.1 seconds and is not guaranteed to be executed; it is not the dataset's true future")
