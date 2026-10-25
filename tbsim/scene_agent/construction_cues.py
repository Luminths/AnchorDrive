"""Measured initial geometry for goal construction; no validity/action decisions."""
import math


def free_spawn_positions(world,catalog,lane_id,max_positions=6):
    """Read-only lane samples passing the existing spawn footprint checks.

    These are free initial positions, not promised reachable collision setups.
    The construction agent still chooses position, speed, role and mechanism.
    """
    from .geometry import point_at,relative,overlap
    lane=catalog.lane(lane_id);length=float(lane.center.total_distance)
    candidates=[]
    distance=3.
    while distance<=length-3.+1e-8:
        xy,yaw=point_at(lane.center.xy,distance)
        actor=dict(x=float(xy[0]),y=float(xy[1]),yaw=float(yaw),speed=0.,length=4.5,width=1.8)
        relation=relative(actor,world.states['ego'])
        if relation['distance_m']<=35 and world.on_road(actor) and not any(
                overlap(actor,other,margin=1.5) for other in world.states.values()):
            candidates.append(dict(distance_along_lane_m=round(distance,2),
                position_ego_m=catalog.ego_xy(xy),
                heading_relative_ego_deg=round(math.degrees((yaw-world.states['ego']['yaw']+math.pi)%(2*math.pi)-math.pi),1),
                distance_to_current_ego_m=relation['distance_m']))
        distance+=2.
    return sorted(candidates,key=lambda p:p['distance_to_current_ego_m'])[:max_positions]


def construction_cues(actor,ego,forecast_samples=()):
    dx,dy=actor['x']-ego['x'],actor['y']-ego['y'];c,s=math.cos(ego['yaw']),math.sin(ego['yaw'])
    x,y=c*dx+s*dy,-s*dx+c*dy
    heading=(actor['yaw']-ego['yaw']+math.pi)%(2*math.pi)-math.pi
    left=[-math.sin(heading),math.cos(heading)];right=[-v for v in left]
    # Explicit coordinates help expose a misread sign without choosing an action.
    probes={side:dict(position_ego_xy_m=[round(x+unit[0],2),round(y+unit[1],2)],
        absolute_lateral_distance_to_current_ego_axis_m=round(abs(y+unit[1]),2))
        for side,unit in [('own_left',left),('own_right',right)]}
    comparisons=[]
    for p in forecast_samples:
        if abs(p['t_s']-round(p['t_s']))>1e-6:continue
        vector=[p['ego_frame_xy_m'][0]-x,p['ego_frame_xy_m'][1]-y]
        comparisons.append(dict(ego_forecast_t_s=p['t_s'],
            from_initial_actor_to_ego_xy_m=[round(v,2) for v in vector],
            component_along_actor_own_left_m=round(sum(a*b for a,b in zip(vector,left)),2),
            component_along_actor_own_right_m=round(sum(a*b for a,b in zip(vector,right)),2)))
    return dict(initial_position_ego_xy_m=[round(x,2),round(y,2)],initial_is_ahead_of_ego=x>0,
        measured_heading_relative_ego_deg=round(math.degrees(heading),1),initial_speed_mps=round(actor['speed'],2),
        own_left_unit_ego_xy=[round(v,4) for v in left],own_right_unit_ego_xy=[round(v,4) for v in right],
        geometric_one_meter_displacement=probes,
        ego_forecast_comparison=comparisons,
        note="Only the initial measured direction component and the fixed current vehicle heading; the left/right components are positive toward the corresponding tentative EGO point and negative away from it. This is not an executable trajectory, collision proof, or automatic rejection condition. Both the EGO prediction and the steering of both parties may change.")


def surface_cues(world,actor,forecast_samples=()):
    """Measure straight-line shortcuts, without treating them as actual paths."""
    if not hasattr(world,'_road_union'):
        return dict(available=False)
    from shapely.geometry import LineString
    road=world._road_union()
    if road.is_empty:return dict(available=False)
    targets=[dict(t_s=0.,x=world.states['ego']['x'],y=world.states['ego']['y'])]
    targets.extend(dict(t_s=p['t_s'],x=p['world_xy_m'][0],y=p['world_xy_m'][1])
        for p in forecast_samples if p['t_s']>0 and abs(p['t_s']-round(p['t_s']))<1e-6)
    samples=[]
    for p in targets:
        segment=LineString([(actor['x'],actor['y']),(p['x'],p['y'])])
        samples.append(dict(ego_target_t_s=p['t_s'],straight_segment_length_m=round(segment.length,2),
            straight_segment_outside_mapped_surface_m=round(segment.difference(road).length,2)))
    return dict(available=True,samples=samples,
        note="Only measure the straight-line length from the attacking vehicle's initial center to the EGO's current/tentative predicted center that passes outside the map-annotated surface. This is not proof of a planned path, vehicle-body legality, or a physical obstacle; curves/intersections may use a continuous curved path, but clearly separated areas must not be crossed directly. It must not be used to automatically pass or reject.")
