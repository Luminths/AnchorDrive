"""Directed, forward-only map-route interaction queries (not logged EGO intent).

This is a conservative geometric/kinematic broad phase. Passing it does not
predict collision; failing means no conflict was found on the OFFERED routes.
"""
import math
import numpy as np
from .geometry import clearance,project,point_at,relative,wrap


def forward_route(actor, route, horizon, config):
    points=route['points']
    s,q,heading,d=project(points,[actor['x'],actor['y']])
    length=float(np.linalg.norm(np.diff(np.asarray(points),axis=0),axis=1).sum())
    vmax=config.get('topology_assumed_max_speed',25.)
    acceleration=config.get('topology_assumed_acceleration',4.)
    reach=min(vmax*horizon,actor['speed']*horizon+.5*acceleration*horizon*horizon)
    reach=min(max(0,length-s),max(1.,reach))
    arc=np.linspace(0,reach,max(2,int(math.ceil(reach))+1))
    xy=np.array([point_at(points,s+x)[0] for x in arc])
    return arc,xy,dict(distance_to_center_m=round(d,3),remaining_route_m=round(length-s,2),
        heading_error_rad=round(wrap(heading-actor['yaw']),3))


def earliest(distance, speed, acceleration, vmax):
    distance=np.maximum(0,distance)
    speed=min(speed,vmax)
    ramp=max(0,(vmax-speed)/acceleration)
    ramp_distance=speed*ramp+.5*acceleration*ramp*ramp
    return np.where(distance<=ramp_distance,
        (np.sqrt(speed*speed+2*acceleration*distance)-speed)/acceleration,
        ramp+(distance-ramp_distance)/vmax)


def latest(distance, speed, deceleration, horizon):
    distance=np.maximum(0,distance)
    # If stopping before the conflict is possible, arrival can be delayed to the horizon.
    stop_distance=speed*speed/(2*deceleration)
    return np.where(distance>=stop_distance,horizon,
        (speed-np.sqrt(np.maximum(0,speed*speed-2*deceleration*distance)))/deceleration)


def route_pair(actor, actor_route, ego, ego_route, horizon, config):
    aa,axy,ainfo=forward_route(actor,actor_route,horizon,config)
    ea,exy,einfo=forward_route(ego,ego_route,horizon,config)
    distances=np.linalg.norm(axy[:,None,:]-exy[None,:,:],axis=2)
    # 1 m sampling plus a conservative geometric tolerance.
    corridor=(actor['width']+ego['width'])/2+.75
    ii,jj=np.where(distances<=corridor+2.5)
    result=dict(actor_route_id=actor_route['id'],ego_route_id=ego_route['id'],
        forward_path_clearance_m=round(float(distances.min()),3),
        geometry_overlap=bool((distances<=corridor).any()),proximity_overlap=bool(len(ii)),temporal_overlap_possible=False,
        actor_route_status=ainfo,ego_route_status=einfo)
    if not len(ii):
        result['reason']='Forward route corridors are separate; a nearby/differently headed actor is not necessarily a crossing threat'
        return result
    acceleration=config.get('topology_assumed_acceleration',4.)
    deceleration=config.get('topology_assumed_deceleration',7.)
    vmax=config.get('topology_assumed_max_speed',25.)
    amin=earliest(aa[ii]-actor['length']/2,actor['speed'],acceleration,vmax)
    emin=earliest(ea[jj]-ego['length']/2,ego['speed'],acceleration,vmax)
    amax=latest(aa[ii]+actor['length']/2,actor['speed'],deceleration,horizon)
    emax=latest(ea[jj]+ego['length']/2,ego['speed'],deceleration,horizon)
    possible=np.maximum(amin,emin)<=np.minimum(np.minimum(amax,emax),horizon)
    result['temporal_overlap_possible']=bool(possible.any())
    candidates=np.where(possible)[0] if possible.any() else np.arange(len(ii))
    k=int(candidates[np.argmin(np.maximum(amin[candidates],emin[candidates]))])
    i,j=int(ii[k]),int(jj[k]);xy=(axy[i]+exy[j])/2
    ad=axy[min(i+1,len(axy)-1)]-axy[max(i-1,0)]
    ed=exy[min(j+1,len(exy)-1)]-exy[max(j-1,0)]
    angle=abs(wrap(math.atan2(ad[1],ad[0])-math.atan2(ed[1],ed[0])))
    result['path_relation']=('adjacent_near_pass_not_crossing' if not result['geometry_overlap'] else
        'same_direction_shared_corridor' if angle<.6 else 'opposing_shared_corridor' if angle>2.5 else 'crossing_corridors')
    delta=xy-[ego['x'],ego['y']]
    c,s=math.cos(ego['yaw']),math.sin(ego['yaw'])
    result['conflict']=dict(ego_frame_xy_m=[round(float(delta@[c,s]),2),round(float(delta@[-s,c]),2)],
        actor_forward_distance_m=round(float(aa[i]),2),ego_forward_distance_m=round(float(ea[j]),2),
        actor_arrival_interval_s=[round(float(amin[k]),2),round(float(amax[k]),2)],
        ego_arrival_interval_s=[round(float(emin[k]),2),round(float(emax[k]),2)])
    result['reason']='Possible forward proximity; see path_relation. Arrival bounds are only a topology-screening estimate' if possible.any() else 'Nearby forward corridors but estimated arrival windows do not overlap in this horizon'
    return result


def analyze_actor(world, actor, options=None, horizon=None, detailed=False):
    horizon=horizon or world.config['max_simulation_s']
    ego=world.states['ego'];options=options if options is not None else world.actor_routes.get(actor['id'],[])
    ego_options=[o for o in world.actor_routes['ego'] if o['kind']=='current']
    pairs=[];descriptors=[]
    for o in options:
        route=world.routes[o['route_id']]
        for eo in ego_options:
            pair=route_pair(actor,route,ego,world.routes[eo['route_id']],horizon,world.config)
            pair['requires_lane_change']=o['kind'] in ('left','right')
            pairs.append(pair)
        if detailed:
            arc,xy,status=forward_route(actor,route,horizon,world.config)
            c,s=math.cos(ego['yaw']),math.sin(ego['yaw'])
            samples=[]
            for i in sorted(set(list(range(0,len(arc),10))+[len(arc)-1])):
                d=xy[i]-[ego['x'],ego['y']]
                samples.append(dict(ahead_along_route_m=round(float(arc[i]),1),
                    ego_frame_xy_m=[round(float(d@[c,s]),2),round(float(d@[-s,c]),2)]))
            descriptors.append(dict(route_id=route['id'],kind=o['kind'],lane_chain=route.get('chain',[]),
                direction='sample order is legal forward travel; x=EGO forward, y=EGO left',samples=samples,**status))
    possible=[p for p in pairs if p['temporal_overlap_possible']]
    return dict(actor_id=actor['id'],horizon_s=horizon,
        potential_conflict=bool(possible),conflicting_route_ids=sorted({p['actor_route_id'] for p in possible}),
        ego_route_assumption='All offered current-lane successor branches, not a known EGO future; unoffered lane changes not covered',
        pairs=pairs if detailed else sorted(pairs,key=lambda p:(not p['temporal_overlap_possible'],p['forward_path_clearance_m']))[:2],
        routes=descriptors if detailed else [])


def analyze_forecast_interactions(world, forecast, actor_ids, candidate_states=None, candidate_routes=None):
    """Match feasible actor route arrivals to the online IDM EGO forecast."""
    horizon=float(forecast['horizon_s']);samples=forecast['samples']
    acceleration=world.config.get('topology_assumed_acceleration',4.)
    deceleration=world.config.get('topology_assumed_deceleration',7.)
    vmax=world.config.get('topology_assumed_max_speed',25.)
    ego_template=world.states['ego'];results=[]
    candidate_states=candidate_states or {};candidate_routes=candidate_routes or {}
    for aid in actor_ids:
        actor=candidate_states.get(aid,world.states.get(aid));route_results=[]
        options=candidate_routes.get(aid,world.actor_routes.get(aid,[]))
        current_relation=relative(actor,ego_template)
        lead_same_corridor=(current_relation['longitudinal_m']>5 and
            abs(current_relation['lateral_m'])<(actor['width']+ego_template['width'])/2 and
            current_relation['heading_delta_rad']<.35)
        nominal_clearance=float('inf')
        for ego_sample in samples:
            t=float(ego_sample['t_s'])
            nominal_actor=nominal_state_at(world,actor,t,options)
            exy=ego_sample['world_xy_m']
            nominal_ego=dict(ego_template,x=float(exy[0]),y=float(exy[1]),
                yaw=wrap(ego_template['yaw']+float(ego_sample['yaw_change_rad'])))
            nominal_clearance=min(nominal_clearance,clearance(nominal_actor,nominal_ego))
        initial_clearance=clearance(actor,ego_template)
        lead_merge_warning=(current_relation['longitudinal_m']>5 and
            abs(current_relation['lateral_m'])<6 and
            current_relation['heading_delta_rad']<.8 and
            nominal_clearance>3.)
        long_chase_warning=(current_relation['longitudinal_m']<-15 and
            current_relation['heading_delta_rad']<.35 and initial_clearance>15.)
        for option in options:
            route=world.routes[option['route_id']];points=route['points']
            start,_,_,initial_route_distance=project(points,[actor['x'],actor['y']])
            route_length=float(np.linalg.norm(np.diff(np.asarray(points),axis=0),axis=1).sum())
            reach=max(1.,min(route_length-start,actor['speed']*horizon+.5*acceleration*horizon*horizon,vmax*horizon))
            arc=np.linspace(0,reach,max(2,int(math.ceil(reach/.5))+1))
            positions=[point_at(points,start+d) for d in arc]
            candidates=[]
            for ego_sample in samples:
                exy=np.asarray(ego_sample['world_xy_m'])
                distances=np.array([np.linalg.norm(xy-exy) for xy,_ in positions])
                for index in np.argsort(distances)[:3]:
                    distance=float(arc[index]);xy,yaw=positions[index]
                    amin=float(earliest(np.array([max(0,distance-actor['length']/2)]),actor['speed'],acceleration,vmax)[0])
                    amax=float(latest(np.array([distance+actor['length']/2]),actor['speed'],deceleration,horizon)[0])
                    t=float(ego_sample['t_s']);timing=max(0.,amin-t,t-amax)
                    hypothetical=dict(actor,x=float(xy[0]),y=float(xy[1]),yaw=float(yaw))
                    ego_state=dict(ego_template,x=float(exy[0]),y=float(exy[1]),
                                   yaw=wrap(ego_template['yaw']+float(ego_sample['yaw_change_rad'])),
                                   speed=float(ego_sample['speed_mps']))
                    candidates.append(dict(score=timing+clearance(hypothetical,ego_state),timing=timing,
                        t=t,distance=distance,arrival=[amin,amax],body_clearance=clearance(hypothetical,ego_state),
                        ego_xy=ego_sample['ego_frame_xy_m']))
            best=min(candidates,key=lambda x:x['score'])
            spatial=best['body_clearance']<=2.0
            timing=best['timing']<=.25
            route_results.append(dict(route_id=option['route_id'],kind=option['kind'],
                spatial_conflict_with_forecast=spatial,
                nominal_body_overlap_at_route_match=best['body_clearance']<=.05,
                arrival_window_matches_forecast=timing,
                usable_spatiotemporal_conflict=spatial and timing,
                predicted_ego_arrival_s=round(best['t'],2),
                actor_feasible_arrival_interval_s=[round(best['arrival'][0],2),round(best['arrival'][1],2)],
                arrival_time_mismatch_s=round(best['timing'],2),
                actor_forward_distance_m=round(best['distance'],2),
                initial_distance_to_route_center_m=round(initial_route_distance,2),
                route_match_ignores_lateral_entry=initial_route_distance>1.5,
                estimated_body_clearance_at_route_match_m=round(best['body_clearance'],2),
                conflict_point_ego_frame_xy_m=best['ego_xy']))
        route_results.sort(key=lambda x:(not x['usable_spatiotemporal_conflict'],x['estimated_body_clearance_at_route_match_m'],x['arrival_time_mismatch_s']))
        results.append(dict(actor_id=aid,current_relation=current_relation,
            initial_body_clearance_m=round(initial_clearance,2),
            same_direction_lead_in_ego_corridor=bool(lead_same_corridor),
            lead_merge_contact_weak=bool(lead_merge_warning),
            long_chase_contact_weak=bool(long_chase_warning),
            nominal_current_route_min_body_clearance_m=round(nominal_clearance,2) if math.isfinite(nominal_clearance) else None,
            opportunities=route_results[:6],has_usable_spatiotemporal_conflict=any(r['usable_spatiotemporal_conflict'] for r in route_results)))
    return dict(forecast_source=forecast['source'],forecast_generated_at_sim_time_s=forecast.get('generated_at_sim_time_s'),
        interpretation=('Arrival intervals are feasibility estimates matched to the nominal EGO forecast. '
                        'They guide scenario choice but are neither collision predictions nor acceptance criteria.'),actors=results)


def plausible_contact_routes(interaction, actor_speed_mps, clearance_m=.75):
    """Conservative construction evidence, never an execution or review verdict.

    Route matching places an actor on a candidate centerline instantly.  The
    earliest useful match must leave some time to enter that route laterally.
    The 6 m/s floor is deliberately optimistic because Stage 1 has no control
    acceleration cap; this only rejects clear timing impossibilities.
    """
    routes=[]
    for route in interaction.get('opportunities',[]):
        if not route.get('usable_spatiotemporal_conflict'):continue
        if route.get('estimated_body_clearance_at_route_match_m',float('inf'))>clearance_m:continue
        lateral=float(route.get('initial_distance_to_route_center_m',0.))
        minimum_entry_s=lateral/max(float(actor_speed_mps),6.)
        if float(route.get('predicted_ego_arrival_s',0.))+.1<minimum_entry_s:continue
        routes.append(route)
    return routes


def nominal_state_at(world, actor, t, options=None):
    """Current-speed continuation on the actor's current road route, never logged future."""
    options=[o for o in (options if options is not None else world.actor_routes.get(actor['id'],[])) if o['kind']=='current']
    if options:
        points=world.routes[options[0]['route_id']]['points']
        s,_,_,distance=project(points,[actor['x'],actor['y']])
        if distance<=2.5:
            xy,yaw=point_at(points,s+actor['speed']*t)
            return dict(actor,x=float(xy[0]),y=float(xy[1]),yaw=float(yaw))
    return dict(actor,x=actor['x']+actor['speed']*math.cos(actor['yaw'])*t,
                y=actor['y']+actor['speed']*math.sin(actor['yaw'])*t)


def preview_slot_traffic(world, actor, options, assumed_speed_mps=8.):
    """Read-only, constant-speed road occupancy for a possible spawn slot.

    This is a broad-phase traffic warning, not a controlled candidate rollout.
    No agent, controller, or EGO policy is executed here.
    """
    alternatives=[]
    for option in options:
        if option['kind']!='current':continue
        points=world.routes[option['route_id']]['points']
        start,_,_,_=project(points,[actor['x'],actor['y']])
        best=None
        for t in (.5,1.,1.5,2.,2.5):
            xy,yaw=point_at(points,start+assumed_speed_mps*t)
            candidate=dict(actor,x=float(xy[0]),y=float(xy[1]),yaw=float(yaw),speed=assumed_speed_mps)
            for oid,other in world.states.items():
                if oid=='ego':continue
                predicted=nominal_state_at(world,other,t)
                gap=clearance(candidate,predicted)
                if best is None or gap<best['min_body_clearance_m']:
                    best=dict(other_actor_id=oid,time_s=t,min_body_clearance_m=round(gap,2))
        alternatives.append(dict(route_id=option['route_id'],**(best or dict(
            other_actor_id=None,time_s=None,min_body_clearance_m=None))))
    alternatives.sort(key=lambda x:-(x['min_body_clearance_m'] if x['min_body_clearance_m'] is not None else 1e6))
    safest=alternatives[0] if alternatives else None
    best_clearance=None if safest is None else safest['min_body_clearance_m']
    level=('nominally_clear' if safest is not None and best_clearance is None
           else 'unknown' if best_clearance is None
           else 'likely_blocked' if best_clearance<=1.
           else 'near_traffic' if best_clearance<=3. else 'nominally_clear')
    return dict(assumed_speed_mps=assumed_speed_mps,horizon_s=2.5,risk_level=level,
        safest_current_route=safest,route_alternatives=alternatives[:3],
        note='Nominal constant-speed background projection; attack commands may change this risk')


def analyze_route_occupancy(world, actor_ids, horizon_s=3.):
    """Read-only route corridor occupancy by other cars under nominal continuation."""
    horizon=min(float(horizon_s),4.)
    vmax=world.config.get('topology_assumed_max_speed',25.)
    times=np.arange(.5,horizon+.01,.5)
    actors=[]
    for aid in actor_ids:
        attacker=world.states[aid];routes=[]
        for option in world.actor_routes.get(aid,[]):
            rid=option['route_id'];points=world.routes[rid]['points']
            start,_,_,_=project(points,[attacker['x'],attacker['y']])
            route_length=float(np.linalg.norm(np.diff(np.asarray(points),axis=0),axis=1).sum())
            corridors=[]
            for lateral_offset in (-5.,-3.,0.,3.,5.):
                hazards=[]
                for oid,other in world.states.items():
                    if oid in (aid,'ego'):continue
                    best=None
                    for t in times:
                        predicted=nominal_state_at(world,other,float(t))
                        s,xy,yaw,d=project(points,[predicted['x'],predicted['y']])
                        along=s-start
                        if along<0 or along>min(60.,route_length-start) or d>9.:continue
                        hypothetical=dict(attacker,
                            x=float(xy[0]-math.sin(yaw)*lateral_offset),
                            y=float(xy[1]+math.cos(yaw)*lateral_offset),yaw=float(yaw))
                        gap=clearance(hypothetical,predicted)
                        if gap>4.:continue
                        # The Stage-1 adapter can change speed immediately. Do not use
                        # the topology acceleration estimate to hide reachable hazards.
                        earliest_s=max(0.,along-attacker['length']/2)/max(vmax,1.)
                        if earliest_s>t+.5:continue
                        item=dict(other_actor_id=oid,time_s=round(float(t),2),
                            forward_distance_m=round(float(along),2),
                            nominal_body_clearance_m=round(gap,2),
                            earliest_arrival_at_25_mps_s=round(earliest_s,2))
                        if best is None or (gap,t)<(best['nominal_body_clearance_m'],best['time_s']):best=item
                    if best is not None:hazards.append(best)
                hazards.sort(key=lambda h:(h['time_s'],h['nominal_body_clearance_m']))
                corridors.append(dict(offset_m=lateral_offset,non_target_hazards=hazards[:4]))
            routes.append(dict(route_id=rid,kind=option['kind'],lateral_corridors=corridors,
                               non_target_hazards=corridors[2]['non_target_hazards']))
        actors.append(dict(actor_id=aid,routes=routes))
    return dict(horizon_s=horizon,
        assumption='Other cars maintain current speed on a current legal route; no recorded future or proposed attacker action is simulated',
        interpretation='Advisory only: corridor offsets are relative to each route and cover -5,-3,0,+3,+5 m. Arrival lower bounds use 25 m/s with immediate speed change, not a control limit. Empty hazards do not guarantee clearance.',
        actors=actors)
