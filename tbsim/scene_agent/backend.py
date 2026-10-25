"""nuScenes Stage-1 world: existing filter, IDM ego, replay BG, bicycle ADV.

The LLM receives only public_snapshot(). Cached recorded futures are private BG
replay data; EGO planning uses current states and map geometry only.
"""
import copy
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .geometry import relative, project, point_at, wrap, overlap
from .controller import BehaviorController


class ForbiddenRearEnd(ValueError):
    """A controlled vehicle would strike EGO from directly behind."""

    def __init__(self, actor_id, relation, time_s):
        self.actor_id=actor_id
        self.relation=relation
        self.time_s=time_s
        super().__init__(f'Forbidden rear-end by {actor_id} at {time_s:.2f}s: '
                         f'longitudinal={relation["longitudinal_m"]:.2f}m, '
                         f'lateral={relation["lateral_m"]:.2f}m. '
                         'Choose a route, speed, or timing that avoids EGO rear contact.')


def forbidden_rear_end(before_actor, after_actor, before_ego, after_ego):
    """Reject only an aligned, directly-behind strike into EGO's rear."""
    from .geometry import swept_overlap
    from .evaluation import interpolated_pose
    fraction=swept_overlap(before_actor,after_actor,before_ego,after_ego)
    if fraction is None:return None
    actor=interpolated_pose(before_actor,after_actor,fraction)
    ego=interpolated_pose(before_ego,after_ego,fraction)
    contact=relative(actor,ego)
    # The user forbids a follower striking from directly behind, not a
    # diagonal cut-in or side/rear-corner contact. Require near-aligned
    # headings AND centerlines, as well as a center behind EGO's rear half.
    if (contact['longitudinal_m'] < -ego['length']/2+.5 and
        abs(contact['lateral_m']) <= min(ego['width'],actor['width'])/2 and
        abs(contact['heading_delta_rad']) <= .25):
        return contact
    return None


def spawn_contact_setup_priority(slot):
    """Broad-phase ordering only; every returned slot remains selectable."""
    rel=slot['relative_to_ego'];kind=rel['relation']
    longitudinal=rel['longitudinal_m'];lateral=abs(rel['lateral_m'])
    gap=slot['forecast_interaction']['nominal_current_route_min_body_clearance_m']
    distance=rel['distance_m']
    if kind in ('crossing','opposing') and distance<=18 and gap is not None and gap<=4:
        return 0
    if kind in ('left_adjacent','right_adjacent') and abs(longitudinal)<=12 and 2<=lateral<=6:
        return 1
    if kind in ('crossing','opposing') and distance<=25 and gap is not None and gap<=8:
        return 2
    if kind in ('left_adjacent','right_adjacent') and longitudinal<0 and abs(longitudinal)<=30:
        return 3
    if kind in ('crossing','opposing'):
        return 4
    if longitudinal>12:
        return 5
    return 6


class NuscStage1World:
    def __init__(self, scene_index, config, dataset_path, start_frame=11):
        import torch
        from scripts.anchordrive_scene_browser_core import NoCollisionSceneBrowser
        from trajdata.simulation.sim_scene import SimulationScene
        torch.set_num_threads(2)
        self.config=config
        self.scene_index=scene_index
        self.start_frame=start_frame
        browser=NoCollisionSceneBrowser(str(Path(dataset_path).resolve()))
        meta=browser.env.dataset.get_scene(scene_index)
        if start_frame<11 or start_frame+int(config['max_simulation_s']/meta.dt)>=meta.length_timesteps:
            raise ValueError('Start/horizon must fit recorded scene with >=1.1 seconds of history')
        sim=SimulationScene(env_name=meta.env_name,scene_name=meta.name,scene=meta,
                            dataset=browser.env.dataset,init_timestep=start_frame,freeze_agents=True,return_dict=True)
        sim.reset()
        # Retain the native SafeSim environment and scene for trajectory execution.
        # The construction/legacy behavior path still uses its original state backend.
        self._safesim_env=browser.env
        self._simulation_scene=sim
        original_ego=next(a for a in sim.agents if a.name=='ego')
        self.original_vehicle_names=[a.name for a in sim.agents]
        if config.get('baseline_existing_actor_ids') is not None:
            from .initialization import select_baseline_agents
            sim.agents=select_baseline_agents(sim.agents,config['baseline_existing_actor_ids'])
        elif len(sim.agents)>1:
            browser.env._disable_offroad_agents(sim)
        # Stage-1 construction explicitly permits singleton EGO scenes and preserves EGO.
        sim.agents=[original_ego]+[a for a in sim.agents if a.name!='ego']
        self.name=meta.name
        self.map_name=meta.location
        self.dt=float(meta.dt)
        if abs(self.dt-config['dt'])>1e-8:
            raise ValueError('Configured timestep differs from cached scene')
        self._recorded=pd.read_feather(Path.home()/'.unified_data_cache'/meta.env_name/meta.name/'agent_data_dt0.10.feather').set_index(['agent_id','scene_ts'])
        self.states={}
        for agent in sim.agents:
            row=self._recorded.loc[(agent.name,start_frame)]
            self.states[agent.name]=dict(id=agent.name,x=float(row.x),y=float(row.y),yaw=float(row.heading),
                  speed=float(np.hypot(row.vx,row.vy)),length=float(agent.extent.length),width=float(agent.extent.width),source='existing')
        from trajdata.maps.map_api import MapAPI
        # The browser's memoized map may have omitted road areas. Load explicitly:
        # lane polygons alone exclude valid junctions and inter-lane road surface.
        self.vector_map=MapAPI(Path.home()/'.unified_data_cache').get_map(
            meta.env_name+':'+meta.location,scene_cache=None,incl_road_areas=True)
        self.lanes={lane.id:lane for lane in self.vector_map.get_lanes_within(np.array([self.states['ego']['x'],self.states['ego']['y'],0.]),150.)}
        self.routes={}
        self.actor_routes={}
        for actor in self.states.values():
            self.actor_routes[actor['id']]=self._ensure_actor_routes(actor,self._routes_for(actor))
        self.ego_model = config.get('ego_model', 'idm')
        if self.ego_model == 'idm':
            choices = [r for r in self.actor_routes['ego'] if r['kind'] == 'current']
            if not choices:
                raise ValueError('IDM EGO needs a mapped, heading-compatible current route')
            self.idm_route = np.asarray(self.routes[choices[0]['route_id']]['points'], dtype=float)
            self.ego_planner = None
        else:
            raise ValueError('Unknown ego_model: '+str(self.ego_model))
        self.time=0.
        self.spawned=[]
        self._spawn_sequence=0
        self.spawn_slots={}
        self.controller=BehaviorController(config)
        self.bg_controller=BehaviorController(config)
        self.initial_states=copy.deepcopy(self.states)
        self.background_fallbacks=set()
        self._road_geometry=None
        self._topology_cache={}
        self._lane_reference_cache={}
        self._forecast_cache={}
        self._ego_plan=None
        self.ego_replan_steps=int(round(float(config.get('ego_replan_interval_s',.5))/self.dt))
        if self.ego_replan_steps<1 or abs(self.ego_replan_steps*self.dt-float(config.get('ego_replan_interval_s',.5)))>1e-8:
            raise ValueError('ego_replan_interval_s must be a positive multiple of dt')

    def _route(self, lane_id, chain=None):
        chain=list(chain or [lane_id])
        lane=self.vector_map.get_road_lane(chain[-1])
        points=np.concatenate([self.vector_map.get_road_lane(k).center.xy for k in chain])
        # Extend connected lane graph, keeping branches distinct and deterministic.
        if len(chain)<8 and np.linalg.norm(np.diff(points,axis=0),axis=1).sum()<100:
            next_ids=[k for k in sorted(lane.next_lanes) if k not in chain]
            if next_ids:
                result=[]
                for k in next_ids[:3]:
                    result.extend(self._route(lane_id,chain+[k]))
                    if len(result)>=4:break
                return result[:4]
        points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-3]]
        if len(points)<2:return []
        rid='route_'+hashlib.sha256('|'.join(chain).encode()).hexdigest()[:10]
        self.routes[rid]=dict(id=rid,lane_id=lane_id,chain=chain,points=points.tolist())
        return [rid]

    def _routes_for(self, actor):
        ranked=[]
        for lane in self.lanes.values():
            if len(lane.center.xy)<2:continue
            _,_,h,d=project(lane.center.xy,[actor['x'],actor['y']])
            if d<5 and abs(wrap(h-actor['yaw']))<.8:
                ranked.append((d+2*abs(wrap(h-actor['yaw'])),lane.id))
        if not ranked:return []
        root=self.vector_map.get_road_lane(sorted(ranked)[0][1])
        # nuScenes cached lanes often omit adjacency links. Infer only nearby,
        # parallel, same-direction lanes from two spatial probes, not EGO futures.
        adjacent={k:'left' for k in root.adj_lanes_left}
        adjacent.update({k:'right' for k in root.adj_lanes_right})
        root_s,root_q,root_h,_=project(root.center.xy,[actor['x'],actor['y']])
        probe,_=point_at(root.center.xy,root_s+10)
        for other in self.lanes.values():
            if other.id==root.id or len(other.center.xy)<2:continue
            _,q,h,d=project(other.center.xy,root_q)
            _,q2,h2,d2=project(other.center.xy,probe)
            lat=float((q-root_q) @ np.array([-math.sin(root_h),math.cos(root_h)]))
            along=float((q-root_q) @ np.array([math.cos(root_h),math.sin(root_h)]))
            if 2.4<=abs(lat)<=5.5 and abs(along)<1.5 and abs(wrap(h-root_h))<.25 and abs(wrap(h2-root_h))<.35 and abs(d2-d)<1:
                adjacent[other.id]='left' if lat>0 else 'right'
        result=[]
        for lane_id,kind in [(root.id,'current')]+sorted(adjacent.items()):
            other=self.vector_map.get_road_lane(lane_id)
            _,_,h,d=project(other.center.xy,[actor['x'],actor['y']])
            if kind!='current' and (d>7 or abs(wrap(h-actor['yaw']))>.6):continue
            for rid in self._route(lane_id):
                result.append(dict(route_id=rid,kind=kind))
        return result

    def _ensure_actor_routes(self, actor, options):
        """Keep every non-EGO actor usable by the behavior controller."""
        if options or actor['id']=='ego':return options
        rid='route_heading_'+hashlib.sha256(actor['id'].encode()).hexdigest()[:10]
        c,s=math.cos(actor['yaw']),math.sin(actor['yaw'])
        points=[[actor['x']-2*c,actor['y']-2*s],
                [actor['x']+80*c,actor['y']+80*s]]
        self.routes[rid]=dict(id=rid,lane_id=None,chain=[],points=points,
                              source='current_heading_fallback')
        return [dict(route_id=rid,kind='current')]

    def public_snapshot(self):
        ego=self.states['ego']
        actors=[]
        for a in self.states.values():
            rel=relative(a,ego)
            options=[]
            for item in self.actor_routes[a['id']]:
                r=self.routes[item['route_id']]
                s,_,h,d=project(r['points'],[a['x'],a['y']])
                _,h2=point_at(r['points'],s+35)
                options.append(dict(item,distance_to_route_m=round(d,2),turn_35m_rad=round(wrap(h2-h),2)))
            controllable=a['id']!='ego'
            topology=self.route_topology(a['id']) if controllable and options and rel['distance_m']<=60 else None
            actors.append(dict(a,relative_to_ego=rel,route_options=options,controllable=controllable,
                outside_ego_observation_radius=rel['distance_m']>80,
                topology=None if topology is None else dict(potential_conflict=topology['potential_conflict'],
                    conflicting_route_ids=topology['conflicting_route_ids'],scope='Forward offered routes only; query inspect_topology for directions/conflicts')))
        return dict(scene_index=self.scene_index,scene_name=self.name,time_s=round(self.time,3),
                    map_name=self.map_name,actors=actors,
                    ego_policy=self.ego_model.upper()+' current-state/map closed-loop planner',
                    observation_scope='Current measured poses/speeds and static map routes; no recorded EGO future',
                    background_policy='Recorded replay; route-follow continuation if the recording ends')

    def route_topology(self,actor_id,horizon=None,detailed=False):
        from .topology import analyze_actor
        horizon=horizon or self.config['max_simulation_s']
        a=self.states[actor_id];ego=self.states['ego']
        key=(actor_id,horizon,detailed,tuple(a[k] for k in ('x','y','yaw','speed')),
             tuple(ego[k] for k in ('x','y','yaw','speed')))
        if key not in self._topology_cache:
            if len(self._topology_cache)>200:self._topology_cache.clear()
            self._topology_cache[key]=analyze_actor(self,a,horizon=horizon,detailed=detailed)
        return copy.deepcopy(self._topology_cache[key])

    def lane_evidence(self,actor_id,state):
        """Measured deviation from original mapped lane, not a fixed-heading shortcut."""
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from .geometry import corners
        if actor_id not in self._lane_reference_cache:
            options=[o for o in self.actor_routes.get(actor_id,[]) if o['kind']=='current']
            if not options:return None
            lanes=[];polys=[]
            for o in options:
                for lid in self.routes[o['route_id']]['chain']:
                    lane=self.vector_map.get_road_lane(lid)
                    if lid not in [x.id for x in lanes]:lanes.append(lane)
                    if lane.left_edge is not None and lane.right_edge is not None:
                        poly=Polygon(np.concatenate([lane.left_edge.xy,lane.right_edge.xy[::-1]]))
                        if poly.is_valid and poly.area>0:polys.append(poly)
            if not lanes:return None
            self._lane_reference_cache[actor_id]=(lanes,unary_union(polys).buffer(.1))
        lanes,polygon=self._lane_reference_cache[actor_id]
        closest=min((project(l.center.xy,[state['x'],state['y']]) for l in lanes),key=lambda p:p[3])
        _,q,h,_=closest
        offset=(state['x']-q[0])*(-math.sin(h))+(state['y']-q[1])*math.cos(h)
        return dict(signed_center_offset_m=float(offset),footprint_in_original_lane=bool(polygon.covers(Polygon(corners(state)))))

    def _road_union(self):
        if self._road_geometry is None:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            from trajdata.maps.vec_map_elements import MapElementType
            polys=[]
            for area in self.vector_map.elements.get(MapElementType.ROAD_AREA,{}).values():
                p=Polygon(area.exterior_polygon.xy,[h.xy for h in area.interior_holes])
                if p.is_valid and p.area>0:polys.append(p)
            for lane in self.lanes.values():
                if lane.left_edge is not None and lane.right_edge is not None:
                    p=Polygon(np.concatenate([lane.left_edge.xy,lane.right_edge.xy[::-1]]))
                    if p.is_valid and p.area>0:polys.append(p)
            self._road_geometry=unary_union(polys)
        return self._road_geometry

    def on_road(self, actor):
        from shapely.geometry import Polygon
        from .geometry import corners
        # Small tolerance for map edge discretization; full oriented body checked.
        return bool(self._road_union().buffer(.25).covers(Polygon(corners(actor))))

    def search_spawn_slots(self, relation):
        if self.time>0:raise ValueError('Spawning is only allowed before the first simulation step')
        if relation not in ('any','ahead','behind','left_adjacent','right_adjacent','crossing','opposing'):
            raise ValueError('Unknown spawn relation')
        ego=self.states['ego']
        found=[]
        for lane in sorted(self.lanes.values(),key=lambda l:l.id):
            length=lane.center.total_distance
            for s in np.arange(3,max(3,length-3),8):
                xy,yaw=point_at(lane.center.xy,s)
                a=dict(id='spawn_candidate',x=float(xy[0]),y=float(xy[1]),yaw=yaw,speed=4.,length=4.5,width=1.8,source='spawned')
                rel=relative(a,ego)
                if (relation!='any' and rel['relation']!=relation) or not 9<=rel['distance_m']<=50:continue
                if any(overlap(a,b,margin=1.5) for b in self.states.values()) or not self.on_road(a):continue
                options=[dict(route_id=rid,kind='current') for rid in self._route(lane.id)]
                if not options:continue
                sid='slot_'+hashlib.sha256((lane.id+':'+str(round(float(s),2))).encode()).hexdigest()[:10]
                self.spawn_slots[sid]=dict(actor=a,options=options)
                found.append(dict(slot_id=sid,relative_to_ego=rel,route_options=options))
        # Spatial side is only a search hint. Expose directed conflict evidence for
        # slots too, so a new actor need not repeat the same disconnected-lane mistake.
        from .topology import analyze_actor,analyze_forecast_interactions,preview_slot_traffic
        ranked=[]
        if relation=='any':
            shortlist=[]
            for kind in ('behind','crossing','left_adjacent','right_adjacent','opposing','ahead'):
                shortlist.extend(sorted((s for s in found if s['relative_to_ego']['relation']==kind),
                    key=lambda s:s['relative_to_ego']['distance_m'])[:4])
        else:
            shortlist=sorted(found,key=lambda a:a['relative_to_ego']['distance_m'])[:24]
        forecast=self.forecast_ego()
        for slot in shortlist:
            data=self.spawn_slots[slot['slot_id']]
            data['options']=self._routes_for(data['actor']) or data['options']
            topology=analyze_actor(self,data['actor'],options=data['options'])
            forecast_interaction=analyze_forecast_interactions(self,forecast,[slot['slot_id']],
                candidate_states={slot['slot_id']:data['actor']},candidate_routes={slot['slot_id']:data['options']})['actors'][0]
            slot.update(route_options=data['options'],topology=topology,
                traffic_preview=preview_slot_traffic(self,data['actor'],data['options']),
                forecast_interaction=dict(has_usable_spatiotemporal_conflict=forecast_interaction['has_usable_spatiotemporal_conflict'],
                    same_direction_lead_in_ego_corridor=forecast_interaction['same_direction_lead_in_ego_corridor'],
                    lead_merge_contact_weak=forecast_interaction['lead_merge_contact_weak'],
                    long_chase_contact_weak=forecast_interaction['long_chase_contact_weak'],
                    initial_body_clearance_m=forecast_interaction['initial_body_clearance_m'],
                    nominal_current_route_min_body_clearance_m=forecast_interaction['nominal_current_route_min_body_clearance_m'],
                    opportunities=forecast_interaction['opportunities'][:2]))
            slot['contact_setup_priority']=spawn_contact_setup_priority(slot)
            ranked.append(slot)
        risk_order={'nominally_clear':0,'near_traffic':1,'unknown':2,'likely_blocked':3}
        ranked.sort(key=lambda s:(not s['topology']['potential_conflict'],
            risk_order[s['traffic_preview']['risk_level']],
            spawn_contact_setup_priority(s),
            s['forecast_interaction']['same_direction_lead_in_ego_corridor'],
            not s['forecast_interaction']['has_usable_spatiotemporal_conflict'],
            s['relative_to_ego']['distance_m']))
        if relation=='any':
            selected=[];counts={}
            for slot in ranked:
                kind=slot['relative_to_ego']['relation']
                if counts.get(kind,0)>=3:continue
                selected.append(slot);counts[kind]=counts.get(kind,0)+1
                if len(selected)>=12:break
            return selected
        return ranked[:12]

    def spawn(self, slot_id, speed):
        if self.time>0:raise ValueError('No mid-rollout spawning or teleporting')
        if len(self.spawned)>=self.config['max_spawned_vehicles']:raise ValueError('Spawn budget exhausted')
        if slot_id not in self.spawn_slots:raise ValueError('Use a returned spawn slot ID')
        data=self.spawn_slots[slot_id]
        a=dict(data['actor'],id='spawn_'+str(self._spawn_sequence+1),speed=float(speed))
        if not 0<=speed<=10:raise ValueError('Initial spawn speed must be 0..10 m/s')
        if any(overlap(a,b,margin=1.5) for b in self.states.values()):raise ValueError('Spawn slot is now occupied')
        if not self.on_road(a):raise ValueError('Spawn footprint is outside mapped road')
        self.states[a['id']]=a
        self._ego_plan=None
        self._forecast_cache.clear()
        self.actor_routes[a['id']]=self._routes_for(a) or data['options']
        self.spawned.append(a['id'])
        self._spawn_sequence+=1
        self.initial_states=copy.deepcopy(self.states)
        return a

    def spawn_at_lane(self, lane_id, desired_xy, speed, snap_to_center=True):
        """Place a vehicle at a model-chosen point on a directed map lane."""
        if self.time>0:raise ValueError('Spawning is only allowed before execution')
        if len(self.spawned)>=self.config['max_spawned_vehicles']:raise ValueError('Spawn budget exhausted')
        if not 0<=speed<=10:raise ValueError('Initial spawn speed must be 0..10 m/s')
        if lane_id not in self.lanes:raise ValueError('Choose a lane ID from the current map view')
        lane=self.vector_map.get_road_lane(lane_id)
        s,xy,yaw,d=project(lane.center.xy,desired_xy)
        if d>2.:raise ValueError('Chosen point is {:.1f} m from the selected lane center; inspect and choose a point on that lane'.format(d))
        if not snap_to_center:xy=desired_xy
        if s<2. or s>lane.center.total_distance-2.:
            raise ValueError('Spawn at least 2 m inside the selected lane segment, away from its junction endpoints')
        a=dict(id='spawn_'+str(self._spawn_sequence+1),x=float(xy[0]),y=float(xy[1]),
               yaw=float(yaw),speed=float(speed),length=4.5,width=1.8,source='spawned')
        if any(overlap(a,b,margin=1.5) for b in self.states.values()):
            raise ValueError('The new vehicle would overlap another vehicle (including 1.5 m clearance)')
        if not self.on_road(a):raise ValueError('Spawn footprint is outside the mapped drivable road')
        self.states[a['id']]=a
        self._ego_plan=None;self._forecast_cache.clear();self._topology_cache.clear()
        self.actor_routes[a['id']]=self._ensure_actor_routes(a,self._routes_for(a))
        self.spawned.append(a['id'])
        self._spawn_sequence+=1
        self.initial_states=copy.deepcopy(self.states)
        return a

    def remove_spawned_actor(self,actor_id):
        """Undo a generated actor during construction, freeing its occupied space."""
        if self.time>0:raise ValueError("Generated vehicles can only be deleted before the simulation starts.")
        if actor_id not in self.spawned or self.states.get(actor_id,{}).get('source')!='spawned':
            raise ValueError("Only vehicles generated by the Agent in the current scene can be deleted; EGO or original vehicles cannot be deleted.")
        self.discard_unselected_spawns(set(self.spawned)-{actor_id})

    def discard_unselected_spawns(self,selected_actor_ids):
        """Remove construction-only spawns before the fixed execution begins."""
        if self.time>0:raise ValueError('Cannot remove a vehicle after execution starts')
        removed=[aid for aid in self.spawned if aid not in selected_actor_ids]
        if not removed:return []
        for aid in removed:
            self.states.pop(aid,None)
            self.actor_routes.pop(aid,None)
            self._lane_reference_cache.pop(aid,None)
        self.spawned=[aid for aid in self.spawned if aid in selected_actor_ids]
        self.initial_states=copy.deepcopy(self.states)
        self._ego_plan=None
        self._forecast_cache.clear()
        self._topology_cache.clear()
        return removed

    def reset_trial(self):
        self.states=copy.deepcopy(self.initial_states)
        self.time=0.
        self.controller=BehaviorController(self.config)
        self.bg_controller=BehaviorController(self.config)
        self.background_fallbacks=set()
        self._ego_plan=None
        self._forecast_cache.clear()

    def _plan_ego_from(self, states, start_time, label='stage1_agent'):
        """Plan IDM motion and sample it at the simulation timestep."""
        if self.config.get('ego_model', 'idm') == 'idm':
            from .idm_ego import plan_idm
            steps = max(self.ego_replan_steps, int(round(float(self.config.get('ego_plan_horizon_s', 3.))/self.dt)))
            return dict(created_at_s=round(start_time, 6),
                        states=plan_idm(states, self.idm_route, self.dt, steps, self.config))
        raise ValueError('Only IDM EGO is supported')

    def _planned_ego_step(self, states, plan, now, label):
        if plan is None or now>=plan['created_at_s']+self.ego_replan_steps*self.dt-1e-8:
            plan=self._plan_ego_from(states,now,label)
        index=int(round((now-plan['created_at_s'])/self.dt))
        return copy.deepcopy(plan['states'][index+1]),plan

    def _ego_step(self):
        next_state,self._ego_plan=self._planned_ego_step(self.states,self._ego_plan,self.time,'stage1_agent')
        return next_state

    def ego_current_plan(self):
        """Current IDM decision, including its tentative path beyond the next replan."""
        if self._ego_plan is None or self.time>=self._ego_plan['created_at_s']+self.ego_replan_steps*self.dt-1e-8:
            self._ego_plan=self._plan_ego_from(self.states,self.time,'stage1_agent')
        plan=self._ego_plan
        points=[dict(t_s=round(plan['created_at_s']+i*self.dt,3),world_xy_m=[round(a['x'],2),round(a['y'],2)],
                     yaw_rad=round(a['yaw'],3),speed_mps=round(a['speed'],2),executed=i*self.dt<=self.time-plan['created_at_s']+1e-8)
                for i,a in enumerate(plan['states'])]
        return dict(source='current_'+self.config.get('ego_model','idm')+'_forecast',created_at_sim_time_s=plan['created_at_s'],
            replan_interval_s=round(self.ego_replan_steps*self.dt,3),next_replan_at_sim_time_s=round(plan['created_at_s']+self.ego_replan_steps*self.dt,3),
            points=points)

    def forecast_ego(self, horizon_s=None):
        """Counterfactual online EGO forecast; never reads the recorded EGO future.

        Non-EGO traffic is advanced at its currently measured speed along its
        current legal route.  Everything runs on copied state and controllers.
        """
        horizon=float(horizon_s or self.config.get('ego_forecast_horizon_s',4.0))
        horizon=min(horizon,max(0.,self.config['max_simulation_s']-self.time))
        if horizon<=0:return dict(source='online_idm_shadow_rollout',horizon_s=0.,samples=[])
        signature=(round(self.time,6),round(horizon,3),tuple((aid,)+tuple(round(float(a[k]),5) for k in ('x','y','yaw','speed'))
                    for aid,a in sorted(self.states.items())))
        if signature in self._forecast_cache:return copy.deepcopy(self._forecast_cache[signature])
        if len(self._forecast_cache)>12:self._forecast_cache.clear()
        shadow=copy.deepcopy(self.states);origin=copy.deepcopy(shadow['ego'])
        shadow_plan=copy.deepcopy(self._ego_plan)
        follower=BehaviorController(self.config)
        interval=float(self.config.get('ego_forecast_sample_interval_s',.5))
        sample_steps=max(1,int(round(interval/self.dt)))
        total_steps=int(math.ceil(horizon/self.dt-1e-9))

        def sample(t,state):
            d=np.array([state['x']-origin['x'],state['y']-origin['y']])
            c,s=math.cos(origin['yaw']),math.sin(origin['yaw'])
            return dict(t_s=round(t,3),ego_frame_xy_m=[round(float(d@[c,s]),2),round(float(d@[-s,c]),2)],
                        world_xy_m=[round(state['x'],2),round(state['y'],2)],speed_mps=round(state['speed'],2),
                        yaw_change_rad=round(wrap(state['yaw']-origin['yaw']),3))

        samples=[sample(0.,shadow['ego'])]
        for i in range(1,total_steps+1):
            before=shadow
            updated_ego,shadow_plan=self._planned_ego_step(before,shadow_plan,self.time+(i-1)*self.dt,'stage1_ego_forecast')
            updated={'ego':updated_ego}
            for aid,a in before.items():
                if aid=='ego':continue
                options=[o for o in self.actor_routes.get(aid,[]) if o['kind']=='current']
                if options:
                    command=dict(primitive='route_follow',route_id=options[0]['route_id'],target_speed_mps=a['speed'])
                    updated[aid],_=follower.step(a,before['ego'],command,self.routes,self.dt)
                else:
                    updated[aid]=dict(a,x=a['x']+a['speed']*math.cos(a['yaw'])*self.dt,
                                      y=a['y']+a['speed']*math.sin(a['yaw'])*self.dt)
            shadow=updated;t=min(i*self.dt,horizon)
            if i%sample_steps==0 or i==total_steps:samples.append(sample(t,shadow['ego']))
        dyaw=wrap(shadow['ego']['yaw']-origin['yaw'])
        motion=('left_turn' if dyaw>.25 else 'right_turn' if dyaw<-.25 else
                'stopping_or_stopped' if shadow['ego']['speed']<1. else 'approximately_straight')
        result=dict(source='online_idm_shadow_rollout',generated_at_sim_time_s=round(self.time,3),
            horizon_s=round(horizon,3),coordinate_frame='EGO pose at forecast start; x forward, y left',
            traffic_assumption='Every non-EGO vehicle keeps its measured speed and follows its current legal route; no recorded future is read',
            uncertainty='Nominal prediction only. It becomes stale after any attacker action and is refreshed every decision cycle.',
            motion_summary=motion,samples=samples)
        self._forecast_cache[signature]=copy.deepcopy(result)
        return result

    def interaction_opportunities(self, actor_ids, horizon_s=None):
        from .topology import analyze_forecast_interactions
        forecast=self.forecast_ego(horizon_s)
        return analyze_forecast_interactions(self,forecast,actor_ids)

    def route_occupancy(self, actor_ids, horizon_s=3.):
        from .topology import analyze_route_occupancy
        return analyze_route_occupancy(self,actor_ids,horizon_s)

    def step(self):
        before=copy.deepcopy(self.states)
        previous_controller=copy.deepcopy(self.controller)
        previous_bg_controller=copy.deepcopy(self.bg_controller)
        previous_fallbacks=set(self.background_fallbacks)
        updated={'ego':self._ego_step()}
        controls={}
        next_frame=self.start_frame+int(round(self.time/self.dt))+1
        for aid,a in before.items():
            if aid=='ego':continue
            if aid in self.controller.progress:
                command=self.controller.command(a,before['ego'],self.time,self.routes)
                updated[aid],controls[aid]=self.controller.step(a,before['ego'],command,self.routes,self.dt)
            elif a['source']=='existing' and (aid,next_frame) in self._recorded.index:
                r=self._recorded.loc[(aid,next_frame)]
                updated[aid]=dict(a,x=float(r.x),y=float(r.y),yaw=float(r.heading),speed=float(np.hypot(r.vx,r.vy)))
            else:
                self.background_fallbacks.add(aid)
                options=self.actor_routes[aid]
                if options:
                    updated[aid],_=self.bg_controller.step(a,before['ego'],dict(primitive='route_follow',route_id=options[0]['route_id'],target_speed_mps=a['speed']),self.routes,self.dt)
                else:
                    updated[aid]=dict(a,x=a['x']+a['speed']*math.cos(a['yaw'])*self.dt,y=a['y']+a['speed']*math.sin(a['yaw'])*self.dt)
        for aid in self.controller.progress:
            relation=forbidden_rear_end(before[aid],updated[aid],before['ego'],updated['ego'])
            if relation is not None:
                self.controller=previous_controller
                self.bg_controller=previous_bg_controller
                self.background_fallbacks=previous_fallbacks
                raise ForbiddenRearEnd(aid,relation,self.time+self.dt)
        self.states=updated
        self.time=round(self.time+self.dt,6)
        self._forecast_cache.clear()
        return before,copy.deepcopy(updated),controls
