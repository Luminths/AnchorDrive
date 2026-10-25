"""Small, persistent-history ReAct agent for Stage-1 goal definition only."""
import base64
import json
import math
import re
from pathlib import Path

from .client import BudgetExhausted
from .contracts import digest, requires_body_contact, text
from .geometry import corners, point_at, project, relative
from .goal_map import LaneCatalog, render_scene


SYSTEM = "You are the construction Agent for nuScenes critical safety scenarios. You only need to investigate the current scenario yourself and define a English attack goal that can be accepted from the execution trajectory; do not run simulation, control vehicles, or perform acceptance. You may use existing vehicles, spawned vehicles, or both; you may set up one or multiple attack vehicles, and may set up auxiliary vehicles as needed, not limited to one main attack vehicle plus several auxiliary vehicles. The total of attack vehicles and auxiliary vehicles is at most 3; existing vehicles and spawned vehicles count together, and EGO is not counted; you are not required to fill all 3, and each vehicle should have a specific attack action or assisting role. When starting the task, first call read_guide to read the scenario generation guide, then call view_scene to view the initial scenario image; after reading the guide and seeing the initial image, then conceive the goal and investigate relevant vehicles and roads.\nVehicle selection and control: All non-EGO existing vehicles Vxxx in the scenario can be selected as attack vehicles or auxiliary vehicles, including vehicles located on intersection connector segments. Listing the vehicle ID and role in define_goal.participants completes the selection; no additional application for control rights is needed. Existing vehicles Vxxx and spawned vehicles Sx use the same behavior control capabilities in the subsequent execution phase, and both can change speed, steering, and lanes according to the goal; existing vehicles do not have to continue their original background vehicle behavior. Whether to spawn a vehicle should depend on position, heading, speed, and scenario conception; do not spawn a new vehicle because you suspect an existing vehicle is uncontrollable. Unselected existing vehicles remain background vehicles. At the current stage, only the goal and participating vehicles are saved, and vehicles are not driven; in the subsequent closed-loop execution phase, the Agent outputs future spatiotemporal anchors based on observations, and then the trajectory execution module executes them. The goal text itself will not automatically drive vehicles. EGO is always controlled by an independent planning and control module.\nlanguage_goal only uses brief natural language to roughly describe the attack process of each participating vehicle and the real collision requirement, without outputting specific speeds, distances, angles, coordinates, seconds, or anchor values, and without specifying precise contact parts; numbers in vehicle IDs are retained. For example: \"S1 waits stationary in the adjacent lane ahead for EGO, and when EGO approaches from the side, starts and changes lanes to intrude into its path and collides with EGO.\" This qualitative waiting, starting, and lane-changing relationship can be retained; specific values and timing are adjusted by the execution Agent based on feedback. Queried initial state values are written into rationale as needed, not as hard requirements of the goal.\nAll user-facing replies, conception explanations, goals, rationales, and role descriptions use English; tool names, JSON field names, and vehicle/lane IDs remain as-is. Explanations should be concise; when facts overturn assumptions, correct them directly, and do not repeatedly engage in lengthy self-questioning and self-answering.\nYou must conceive first, then query: the first reply after reading the guide and seeing the initial image must first give a preliminary scenario conception in English, then initiate calls to get_vehicles, get_lane, get_lane_connections, or spawn vehicles. Use 2\u20134 sentences to clearly state: which existing vehicle you plan to use or where to spawn a vehicle, what it plans to do, how it will intrude into EGO's future driving path, whether an auxiliary vehicle is needed and its role, and finally point out the information currently most in need of verification. Merely describing the image or saying \"first check the vehicles and map\" does not count as a conception. Precise parameters are not required at this point; unknown speeds, positions, and timing should be marked as to be checked, and do not fabricate facts just to give a conception first; the preliminary conception may be modified based on tool evidence.\nIn each subsequent round, only briefly describe new findings, their impact on the current conception, and the next necessary query, without repeating known information; when there is no new evidence, do not repeatedly reopen already rejected plans. Around the current conception, query vehicles, lanes, or C as needed, then determine attack vehicles and auxiliary vehicles; spawn first if spawning is needed, then highlight only the participating vehicles for review and submit define_goal. If the information is sufficient, submit; there is no need to traverse the map or enumerate all future reactions. An existing vehicle's current speed and route can be changed in subsequent control; do not exclude it merely because it is currently moving away from EGO. In congestion, prioritize considering nearby vehicles changing lanes to intrude into EGO's path; you may also consider a stationary vehicle starting and actively intruding, judging specifically according to the guide and actual space and timing. When constructing a scenario of a lane-change collision with EGO or a stationary opposing vehicle starting and colliding with EGO, the attack vehicle's initial position must be in front of EGO (EGO coordinate x_m > 0, and it may be front-left or front-right); this applies to both selecting existing vehicles and spawned vehicles; do not select a rear vehicle to catch up and then change lanes or start and rear-end. Judge the future interaction timing by combining forward distance, both sides' speeds, and the time required for actions. If spawning repeatedly fails, you may change position, change vehicle, or modify the conception. The specific tools and plan adjustments are chosen autonomously by you.\nCollision mechanism and timing: Do not construct a plan where the attack vehicle first cuts directly in front of EGO and then decelerates or stops, waiting for EGO to rear-end the attack vehicle; EGO can usually avoid it by braking. Prioritize actively intruding laterally to form a collision when approaching side-by-side. If the attack vehicle is initially in the adjacent lane ahead of EGO, it may first stay in the adjacent lane to adjust speed or decelerate and wait; based on the relative positions of both sides, speeds, and the time required for lateral action, begin changing lanes to collide with EGO when EGO is about to approach side-by-side or at another suitable timing; do not cut in front too early and then brake. A stationary vehicle in the adjacent lane ahead starting an attack should also start according to EGO's approach timing and intrude laterally, not first drive in front of EGO and stop. Considering that steering and lateral movement take time, you cannot mechanically wait until fully side-by-side to begin. The timing is adjusted by the subsequent execution Agent according to EGO's real motion; EGO is not required to cooperate.\nMulti-attack-vehicle conception: You may select or spawn two or three vehicles that directly attack EGO; the second vehicle does not have to be set as an auxiliary vehicle. For example, an intersection crossing plan may place one attack vehicle in each of two suitable adjacent lanes, coordinating the crossing through speed and departure timing to form different intersection windows; first check the headings, space, and background vehicles of the two directed lanes, do not assume adjacent lanes are necessarily same-direction, and do not let attack vehicles collide with each other. Explain how each vehicle intrudes into EGO's path; the first contact stops, so the goal may be written as a joint attack and at least one of them colliding with EGO, without requiring each vehicle to complete a collision in sequence. If the road and timing do not support it, you may use only one vehicle.\nDistinguish current facts from future assumptions: EGO being currently stationary does not mean it will start immediately, and a map exit does not mean EGO has chosen that route. language_goal describes the attack vehicle actions and the actual collision goal; rationale separately writes observed facts, the EGO motion assumptions relied upon, and timing not yet confirmed, and does not write assumptions as measured facts. If the goal depends on unknown EGO starting or route, clearly state that this is a condition to be verified in the execution phase; if there is no credible interaction, you may finish, and you cannot merely add the word \"collision\" while lacking a reasonable action basis. When a single vehicle is sufficient, there is no need to add other participating vehicles; a second vehicle coming from another direction does not automatically mean the coordination is effective. The top of the image is only EGO's front, not geographic north.\nIf the spawn position or plan is unsuitable, you may call delete_spawned_vehicle to delete the Sx vehicle you spawned, then spawn again; deletion releases the position and spawn quota, and you cannot delete EGO or original vehicles. When calling view_scene, highlight_actor_ids should only include the attack vehicles and auxiliary vehicles you plan to control; do not add EGO or irrelevant background vehicles used for observation. Leave it empty when participating vehicles have not yet been selected. After selecting participating vehicles, before submitting define_goal, call view_scene again, highlighting only the attack vehicles and auxiliary vehicles finally intended to participate, to check their positions, headings, mutual relationships, and surrounding background vehicles; if you adjusted participating vehicles or spawn positions, you need to view the image again for review. Highlighted vehicles are shown in orange, EGO in blue, and unhighlighted background vehicles in gray.\nAfter completing the image review and having sufficient evidence, call define_goal; when you truly cannot find a credible goal, call finish. The tools are chosen by you, but you must comply with the above requirement of conceiving first and querying afterward. You may use get_vehicles to query several relevant vehicles at once; keep subsequent explanations brief.\nThe image is centered on EGO with the vehicle front facing up, and vehicle arrows indicate the front direction. The positions queried and shown in the image are [meters in front of EGO, meters to the left of EGO], and EGO is always currently at (0,0); the lane start point is not EGO's position. A vehicle's own left and right are based on that vehicle's front; for oncoming vehicles, left and right are opposite to the image's left and right; when describing the intrusion direction, prioritize using the positional relationship relative to EGO's path. When spawning a vehicle, you may use the distance measured along the centerline from the start point, or EGO coordinates x_m/y_m (front positive, left positive). L is an ordinary directed lane, and C is a directed intersection path connecting an entrance and an exit lane, and may go straight, turn, or merge; in the image, C is represented by light-colored curves, and most are not labeled with IDs. When the entrance and exit L are known, use get_lane_connections(from_lane_id,to_lane_id) to query the direction and centerline of the intermediate C; when the ID of C is known, you may also directly use get_lane. C queries are used to verify routes traveling along road connections; lateral lane changes or crossing the line from the opposite direction to intrude do not require a corresponding C to exist, but you still need to check road boundaries and background vehicles, and cannot cross physical obstacles. Map connections only indicate where driving is allowed, and do not indicate that any vehicle will necessarily go there in the future. The image and tools only provide the current state, not EGO's future trajectory from the record. The position of a spawned vehicle is chosen by you; the underlying system checks lane and vehicle body occupancy.\nThe goal is to make the attack vehicle actually collide with EGO. Clearly write the actions the attack vehicle performs starting from its current state, and the final vehicle body collision with EGO; only a collision threat, approach, small gap, or near miss does not count as goal achievement, and do not write it as a choice between \"collision or small gap.\" It is not necessary to limit the collision to a door, front corner, or rear corner. The Agent can only control attack vehicles and auxiliary vehicles, and cannot control EGO; EGO is controlled by an independent planning and control module, so do not require EGO to necessarily decelerate, change lanes, accelerate, or drive along a specified route. Conceive the attack according to EGO's actual motion, and do not fabricate traffic lights, right of way, or a certain future. When vehicle fronts are opposite, the meanings of left and right are easily reversed; when unable to confirm, describe the intrusion direction relative to EGO's path. Direct rear-ending of EGO from directly behind in the same direction and same lane is prohibited."


def _function(name, description, properties, required):
    return dict(type='function', function=dict(name=name, description=description,
        parameters=dict(type='object', properties=properties, required=required)))


TOOLS = [
    _function('view_scene', "View the current scene image with EGO facing up; you can highlight vehicles and adjust the field of view.",
        {'radius_m': {'type':'number','minimum':20,'maximum':75,'description':"field-of-view radius, 20 to 75 meters"},
         'highlight_actor_ids': {'type':'array','items':{'type':'string'},'description':"Fill in only the IDs of the attacking vehicles and auxiliary vehicles to be controlled; do not fill in EGO or unrelated background vehicles, and leave blank when no vehicle is selected."}}, []),
    _function('get_vehicles', "Query the current speed, position relative to EGO, heading direction, and lane of one or more vehicles by the vehicle IDs shown in the figure.",
        {'actor_ids': {'type':'array','minItems':1,'maxItems':8,'uniqueItems':True,'items':{'type':'string'},
            'description':"1 to 8 EGO, Vxxx, or Sx IDs"}}, ['actor_ids']),
    _function('get_lane', "Query by Lxxx/Cxxx the directed centerline EGO coordinates of that road segment, the heading along the line relative to EGO, and the coordinate description.",
        {'lane_id': {'type':'string','description':"Lxxx/Cxxx on the map or known"}}, ['lane_id']),
    _function('get_lane_connections', "Query the directed connection path from the entry L lane to the exit L lane and the geometric information of the intermediate C; it does not cross other ordinary lanes and does not mean the vehicle will necessarily follow this route in the future.",
        {'from_lane_id': {'type':'string','description':"Entry Lxxx"},
         'to_lane_id': {'type':'string','description':"Exit Lxxx"}}, ['from_lane_id','to_lane_id']),
    _function('read_guide', "Read the scenario generation guide and the NHTSA pre-crash vehicle relationship reference for categories 16\u201331.", {}, []),
    _function('spawn_vehicle', "Generate vehicles in ordinary L lanes. Choose one of two position forms: distance_along_lane_m, or give both x_m/y_m (meters ahead of/left of EGO). The coordinate form retains the requested position and does not automatically snap to the centerline; the vehicle heading takes the lane direction near that position. At least 2 meters from both ends of the lane, and it must pass the road and vehicle-body occupancy check.",
        {'lane_id': {'type':'string'},
         'distance_along_lane_m': {'type':'number','minimum':2,'description':"The number of meters measured along the centerline from the start of the directed lane; mutually exclusive with x_m/y_m."},
         'x_m': {'type':'number','description':"EGO coordinates: forward is positive, backward is negative, in meters; must be given together with y_m."},
         'y_m': {'type':'number','description':"EGO coordinates: left is positive, right is negative, in meters; must be given together with x_m."},
         'speed_mps': {'type':'number','minimum':0,'maximum':10,'description':"Initial speed 0\u201310 m/s (endpoints inclusive)."}},
        ['lane_id','speed_mps']),
    _function('delete_spawned_vehicle', "Before fixing the target, delete the vehicle you generated yourself, releasing the occupied position and generation slot; only Sx can be deleted, not EGO or existing vehicles. After deletion, reselect a position or look at the map; subsequently generated vehicles use new IDs.",
        {'actor_id':{'type':'string','description':"ID of the current generated vehicle to delete, for example S1."}}, ['actor_id']),
    _function('define_goal', "Fix one English observable attack objective and select the participating vehicles, then end this stage. The Vxxx/Sx in participants are all handed to subsequent Agents to control with the same behavioral capabilities; no actions are executed currently. List only the attacking vehicle and auxiliary vehicles, not the EGO.",
        {'language_goal': {'type':'string','minLength':1,'maxLength':400,'description':"At most 400 characters. Briefly describe the attack process and the final actual body collision with EGO; do not write specific values such as speed, distance, coordinates, or time (except vehicle IDs); small gaps or near collisions do not count as achieved, and EGO is independently controlled by the planner-controller; do not pile up initial states or treat assumptions about EGO's future as facts."},
         'rationale': {'type':'string','minLength':1,'maxLength':1200,'description':"Maximum 1200 characters. Distinguish observed facts, relied-upon EGO motion assumptions, and unconfirmed timing; do not replace evidence by adding numbers."},
         'participants': {'type':'array','minItems':1,'maxItems':3,'description':"List only the attacking vehicles or auxiliary vehicles to be controlled, at most three in total; multiple attacking vehicles are allowed, for example [{\"actor_id\":\"S1\",\"role\":\"attacking vehicle\"},{\"actor_id\":\"S2\",\"role\":\"attacking vehicle\"}], and auxiliary vehicles are not required. EGO must not be included.",'items':{'type':'object','properties':{
            'actor_id':{'type':'string','description':"For an existing vehicle Vxxx or a generated vehicle Sx ID, do not fill in EGO."},'role':{'type':'string','minLength':1,'maxLength':80}},
            'required':['actor_id','role']}}},
        ['language_goal','rationale','participants']),
    _function('finish', "When there is indeed no trustworthy target, explain the reason and end.",
        {'reason': {'type':'string','minLength':1,'maxLength':600}}, ['reason']),
]


class GoalTools:
    def __init__(self, world, client, dataset_path, root):
        self.world=world;self.client=client;self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.catalog=LaneCatalog(world,dataset_path)
        self.alias_actors={'EGO':'ego'}
        self.actor_aliases={'ego':'EGO'}
        ego=world.states['ego']
        existing=sorted((a for a in world.states.values() if a['id']!='ego'),
            key=lambda a:relative(a,ego)['distance_m'])
        for n,a in enumerate(existing,1):self._bind(a['id'],'V{:03d}'.format(n))
        self.image_count=0;self.goal=None;self.goal_display=None;self.finished=None;self.trace=[]

    def _bind(self,aid,alias):
        self.alias_actors[alias]=aid;self.actor_aliases[aid]=alias

    def _actor(self,alias):
        if alias not in self._available_actor_ids():
            raise ValueError("Unknown vehicle ID "+str(alias)+"; available IDs: "+', '.join(self._available_actor_ids()))
        return self.alias_actors[alias]

    def _available_actor_ids(self):
        return [alias for alias,aid in self.alias_actors.items() if aid in self.world.states]

    def view_scene(self,radius_m=55,highlight_actor_ids=None,ego_plan=None,anchor_preview=None,ego_history=None,execution_comparison=None,execution_layers=None,camera_ego=None,separate_vehicle_labels=False):
        highlight_actor_ids=highlight_actor_ids or []
        if not isinstance(radius_m,(int,float)) or not math.isfinite(radius_m) or not 20<=radius_m<=75:
            raise ValueError('radius_m must be within 20..75')
        if not isinstance(highlight_actor_ids,list):raise ValueError('highlight_actor_ids must be a list')
        selected={self._actor(alias) for alias in highlight_actor_ids}
        name='observation_{:02d}.png'.format(self.image_count);self.image_count+=1
        forecast=None
        if self.world.config.get('goal_regeneration') and ego_plan is None and execution_layers is None:
            forecast=self.world.forecast_ego(horizon_s=self.world.config['max_simulation_s'])
            ego_plan=dict(points=forecast['samples'],replan_interval_s=.1)
        render_kwargs={'ego_plan':ego_plan} if ego_plan is not None else {}
        if anchor_preview is not None:render_kwargs['anchor_preview']=anchor_preview
        if ego_history is not None:render_kwargs['ego_history']=ego_history
        if execution_comparison is not None:render_kwargs['execution_comparison']=execution_comparison
        if execution_layers is not None:render_kwargs['execution_layers']=execution_layers
        if camera_ego is not None:render_kwargs['camera_ego']=camera_ego
        if separate_vehicle_labels:render_kwargs['separate_vehicle_labels']=True
        visible=render_scene(self.world,self.catalog,self.root/name,
            self.actor_aliases.__getitem__,radius_m,selected,**render_kwargs)
        ego=self.world.states['ego']
        visible_actors=[]
        for aid,actor in self.world.states.items():
            pts=[self.catalog.ego_xy(p) for p in corners(actor)]
            if all(min(p[i] for p in pts)<=radius_m and max(p[i] for p in pts)>=-radius_m for i in (0,1)):
                visible_actors.append(self.actor_aliases[aid])
        result=dict(image_name=name,image_path=str(self.root/name),
            ego=dict(speed_mps=round(ego['speed'],2),current_lane=self.catalog.nearest_lane(ego)),
            visible_lane_ids=visible,highlighted=highlight_actor_ids,
            visible_actor_ids=visible_actors,available_actor_ids=self._available_actor_ids(),
            actor_count=len(self.world.states),non_ego_actor_count=len(self.world.states)-1,
            spawn_limits=dict(initial_speed_mps=[0,10],max_spawned_vehicles=self.world.config['max_spawned_vehicles'],
                max_controlled_vehicles=self.world.config['max_controlled_vehicles']),
            frame='EGO forward is image up; EGO left is image left')
        if forecast is not None:
            result['ego_online_forecast']=dict(source='current_IDM_shadow_forecast',
                note="Inferred from the current state, not the dataset future. Other vehicles maintain measured speed/road; attacks will change responses, and EGO updates every 0.1 seconds.",
                samples=[{k:p[k] for k in ('t_s','ego_frame_xy_m','speed_mps','yaw_change_rad')}
                         for p in forecast['samples'] if abs(p['t_s']-round(p['t_s']))<1e-6])
        return result

    def get_vehicles(self,actor_ids):
        if not isinstance(actor_ids,list) or not 1<=len(actor_ids)<=8 or len(set(actor_ids))!=len(actor_ids):
            raise ValueError('Query 1..8 distinct actor IDs')
        available=self._available_actor_ids()
        unknown=[alias for alias in actor_ids if alias not in available]
        result=dict(vehicles=[self._vehicle(alias) for alias in actor_ids if alias in available])
        if unknown:result.update(unknown_actor_ids=unknown,available_actor_ids=available,
            note="Some IDs do not exist; the remaining valid vehicles have been returned. Please use available_actor_ids and do not guess numbers.")
        return result

    def _vehicle(self,actor_id):
        aid=self._actor(actor_id)
        actor=self.world.states[aid]
        result=dict(actor_id=actor_id,source=actor['source'],
            position_ego_m=self.catalog.ego_xy([actor['x'],actor['y']]),
            speed_mps=round(actor['speed'],2),heading=self.catalog.heading(actor['yaw']),
            current_lane=self.catalog.nearest_lane(actor),
            lane_match_note="current_lane is the nearby centerline matched by distance and vehicle heading direction, not a strict lane assignment; when distance_to_center_m is large, the vehicle cannot be determined to be in that lane based on this, and it also does not affect selecting that vehicle for control.",
            size_m=[actor['length'],actor['width']])
        if self.world.config.get('goal_regeneration'):
            from .construction_cues import construction_cues,surface_cues
            result['construction_geometry']=construction_cues(actor,self.world.states['ego'])
            result['mapped_surface_geometry']=surface_cues(self.world,actor)
        return result

    def get_lane(self,lane_id):
        lid=self.catalog.resolve(lane_id);lane=self.catalog.lane(lid)
        pts=lane.center.xy
        indices=sorted(set(round(i*(len(pts)-1)/9) for i in range(10)))
        headings=[]
        for i in indices:
            _,_,yaw,_=project(pts,pts[i])
            headings.append(self.catalog.heading(yaw)['relative_heading_deg'])
        result=dict(lane_id=lane_id,
            centerline_ego_m=[self.catalog.ego_xy(pts[i]) for i in indices],
            heading_relative_ego_deg=headings,
            note="Coordinates are [meters ahead of EGO, meters to EGO's left]; points are ordered from start to end along the lane's driving direction. heading corresponds one-to-one with each centerline point: 0\u00b0 faces the top of the image, +90\u00b0 faces left, -90\u00b0 faces right, \u00b1180\u00b0 faces down; this is not geographic orientation. Centerline coordinates can be used to select generation positions.")
        if self.world.config.get('spawn_setup_hints') and lid in self.catalog.ordinary:
            from .construction_cues import free_spawn_positions
            result['free_spawn_positions']=free_spawn_positions(self.world,self.catalog,lid)
            result['spawn_note']="This only indicates that the current empty position passes the original vehicle body road/occupancy check; it does not indicate that a collision can occur. Choose a position and an initial speed of 0\u201310 m/s yourself, then call spawn_vehicle; nearby vehicles will move."
        return result

    def get_lane_connections(self,from_lane_id,to_lane_id):
        start=self.catalog.resolve(from_lane_id);target=self.catalog.resolve(to_lane_id)
        if start not in self.catalog.ordinary or target not in self.catalog.ordinary:
            raise ValueError("The entry and exit must be ordinary L lanes; to query a single C, use get_lane.")
        routes=[]
        pending=[(start,[start])]
        while pending:
            current,path=pending.pop()
            for nxt in sorted(self.catalog.lane(current).next_lanes):
                if nxt==target:
                    route=path+[nxt]
                    routes.append(dict(lane_ids=[self.catalog.alias(x) for x in route],
                        connectors=[self.get_lane(self.catalog.alias(x)) for x in path[1:]]))
                elif nxt in self.catalog.connectors and nxt not in path:
                    pending.append((nxt,path+[nxt]))
        return dict(from_lane_id=from_lane_id,to_lane_id=to_lane_id,
            connected=bool(routes),routes=routes,
            note="Only query directed C connections along the road from entrance to exit. Lane changes to adjacent lanes or crossing lines on continuous pavement do not require C; an empty result does not mean lateral passage is impossible, and do not repeatedly enumerate connections for lateral movement. Whether there is a separation and whether the entire vehicle body can pass should be judged based on the images and the actual pavement.")

    def read_guide(self):
        if self.world.config.get('goal_regeneration'):
            from .goal_regeneration import GUIDE,PHYSICAL_GUIDE
            return dict(guide=PHYSICAL_GUIDE if self.world.config.get('physical_waypoint_tracking') else GUIDE)
        path=Path(__file__).with_name('critical_conflict_guide.md')
        return dict(guide=path.read_text())

    def spawn_vehicle(self,lane_id,distance_along_lane_m=None,speed_mps=None,x_m=None,y_m=None):
        lid=self.catalog.resolve(lane_id)
        if lid not in self.catalog.ordinary:raise ValueError('Spawn on an ordinary L lane')
        lane=self.catalog.lane(lid)
        by_xy=x_m is not None or y_m is not None
        if by_xy and distance_along_lane_m is not None:
            raise ValueError("Choose one of two position forms: distance_along_lane_m or x_m/y_m; they cannot be used simultaneously.")
        if by_xy:
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (x_m,y_m)):
                raise ValueError("x_m and y_m must be provided together, and both must be finite numeric values.")
            desired=self.catalog.world_xy(x_m,y_m)
        else:
            if (isinstance(distance_along_lane_m,bool) or not isinstance(distance_along_lane_m,(int,float)) or
                not math.isfinite(distance_along_lane_m) or
                not 2<=distance_along_lane_m<=lane.center.total_distance-2):
                raise ValueError("distance_along_lane_m must be between 2 and {:.2f} meters; x_m/y_m may also be used to select the position.".format(lane.center.total_distance-2))
            desired,_=point_at(lane.center.xy,distance_along_lane_m)
        if isinstance(speed_mps,bool) or not isinstance(speed_mps,(int,float)) or not math.isfinite(speed_mps):
            raise ValueError("speed_mps must be a finite numeric value.")
        if not 0<=speed_mps<=10:
            raise ValueError("speed_mps is currently {} m/s; the allowed range is 0\u201310 m/s.".format(speed_mps))
        actor=self.world.spawn_at_lane(lid,desired,speed_mps,snap_to_center=not by_xy)
        alias='S'+actor['id'].split('_')[-1];self._bind(actor['id'],alias)
        distance,_,_,_=project(lane.center.xy,[actor['x'],actor['y']])
        return dict(actor_id=alias,position_ego_m=self.catalog.ego_xy([actor['x'],actor['y']]),
            heading=self.catalog.heading(actor['yaw']),current_lane=lane_id,
            placement_mode='ego_xy' if by_xy else 'distance_along_lane',
            distance_along_lane_m=round(distance,2))

    def delete_spawned_vehicle(self,actor_id):
        if self.goal is not None:raise ValueError("The goal has been fixed; vehicles can no longer be deleted.")
        aid=self._actor(actor_id)
        self.world.remove_spawned_actor(aid)
        del self.alias_actors[actor_id]
        del self.actor_aliases[aid]
        return dict(deleted_actor_id=actor_id,available_actor_ids=self._available_actor_ids(),
            remaining_spawn_capacity=max(0,self.world.config['max_spawned_vehicles']-len(self.world.spawned)),
            note="The vehicle has been deleted, and its position and generation slot have been released; please re-examine the image to check the layout. This ID is no longer valid; subsequent generated vehicles use new IDs.")

    def define_goal(self,language_goal,rationale,participants):
        for name,value,limit in (('language_goal',language_goal,400),('rationale',rationale,1200)):
            if isinstance(value,str) and len(value)>limit:
                raise ValueError("{} currently has {} characters, maximum {} characters; please shorten this field.".format(name,len(value),limit))
            text(value,name,limit)
        if not isinstance(participants,list) or not 1<=len(participants)<=min(3,self.world.config['max_controlled_vehicles']):
            raise ValueError('Choose 1..max_controlled_vehicles participants')
        selected=[];full=[]
        for p in participants:
            if not isinstance(p,dict) or set(p)!={'actor_id','role'}:raise ValueError('Participant needs actor_id and role')
            alias=p['actor_id'];aid=self._actor(alias)
            if aid=='ego':
                raise ValueError('participants contains EGO, which is the tested vehicle and is not controlled by this agent. Remove only the EGO entry; keep the selected attacker/helper entries (Vxxx or Sx). EGO may remain in language_goal. Do not respawn or re-query those vehicles to fix this error.')
            text(p['role'],'role',80)
            if alias not in language_goal:raise ValueError('Name every participant in language_goal')
            if aid not in self.world.states:
                raise ValueError('Vehicle '+str(alias)+' is not present in the current scene; select a current vehicle ID')
            if aid in selected:raise ValueError('Duplicate participant')
            selected.append(aid);full.append(dict(actor_id=aid,role=p['role']))
        active_maneuver=re.search("\u542f\u52a8|\u8d77\u6b65|\u52a0\u901f|\u79fb\u52a8|\u9a76\u51fa|\u9a76\u5165|\u5207\u5165|\u4fb5\u5165|\u8f6c\u5f2f|\u53d8\u9053|\u6389\u5934|\u6a2a\u7a7f|\u524d\u8fdb|\u540e\u9000",language_goal)
        if (any(self.world.states[aid]['speed']<.3 for aid in selected) and
            re.search("(?:\u4fdd\u6301|\u7ef4\u6301|\u7ee7\u7eed).{0,12}(?:\u9759\u6b62|\u505c\u8f66|\u505c\u9a76|\u96f6\u901f)",language_goal) and
            not active_maneuver):
            raise ValueError('An already stationary attacker merely remaining stationary is not a new attack maneuver')
        if re.search("(?:EGO|\u81ea\u8f66|\u672c\u8f66)\\s*\u4ee5\\s*(?:\u5f53\u524d\u901f\u5ea6\\s*)?\\d+(?:\\.\\d+)?\\s*(?:m/s|\u7c73/\u79d2)",
                     language_goal,re.IGNORECASE):
            raise ValueError('Put the observed initial EGO speed in rationale, not a fixed future goal')
        measurable_language=re.sub("(?:\u907f\u514d|\u9632\u6b62).{0,12}(?:\u63a5\u89e6|\u78b0\u649e|\u8ffd\u5c3e|\u649e\u51fb)",
                                   "\u98ce\u9669",language_goal)
        measurable_language=re.sub("(?:\u63a5\u89e6|\u78b0\u649e|\u8ffd\u5c3e|\u649e\u51fb)(?:\u7684)?(?:\u98ce\u9669|\u5a01\u80c1|\u5371\u9669|\u53ef\u80fd\u6027)",
                                   "\u98ce\u9669",measurable_language)
        if not requires_body_contact(measurable_language):
            raise ValueError("The goal must include an actual body collision between the attacking vehicle and EGO; mere risk, a small gap, or a near miss does not count as achieved.")
        if re.search("(?:EGO|\u81ea\u8f66|\u672c\u8f66).{0,12}(?:\u5fc5\u987b|\u9700\u8981|\u5e94\u5f53).{0,8}(?:\u5236\u52a8|\u5239\u8f66|\u51cf\u901f|\u52a0\u901f|\u53d8\u9053|\u907f\u8ba9)|(?:\u8feb\u4f7f|\u8981\u6c42|\u4f7f\u5f97).{0,10}(?:EGO|\u81ea\u8f66|\u672c\u8f66).{0,12}(?:\u5236\u52a8|\u5239\u8f66|\u51cf\u901f|\u52a0\u901f|\u53d8\u9053|\u907f\u8ba9)",
                     language_goal):
            raise ValueError('Goal must not require a particular EGO response')
        if requires_body_contact(measurable_language):
            ego=self.world.states['ego']
            for aid in selected:
                r=relative(self.world.states[aid],ego)
                if r['longitudinal_m']< -2 and abs(r['lateral_m'])<1.5 and r['heading_delta_rad']<.25:
                    raise ValueError('Direct same-direction same-lane rear-end attack is disallowed')
        aliases=sorted(self.alias_actors,key=len,reverse=True)
        pattern=re.compile(r'(?<![A-Za-z0-9_])(?:'+'|'.join(re.escape(a) for a in aliases)+r')(?![A-Za-z0-9_])')
        resolved=pattern.sub(lambda m:self.alias_actors[m.group()],language_goal)
        goal=dict(language_goal=resolved,rationale=rationale,participants=full)
        if self.world.config.get('goal_quality_review'):
            from .goal_quality_review import review_goal
            verdict=review_goal(self,language_goal,rationale,participants)
            if not verdict['feasible']:
                raise ValueError("Initial layout and goal review failed:"+json.dumps(verdict,ensure_ascii=False)+" Please correct the concept/participating vehicles/generation positions, then look at the figure again before resubmitting. The state is still at the initial moment, and the goal has not yet been fixed.")
        self.world.discard_unselected_spawns(set(selected))
        self.goal=goal;self.goal_display=language_goal
        record=dict(goal=goal,display_goal=language_goal,goal_sha256=digest(goal),
            scene_index=self.world.scene_index,scene_name=self.world.name,
            selected_vehicle_aliases=[p['actor_id'] for p in participants],
            simulation_time_s=self.world.time,stage='goal_only_no_execution')
        (self.root/'goal.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        return dict(finished=True,status='goal_defined',display_goal=language_goal,
            selected_vehicle_aliases=record['selected_vehicle_aliases'])

    def finish(self,reason):
        text(reason,'reason',600)
        self.finished=dict(status='unsupported',reason=reason)
        return dict(finished=True,**self.finished)


def _image_message(path,name):
    data=base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return dict(role='user',content=[dict(type='text',text="view_scene's image "+name),
        dict(type='image_url',image_url=dict(url='data:image/png;base64,'+data))])


def _estimated_context_tokens(messages):
    total=0
    for message in messages:
        content=message.get('content')
        if isinstance(content,str):total+=len(content)//2
        elif isinstance(content,list):
            total+=2000*sum(part.get('type')=='image_url' for part in content)
            total+=sum(len(part.get('text',''))//2 for part in content)
        total+=len(json.dumps(message.get('tool_calls',[]),ensure_ascii=False))//2
    return total


def _compact_if_needed(messages,client,threshold=750000):
    """Only summarize old turns near the model context limit; retain recent tool/image turns."""
    if _estimated_context_tokens(messages)<threshold or len(messages)<12:return messages
    # Find a complete old prefix ending just before an assistant turn; do not split a tool batch.
    cut=max(2,len(messages)-10)
    while cut>2 and messages[cut]['role']!='assistant':cut-=1
    if cut<=2:return messages
    old=[]
    for message in messages[1:cut]:
        content=message.get('content')
        if isinstance(content,list):content='[scene image retained in recent history only]'
        old.append(dict(role=message['role'],content=content,
            tool_calls=message.get('tool_calls')))
    summary=client.chat([dict(role='system',content="Faithfully compress the history. Preserve specific facts about vehicles and lanes already queried, tool errors, vehicles already generated, and unresolved assumptions. Do not fabricate."),
        dict(role='user',content=json.dumps(old,ensure_ascii=False))])
    return [messages[0],dict(role='user',content="Compressed summary of early tools and reasoning:\n"+str(summary.get('content') or ''))]+messages[cut:]


def run_goal_agent(toolkit,max_steps=16):
    from .goal_regeneration import initial_messages,retain_latest_image
    messages=initial_messages(toolkit)
    (toolkit.root/'initial_request.json').write_text(json.dumps(dict(messages=messages,tools=TOOLS),ensure_ascii=False,indent=2))
    for step in range(1,max_steps+1):
        try:
            if getattr(toolkit.world,'config',{}).get('goal_regeneration'):
                messages=retain_latest_image(messages)
            messages=_compact_if_needed(messages,toolkit.client)
            answer=toolkit.client.chat(messages,TOOLS)
        except (BudgetExhausted,ValueError,RuntimeError,OSError) as exc:
            toolkit.trace.append(dict(step=step,tool=None,observation=dict(ok=False,error=str(exc))))
            if 'response truncated' in str(exc) and step<max_steps:
                messages.append(dict(role='user',content="The previous reply was truncated. Please continue briefly and directly call a tool."))
                continue
            break
        calls=answer.get('tool_calls') or []
        messages.append(answer)
        if not calls:
            toolkit.trace.append(dict(step=step,tool=None,assistant=answer.get('content'),
                observation=dict(ok=False,error='Agent ended without define_goal')))
            break
        pending_images=[];terminal=None
        for call in calls:
            name=call.get('function',{}).get('name')
            args={};image_path=None
            try:
                args=json.loads(call['function']['arguments'])
                if not isinstance(args,dict) or name not in {x['function']['name'] for x in TOOLS}:
                    raise ValueError('Unknown tool or non-object arguments')
                if toolkit.goal or toolkit.finished:raise ValueError('Goal stage already finished')
                observation=getattr(toolkit,name)(**args)
                image_path=observation.pop('image_path',None)
                observation=dict(ok=True,**observation)
            except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                observation=dict(ok=False,error=str(exc))
            messages.append(dict(role='tool',tool_call_id=call['id'],name=name,
                content=json.dumps(observation,ensure_ascii=False,allow_nan=False)))
            if image_path:pending_images.append((image_path,observation['image_name']))
            row=dict(step=step,tool=name,arguments=args,observation=observation,
                message_count=len(messages)+len(pending_images),
                image_count=sum(isinstance(m.get('content'),list) for m in messages)+len(pending_images),
                assistant=answer.get('content'))
            toolkit.trace.append(row)
            (toolkit.root/'agent_trace.json').write_text(json.dumps(toolkit.trace,ensure_ascii=False,indent=2))
            print(json.dumps(dict(scene=toolkit.world.scene_index,step=step,tool=name,
                ok=observation['ok'],error=observation.get('error','')[:120]),ensure_ascii=False),flush=True)
            if observation.get('finished'):
                terminal=observation
        # A model turn may issue several calls. Complete every tool response before
        # appending image-bearing user messages, or the tool-call sequence is invalid.
        for path,name in pending_images:messages.append(_image_message(path,name))
        if terminal:
            return dict(status=terminal['status'],reason=terminal.get('reason',''),
                display_goal=toolkit.goal_display,api_requests=toolkit.client.requests,
                reported_tokens=toolkit.client.tokens)
    (toolkit.root/'agent_trace.json').write_text(json.dumps(toolkit.trace,ensure_ascii=False,indent=2))
    return dict(status='no_goal',reason='Agent ended before define_goal',
        api_requests=toolkit.client.requests,reported_tokens=toolkit.client.tokens)
