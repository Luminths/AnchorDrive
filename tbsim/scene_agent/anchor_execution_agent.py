"""Native ReAct execution with sparse timed anchors and two corrective rollbacks."""
import copy
import math

from .anchor_controller import AnchorController
from .backend import ForbiddenRearEnd
from .contracts import digest
from .execution_agent import ExecutionTools, TOOLS as BEHAVIOR_TOOLS
from .geometry import point_at, project, wrap
from .simple_goal_agent import _function


SYSTEM = "You are the scenario execution Agent. According to the fixed English goal, closed-loop control the specified attacking vehicle and auxiliary vehicle so that the attacking vehicle actually collides with EGO. Briefly explain decisions in English. You cannot modify the goal, add or remove vehicles, or control EGO; EGO is planned independently by STRIVE, by default replanning every 0.1 seconds, and you cannot require it to cooperate.\nFirst observe, and directly decide actions and act based on the image, position, heading, and speed; usually there is no need to query the road. Only when you need to drive along a lane but the direction is unclear, or when it is unclear whether road boundaries or separation areas are passable, query the relevant lanes or connection segments as needed; querying is not a prerequisite step for act, and do not traverse the road network.\nEach round plans for the future horizon_s seconds (default 3.2 seconds), executes only the first execution_interval_s seconds (default 1 second), and then adjusts based on new observations. Give each vehicle 1-4 event anchors, such as lane-change start, process, and end; do not output dense trajectories. Anchors (x_m,y_m,t_s) use the EGO coordinate system fixed at the time of this round's call: x is meters ahead, y is meters to the left, and t starts from the current moment; 0<t<=horizon_s, strictly increasing and an integer multiple of dt_s, and the last point must cover at least the execution interval. The current vehicle body center is the implicit t=0 starting point; waiting may repeat the current position. remaining_anchors are this vehicle's anchors from the previous round that were not executed, and may be adjusted based on measured state. In the image EGO faces up; relative to the vehicle front, angle 0 degrees is up, +90 degrees is left, and -90 degrees is right.\nMotion must remain reasonable and continuous: connect according to each vehicle's measured heading and speed, gradually move laterally while moving forward, and allow time for acceleration, deceleration, and turning; across rounds there must be no sudden head turning, speed jumps, reversing, turning around in place, or lateral sliding to force a collision.\nFirst-anchor direction check: in this round's fixed EGO coordinate system, subtract this vehicle's current position from the first anchor to obtain the first-segment displacement; the minimum angle between its direction atan2(delta_y,delta_x) and this vehicle's measured heading_relative_ego_deg must not exceed 90 degrees (angle difference wrapped by +/-180), and this is not a restriction on the vehicle front's absolute yaw relative to EGO. Check this before act and before preview confirmation; if it exceeds the limit, modify the anchor and do not confirm execution. If the first point coincides with the current position, it means waiting, does not define a displacement direction, and waiting cannot be used to change the vehicle front; even if the angle does not exceed 90 degrees, the above motion continuity must still be satisfied, and it cannot be regarded as permission for instantaneous turning.\nRoad boundary check: before act and before preview confirmation, check all waypoints, the path from the current position to the first point and between points, and consider vehicle body width; you must not go beyond the outer boundary of the drivable road. The road outer boundary is different from lane divider lines inside the road; when there is no physical barrier between adjacent opposite-direction lanes and the road surface is continuous, you may intrude as needed for the goal, but motion must still remain reasonable. If there is a median strip, curb, guardrail, or non-drivable separation area between the roads on both sides, you cannot directly cross it; you may enter only through a passable opening or connection segment if one exists. Blank space on the map or the spacing of centerlines alone cannot prove whether there is a barrier; if it is unclear, query as needed, and if passability still cannot be confirmed, do not plan to cross that area; if you find boundary violation or crossing of a barrier, modify the plan rather than confirm it.\nFor same-direction lane-change collisions, prioritize staying in the adjacent lane, and cut in only when EGO approaches side by side from the side. Do not change lanes early into directly in front of EGO and then decelerate or stop, as this will usually be avoided by EGO braking. An opposite-direction vehicle intruding into EGO's lane does not belong to the same-direction lane-change mechanism. Same-direction same-lane direct rear-end collision from directly behind is prohibited; oblique cutting from the adjacent lane to contact the side or rear corner is allowed. Do not hit background vehicles, do not collide with each other, and do not cross physical obstacles; the auxiliary vehicle should take effect before contact.\nThe cyan line and ego_current_plan in the image are EGO's plan generated for this round based on current information, not the actual future trajectory; EGO will replan according to replan_interval_s and respond to vehicle actions, so do not assume it will continue along that line. Road queries use the L lane / C connection segment numbers in the image, and the connection direction is from entrance to exit; queries do not advance the simulation.\nIf, because you chose the wrong anchor or timing, you miss a collision that originally had a chance to occur, you should rollback to the given previous decision point to correct it, at most twice successfully; briefly describe the error and correction method. EGO's autonomous avoidance is itself not a reason for rollback. If rollback opportunities are used up and the collision timing is still missed, then drive normally along the road, and do not use unreasonable trajectories such as sudden U-turns or reversing to remedy it. After each act, read the action_reminder in the feedback, and before the next act, check motion continuity and collision timing in combination with the measured state. Stop advancing only upon contact with EGO or when time is up; background vehicle collisions are only recorded, and the simulation continues. If it is not appropriate to continue, finish; whether the goal is achieved is left to independent acceptance."

TOOLS = [copy.deepcopy(tool) for tool in BEHAVIOR_TOOLS]
TOOLS[0] = _function('observe', "View the current image, vehicle states, and EGO's plan for this round (cyan line in the figure; may change in the next decision).", {}, [])
TOOLS[1] = _function('get_routes', "When the road direction, boundaries, or passability of separated areas are unclear, query the road topology and geometry as needed by vehicle or L/C ID; generally you can act directly, and the centerline itself does not prove the absence of physical obstruction.", {
    'actor_id': {'type': 'string', 'description': "Participating vehicle ID; provide at least one along with lane_id."},
    'lane_id': {'type': 'string', 'description': "Lxxx / Cxxx in the graph or query results; when omitted, query the vehicle's current lane."},
    'include_centerline': {'type': 'boolean', 'description': "Default false; true returns the geometric information of the queried lane centerline."}}, [])
TOOLS[2] = _function('act', "Submit sparse anchor points for multiple seconds into the future, execute only the first segment (1 second by default), and return the current state.", {
    'reason': {'type': 'string', 'description': "Briefly describe the anchor intent and basis."},
    'actions': {'type': 'array', 'minItems': 1, 'maxItems': 3,
                'items': {'type': 'object', 'additionalProperties': False, 'properties': {
                    'actor_id': {'type': 'string'},
                    'anchors': {'type': 'array', 'minItems': 1, 'maxItems': 4,
                                'description': "1\u20134 event key points, with time not exceeding horizon_s, and the last point covering at least the execution interval; do not output dense trajectories.",
                                'items': {'type': 'object', 'additionalProperties': False,
                                          'properties': {
                                              'x_m': {'type': 'number'},
                                              'y_m': {'type': 'number'},
                                              't_s': {'type': 'number', 'exclusiveMinimum': 0}},
                                          'required': ['x_m', 'y_m', 't_s']}}},
                          'required': ['actor_id', 'anchors']}}}, ['actions', 'reason'])
TOOLS[3]['function']['parameters']['properties']['to_time_s']['description'] = "The previous decision starting point must be selected from rollback_available_times."
TOOLS.append(_function('get_lane_connections',
    "When driving along a connecting segment is required, input two L lanes to query the C connecting segment and centerline from from\u2192to; generally no query is needed.", {
        'from_lane_id': {'type': 'string', 'description': "Entry Lxxx"},
        'to_lane_id': {'type': 'string', 'description': "Exit Lxxx"}}, ['from_lane_id', 'to_lane_id']))

class AnchorExecutionTools(ExecutionTools):
    system_prompt = SYSTEM
    tool_definitions = TOOLS
    execution_mode = 'anchors_direct'

    def __init__(self, world, view, saved, reviewer, root):
        self.execution_s = float(world.config.get('decision_interval_s', 1.))
        self.horizon_s = float(world.config.get('anchor_horizon_s', 3.2))
        if (not math.isfinite(self.execution_s) or not math.isfinite(self.horizon_s)
                or not world.dt <= self.execution_s <= self.horizon_s
                or abs(self.execution_s/world.dt-round(self.execution_s/world.dt)) > 1e-7):
            raise ValueError('Execution interval must be a positive dt multiple within the anchor horizon')
        world.begin_anchor_execution({p['actor_id'] for p in saved['goal']['participants']})
        super().__init__(world, view, saved, reviewer, root)

    def _lane_summary(self, lane_id, state=None):
        catalog = self.view.catalog
        lane = catalog.lane(lane_id)
        reference = ([state['x'], state['y']] if state is not None else
                     point_at(lane.center.xy, lane.center.total_distance*.5)[0])
        along, near, heading, _ = project(lane.center.xy, reference)
        connector = lane_id in catalog.connectors
        left = set(lane.adj_lanes_left) if not connector else set()
        right = set(lane.adj_lanes_right) if not connector else set()
        if not connector:
            # Match the legacy local adjacency inference without constructing routes.
            # Cached nuScenes lanes often lack explicit lateral neighbor links.
            probe, _ = point_at(lane.center.xy, along+10.)
            for other_id, other in self.world.lanes.items():
                if other_id == lane_id or other_id not in catalog.ordinary or len(other.center.xy) < 2:
                    continue
                _, q, h, distance = project(other.center.xy, near)
                _, _, h2, distance2 = project(other.center.xy, probe)
                dx, dy = q[0]-near[0], q[1]-near[1]
                lateral = -dx*math.sin(heading)+dy*math.cos(heading)
                forward = dx*math.cos(heading)+dy*math.sin(heading)
                if (2.4 <= abs(lateral) <= 5.5 and abs(forward) < 1.5
                        and abs(wrap(h-heading)) < .25 and abs(wrap(h2-heading)) < .35
                        and abs(distance2-distance) < 1.):
                    (left if lateral > 0 else right).add(other_id)
        return dict(lane_id=catalog.alias(lane_id), kind='connector' if connector else 'ordinary_lane',
            heading_relative_ego_deg=catalog.heading(heading)['relative_heading_deg'],
            previous_lane_ids=[catalog.alias(x) for x in sorted(lane.prev_lanes)],
            next_lane_ids=[catalog.alias(x) for x in sorted(lane.next_lanes)],
            adjacent_left_lane_ids=[catalog.alias(x) for x in sorted(left) if x in catalog.ordinary],
            adjacent_right_lane_ids=[catalog.alias(x) for x in sorted(right) if x in catalog.ordinary])

    def get_routes(self, actor_id=None, lane_id=None, include_centerline=False):
        if actor_id is None and lane_id is None:
            raise ValueError('Provide actor_id or an L/C lane_id from the image/query')
        if type(include_centerline) is not bool:
            raise ValueError('include_centerline must be a boolean')
        state = None
        current = None
        if actor_id is not None:
            if actor_id not in self.selected:
                raise ValueError('Query a selected attacker/helper')
            state = self.world.states[self.view._actor(actor_id)]
            current = self.view.catalog.nearest_lane(state)
        if lane_id is None:
            if current is None:
                return dict(actor_id=actor_id, current_lane_id=None, lane_id=None,
                    note="No match to the current lane. Please query by the L/C ID in the figure.")
            lane_id = current['lane_id']
        raw_id = self.view.catalog.resolve(lane_id)
        result = self._lane_summary(raw_id, state)
        result['branches'] = [{key: branch[key] for key in
            ('lane_chain', 'exit_lane_id', 'junction_turn_from_lane_end')}
            for branch in self.view.catalog.branches(raw_id, include_centerline=False)]
        if actor_id is not None:
            result.update(actor_id=actor_id, current_lane_id=current['lane_id'] if current else None)
        if include_centerline:
            result['centerline_ego_m'] = self.view.catalog._points([raw_id], max_points=6)
        return result

    def get_lane_connections(self, from_lane_id, to_lane_id):
        found = self.view.get_lane_connections(from_lane_id, to_lane_id)
        return dict(from_lane_id=from_lane_id, to_lane_id=to_lane_id,
            connected=found['connected'], routes=[dict(lane_ids=route['lane_ids'],
                connectors=[dict(lane_id=c['lane_id'], centerline_ego_m=self.view.catalog._points(
                    [self.view.catalog.resolve(c['lane_id'])], max_points=6)) for c in route['connectors']])
                for route in found['routes']])

    def _current_lane(self, state):
        current = self.view.catalog.nearest_lane(state)
        if current is None:
            return None
        raw_id = self.view.catalog.resolve(current['lane_id'])
        return dict(current, kind='connector' if raw_id in self.view.catalog.connectors else 'ordinary_lane')

    def observe(self):
        initial = not self._initial_observation_sent
        result = super().observe(include_routes=False, include_ego_plan=True)
        result['ego']['current_lane'] = self._current_lane(self.world.states['ego'])
        if initial:
            result['ego']['size_m'] = [round(self.world.states['ego'][key], 2) for key in ('length', 'width')]
        result['anchor_contract'] = dict(dt_s=self.world.dt, horizon_s=self.horizon_s, execution_interval_s=self.execution_s,
            max_anchors_per_vehicle=4)
        for actor in result['participants']:
            aid = self.view._actor(actor['actor_id'])
            actor['current_lane'] = self._current_lane(self.world.states[aid])
            plan = self.world.anchor_controller.plans.get(aid)
            if plan:
                remaining = [dict(t_s=round(plan['created_at_s']+p['t_s']-self.world.time, 6),
                    position_ego_m=self.view.catalog.ego_xy(p['position_world_m']))
                    for p in plan['points'][1:] if plan['created_at_s']+p['t_s'] > self.world.time+1e-8]
                if remaining:
                    actor['remaining_anchors'] = remaining
        self.episode.write('anchor_plans.json', [dict(time_s=row['time_s'], plans=row['plans_world'])
            for row in self.episode.action_history])
        return result

    def _build_action_plans(self, actions, reason):
        ep, world = self.episode, self.world
        if ep.stopped:
            raise ValueError('Execution already stopped')
        if digest(ep.proposal) != ep.goal_hash:
            raise ValueError('Immutable goal changed')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Provide a brief reason')
        if (not isinstance(actions, list) or len(actions) != len(self.selected)
                or any(not isinstance(a, dict) or set(a) != {'actor_id', 'anchors'} for a in actions)):
            raise ValueError('Provide actor_id and anchors exactly once for every participant')
        if {a['actor_id'] for a in actions} != set(self.selected):
            raise ValueError('Only command every selected attacker/helper once; never EGO')
        remaining_s = round(world.config['max_simulation_s']-world.time, 6)
        interval_s = min(self.execution_s, remaining_s)
        plans = {}
        # Fully validate the whole multi-car call before mutating any runtime state.
        for action in actions:
            aid = self.view._actor(action['actor_id'])
            plans[aid] = AnchorController.build_plan(world.states[aid], world.states['ego'],
                action['anchors'], world.time, world.dt, interval_s, self.horizon_s)
        return plans, interval_s

    def act(self, actions, reason):
        plans, interval_s = self._build_action_plans(actions, reason)
        ep, world = self.episode, self.world
        self.checkpoints[round(world.time, 6)] = self._checkpoint()
        if hasattr(world, 'prepare_diffusion'):
            world.prepare_diffusion(plans, interval_s)
        world.anchor_controller.set_plans(plans)
        ep.action_history.append(dict(time_s=world.time, duration_s=interval_s,
            actions=copy.deepcopy(actions), plans_world=copy.deepcopy(plans), reason=reason,
            trajectory_method=getattr(world,'trajectory_method','piecewise_linear_direct_no_diffusion')))
        rejection = None
        for _ in range(round(interval_s/world.dt)):
            try:
                before, after, controls = world.step()
            except ForbiddenRearEnd as exc:
                rejection = str(exc)
                break
            contact = ep.monitor.update(before, after, controls, world.time, world.on_road)
            ep.trace.append(dict(time_s=world.time, states=after, controls=controls))
            if contact:
                ep.stopped = True
                break
        if world.time >= world.config['max_simulation_s']-1e-8:
            ep.stopped = True
        if hasattr(world, 'diffusion_record'):
            ep.action_history[-1]['diffusion'] = copy.deepcopy(world.diffusion_record)
            ep.write('diffusion_plans.json', [row.get('diffusion') for row in ep.action_history])
        ep._save_trace()
        result = self.observe()
        result['action_reminder'] = self.action_reminder()
        if rejection:
            ep.action_history[-1]['rejected_rear_end'] = rejection
            ep._save_trace()
            result['rejected_direct_rear_end'] = rejection
        return result
