"""Sparse timed anchors -> direct trajectories, without diffusion or dynamics optimization."""
import copy
import math

import numpy as np

from .contracts import finite_number
from .geometry import wrap


class AnchorController:
    def __init__(self):
        self.plans = {}
        self.previous = {}

    @staticmethod
    def build_plan(actor, ego, anchors, now, dt, execution_s, horizon_s):
        """Freeze this call's EGO frame; prepend the measured actor pose at t=0."""
        if not isinstance(anchors, list) or not 1 <= len(anchors) <= 4:
            raise ValueError('Provide 1..4 sparse event anchors per vehicle, not dense trajectory samples')
        c, s = math.cos(ego['yaw']), math.sin(ego['yaw'])
        points = [dict(t_s=0., position_world_m=[actor['x'], actor['y']])]
        previous_t = 0.
        for anchor in anchors:
            if not isinstance(anchor, dict) or set(anchor) != {'x_m', 'y_m', 't_s'}:
                raise ValueError('Each anchor requires exactly x_m, y_m, t_s')
            x = finite_number(anchor['x_m'], 'x_m')
            y = finite_number(anchor['y_m'], 'y_m')
            t = finite_number(anchor['t_s'], 't_s')
            if not previous_t < t <= horizon_s + 1e-8:
                raise ValueError('Anchor times must increase from >0 to at most the planning horizon')
            if abs(t / dt - round(t / dt)) > 1e-7:
                raise ValueError('Anchor t_s must be a multiple of the simulation dt')
            world_xy = [ego['x'] + c*x - s*y, ego['y'] + s*x + c*y]
            # Observations round positions to centimeters. Repeating that observed
            # position means waiting, rather than creeping or rotating from roundoff.
            if math.hypot(world_xy[0]-points[-1]['position_world_m'][0],
                          world_xy[1]-points[-1]['position_world_m'][1]) <= .01:
                world_xy = list(points[-1]['position_world_m'])
            points.append(dict(t_s=t, position_world_m=world_xy))
            previous_t = t
        if previous_t < execution_s - 1e-8:
            raise ValueError('Last anchor must cover the whole execution interval')
        return dict(created_at_s=now, frame_ego_world=dict(x=ego['x'], y=ego['y'], yaw=ego['yaw']),
                    anchors_ego_m=copy.deepcopy(anchors), points=points)

    def set_plans(self, plans):
        self.plans = copy.deepcopy(plans)

    def step(self, actor, now, dt):
        plan = self.plans[actor['id']]
        future_t = round(now + dt - plan['created_at_s'], 9)
        if future_t > plan['points'][-1]['t_s'] + 1e-8:
            raise ValueError('Anchor plan expired; request new anchors before advancing')
        times = [p['t_s'] for p in plan['points']]
        xy = np.asarray([p['position_world_m'] for p in plan['points']])
        x, y = [float(np.interp(future_t, times, xy[:, i])) for i in (0, 1)]
        dx, dy = x - actor['x'], y - actor['y']
        speed = math.hypot(dx, dy) / dt
        segment = max(0, min(int(np.searchsorted(times, future_t, side='left'))-1, len(times)-2))
        direction = xy[segment+1]-xy[segment]
        yaw = math.atan2(direction[1], direction[0]) if np.linalg.norm(direction) > .01 else actor['yaw']
        updated = dict(actor, x=x, y=y, yaw=yaw, speed=speed)
        return updated, self.measured_control(actor, updated, dt)

    def measured_control(self, before, after, dt):
        previous = self.previous.get(before['id'], dict(acceleration=0., steering=0.))
        acceleration = (after['speed'] - before['speed']) / dt
        yaw_rate = wrap(after['yaw'] - before['yaw']) / dt
        steering = math.atan2(max(1.8, before['length']*.6)*yaw_rate, after['speed']) if after['speed'] > 1e-7 else 0.
        control = dict(primitive='anchor_follow', acceleration=acceleration, yaw_rate=yaw_rate,
                       lateral_acceleration=after['speed']*yaw_rate,
                       jerk=(acceleration-previous['acceleration'])/dt, steering=steering,
                       steering_rate=(steering-previous['steering'])/dt,
                       control_source='derived_from_executed_poses_not_applied_actuators')
        self.previous[before['id']] = control
        return control
