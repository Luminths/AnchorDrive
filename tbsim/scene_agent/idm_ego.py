"""Scalar IDM speed control with pure-pursuit lane following, using measured state only."""
import math
import numpy as np
from .geometry import project, point_at, wrap


def step_idm(ego, actors, route, dt, config):
    s, _, _, _ = project(route, [ego['x'], ego['y']])
    speed = max(0., ego['speed'])
    desired = float(config.get('idm_desired_speed_mps', 10.))
    accel = float(config.get('idm_acceleration_mps2', 1.5))
    braking = float(config.get('idm_comfort_braking_mps2', 2.))
    total = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    # Treat the end of the known connected route as a stopped obstacle.
    gap, lead_speed = max(.1, total-s-ego['length']/2), 0.
    for actor in actors:
        if actor['id'] == ego['id']:
            continue
        other_s, _, tangent, lateral = project(route, [actor['x'], actor['y']])
        angle = wrap(actor['yaw']-tangent)
        half_width = (abs(math.cos(angle))*actor['width'] + abs(math.sin(angle))*actor['length'])/2
        half_length = (abs(math.cos(angle))*actor['length'] + abs(math.sin(angle))*actor['width'])/2
        if other_s <= s or lateral > ego['width']/2+half_width:
            continue
        other_gap = max(.1, other_s-s-ego['length']/2-half_length)
        if other_gap < gap:
            gap, lead_speed = other_gap, actor['speed']*math.cos(angle)
    wanted = float(config.get('idm_min_gap_m', 2.)) + max(0., speed*float(config.get('idm_headway_s', 1.5))
                    + speed*(speed-lead_speed)/(2*math.sqrt(accel*braking)))
    acceleration = accel*(1-(speed/desired)**4-(wanted/gap)**2)
    acceleration = float(np.clip(acceleration, -float(config.get('idm_max_braking_mps2', 6.)), accel))
    velocity = max(0., speed+acceleration*dt)
    target, _ = point_at(route, s+max(4., speed*.8))
    alpha = wrap(math.atan2(target[1]-ego['y'], target[0]-ego['x'])-ego['yaw'])
    distance = max(1., float(np.linalg.norm(target-[ego['x'], ego['y']])))
    curvature = 2*math.sin(alpha)/distance
    yaw_delta = velocity*dt*curvature
    middle = ego['yaw']+yaw_delta/2
    return dict(ego, x=ego['x']+velocity*dt*math.cos(middle),
                y=ego['y']+velocity*dt*math.sin(middle), yaw=wrap(ego['yaw']+yaw_delta), speed=velocity)


def plan_idm(states, route, dt, steps, config):
    """Forecast with constant-velocity neighbors; live execution replans each dt."""
    ego = dict(states['ego'])
    neighbors = [dict(a) for key, a in states.items() if key != 'ego']
    samples = [ego]
    for _ in range(steps):
        ego = step_idm(ego, neighbors, route, dt, config)
        samples.append(ego)
        for a in neighbors:
            a['x'] += a['speed']*dt*math.cos(a['yaw'])
            a['y'] += a['speed']*dt*math.sin(a['yaw'])
    return samples
