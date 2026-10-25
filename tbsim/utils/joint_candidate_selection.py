"""Reconcile BG candidates against the trajectories that will actually execute.

This operates on already projected samples. Controlled participants keep their
selected candidate; no new trajectory, future dataset state or collision label
is introduced. Coordinate descent minimizes one shared pairwise objective.
"""
import time

import numpy as np


def box_penetration(poses, other, extent, other_extents, margin=0.0):
    """Oriented-box SAT overlap depth, [samples, other actors, time]."""
    delta = poses[:, None, :, :2] - other[None, :, :, :2]
    a = poses[:, None, :, 2]
    b = other[None, :, :, 2]
    depths = []
    for angle in (a, a + np.pi / 2, b, b + np.pi / 2):
        distance = np.abs(delta[..., 0] * np.cos(angle) + delta[..., 1] * np.sin(angle))
        ra = (extent[0] * np.abs(np.cos(a - angle)) + extent[1] * np.abs(np.sin(a - angle))) / 2
        rb = (other_extents[None, :, None, 0] * np.abs(np.cos(b - angle))
              + other_extents[None, :, None, 1] * np.abs(np.sin(b - angle))) / 2
        depths.append(ra + rb + margin - distance)
    return np.maximum(np.minimum.reduce(depths), 0)


def select_background_candidates(poses, extents, fixed, route_cost, road_cost,
                                 ignored_pairs=None, scene_ids=None, dt=.1,
                                 passes=2, collision_weight=100., margin_m=.25):
    """Choose actor candidates jointly, starting from each actor's prior best.

    Candidate zero contains the original per-actor selection. A BG replacement
    cannot worsen its road-body cost. Controlled actors and EGO are fixed.
    Initial original overlaps and different scenes are excluded symmetrically.
    """
    start = time.perf_counter()
    poses = np.asarray(poses, dtype=np.float64)
    n, samples, horizon, dimensions = poses.shape
    if dimensions != 3 or not np.isfinite(poses).all():
        raise ValueError('Expected finite [actors,samples,time,x/y/yaw] poses')
    if passes < 1 or collision_weight <= 0 or margin_m < 0 or dt <= 0:
        raise ValueError('Invalid joint candidate selection options')
    fixed = np.asarray(fixed, dtype=bool)
    extents = np.asarray(extents)
    road_cost = np.asarray(road_cost)
    route_cost = np.asarray(route_cost)
    if fixed.shape != (n,) or extents.shape != (n, 2) or road_cost.shape != (n, samples) or route_cost.shape != (n, samples):
        raise ValueError('Candidate selection shapes do not match')
    ignore = np.eye(n, dtype=bool)
    if ignored_pairs is not None:
        mask = np.asarray(ignored_pairs, dtype=bool)
        if mask.shape != (n, n) or not np.array_equal(mask, mask.T):
            raise ValueError('Initial pair exclusions must be symmetric')
        ignore |= mask
    if scene_ids is not None:
        ids = np.asarray(scene_ids).reshape(n)
        ignore |= ids[:, None] != ids[None, :]
    # Pair penalties containing no BG cannot be changed by this selector.
    ignore |= fixed[:, None] & fixed[None, :]
    eligible = np.isfinite(route_cost) & np.isfinite(road_cost)
    eligible &= road_cost <= np.maximum(road_cost[:, :1], .02) + 1e-6
    eligible[:, 0] = True
    low = np.min(np.where(eligible, route_cost, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(eligible, route_cost, -np.inf), axis=1, keepdims=True)
    regret = (route_cost - low) / np.maximum(high - low, 1e-6)
    weight = np.exp(-np.maximum((np.arange(horizon) + 1) * dt - 1., 0))
    selected = np.zeros(n, dtype=np.int64)

    def pair_counts(indices):
        chosen = poses[np.arange(n), indices]
        total = early = 0
        for i in range(n):
            js = np.flatnonzero(~ignore[i] & (np.arange(n) > i))
            if js.size:
                contact = box_penetration(chosen[i:i+1], chosen[js], extents[i], extents[js])[0] > 1e-6
                total += int(np.any(contact, axis=1).sum())
                early += int(np.any(contact[:, :(max(1, round(1. / dt)))], axis=1).sum())
        return dict(predicted_incident_pairs=total, predicted_first_interval_pairs=early)

    before = pair_counts(selected)
    updates = 0
    for _ in range(passes):
        changed = False
        for i in np.flatnonzero(~fixed):
            js = np.flatnonzero(~ignore[i])
            if not js.size:
                continue
            chosen = poses[js, selected[js]]
            actual = box_penetration(poses[i], chosen, extents[i], extents[js])
            if not np.any(actual[selected[i]] > 1e-6):
                # Do not disturb an already non-colliding choice just for a
                # minor route or clearance preference.
                continue
            depth = box_penetration(poses[i], chosen, extents[i], extents[js], margin_m)
            # Equal pair weights make local improvements reduce the shared
            # objective. Near-term contact dominates small route differences.
            penalty = (100 * (actual > 1e-6) + actual ** 2 + .01 * depth ** 2) * weight[None, None, :]
            score = collision_weight * penalty.sum(axis=(1, 2)) / horizon + regret[i]
            score[~eligible[i]] = np.inf
            best = int(np.argmin(score))
            if score[best] + 1e-8 < score[selected[i]]:
                selected[i] = best
                changed = True
                updates += 1
        if not changed:
            break
    return selected, dict(before=before, after=pair_counts(selected),
        changed_backgrounds=int(np.count_nonzero(selected)), coordinate_updates=updates,
        selected_sample_indices=selected.tolist(), wall_s=time.perf_counter()-start,
        method='background_coordinate_descent_post_projection')


def reconcile_projected_samples(output, data, calculator, options):
    """Model adapter: use EGO's current forecast, never its diffusion sample."""
    import torch
    from tbsim.utils.guidance_utils import CollisionLossCalculator
    from tbsim.utils.road_body_guidance import body_road_loss
    from tbsim.utils.geometry_utils import transform_points_tensor

    states = output['states']
    n, samples, horizon, _ = states.shape
    flat = states.reshape(n * samples, horizon, -1)
    controls = output['controls'].reshape(n * samples, horizon, -1)
    world = transform_points_tensor(flat[..., :2], data['world_from_agent']).reshape(n, samples, horizon, 2)
    yaw = (flat[..., 3] + data['yaw'].reshape(n * samples, 1)).reshape(n, samples, horizon, 1)
    poses = torch.cat((world, yaw), -1).detach().cpu().numpy()
    ego = torch.as_tensor(data['batch_ego_indices']).bool().cpu().numpy()
    if ego.any():
        indices, xy, heading = CollisionLossCalculator._current_ego_plan_world_poses(data)
        poses[indices.cpu().numpy()] = torch.cat((xy, heading), -1)[:, None].cpu().numpy()
    fixed = torch.as_tensor(data['batch_ctrl_indices']).bool().cpu().numpy() | ego
    route = calculator.loss_calculator_dict['route'].calculate_loss(controls, flat, data)
    route = route.mean(-1).reshape(n, samples).detach().cpu().numpy()
    road = body_road_loss(flat, data).mean(-1).reshape(n, samples).detach().cpu().numpy()
    ignore = data.get('initial_collision_pair_mask')
    if ignore is not None:
        ignore = ignore.cpu().numpy()
    indices, stats = select_background_candidates(poses, data['ego_extents'].cpu().numpy(),
        fixed, route, road, ignored_pairs=ignore,
        scene_ids=data['scene_ids'].reshape(n).cpu().numpy(), dt=data['dt'], **options)
    actor = torch.arange(n, device=states.device)
    chosen = torch.as_tensor(indices, device=states.device)
    for key in ('states', 'controls', 'trajectories'):
        output[key][:, 0] = output[key][actor, chosen].clone()
    for value in output['predictions'].values():
        value[:, 0] = value[actor, chosen].clone()
    output['joint_candidate_selection'] = stats
