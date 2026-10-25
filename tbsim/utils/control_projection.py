"""Fast batched feasibility projection of clean unicycle controls (SI units).

Each time point is the closest control to the reference within the feasible
interval given the already projected prefix. This is a causal proximal sweep,
not a claim of the globally nearest trajectory. Feasible references are unchanged.
Conservative speed viability envelopes prevent speed clipping from breaking jerk.
"""
import torch

DEFAULT_LIMITS = dict(acceleration_min=-4., acceleration_max=3., jerk_max=6.,
    yaw_rate_max=.5, yaw_acceleration_max=.8, lateral_acceleration_max=4., speed_max=30., curvature_max=.15)


def project_controls(reference, initial_states, previous_controls, dt, limits=None):
    cfg=dict(DEFAULT_LIMITS, **(limits or {}))
    previous=previous_controls
    speed=initial_states[..., 2]
    result=[]
    braking_time=dt+abs(cfg['acceleration_min'])/cfg['jerk_max']
    accelerating_time=dt+cfg['acceleration_max']/cfg['jerk_max']
    for t in range(reference.shape[-2]):
        # These speed envelopes leave enough time to return acceleration to zero
        # under the jerk bound before hitting a speed boundary.
        low=torch.maximum(torch.full_like(speed,cfg['acceleration_min']), -speed/braking_time)
        high=torch.minimum(torch.full_like(speed,cfg['acceleration_max']), (cfg['speed_max']-speed)/accelerating_time)
        low=torch.maximum(low,previous[...,0]-cfg['jerk_max']*dt)
        high=torch.minimum(high,previous[...,0]+cfg['jerk_max']*dt)
        accel=torch.maximum(low,torch.minimum(high,reference[...,t,0]))
        next_speed=speed+accel*dt
        yaw_bound=torch.minimum(torch.full_like(speed,cfg['yaw_rate_max']),
            cfg['lateral_acceleration_max']/next_speed.clamp_min(.1))
        # Reserve a braking margin for yaw-rate reduction when accelerating.
        yaw_bound=torch.minimum(yaw_bound, cfg['lateral_acceleration_max']/
            (next_speed+cfg['acceleration_max']*cfg['yaw_rate_max']/cfg['yaw_acceleration_max']).clamp_min(.1))
        yaw_bound=torch.minimum(yaw_bound,next_speed.clamp_min(0)*cfg['curvature_max'])
        lo=torch.maximum(-yaw_bound,previous[...,1]-cfg['yaw_acceleration_max']*dt)
        hi=torch.minimum(yaw_bound,previous[...,1]+cfg['yaw_acceleration_max']*dt)
        omega=torch.maximum(lo,torch.minimum(hi,reference[...,t,1]))
        previous=torch.stack((accel,omega),-1)
        result.append(previous);speed=next_speed
    return torch.stack(result,-2)


def rollout_controls(initial_states, controls, dt):
    """Endpoint velocity integration: native finite-difference speed matches v.

    Heading follows the integrated yaw rate, including at rest. No segment-based
    heading reconstruction or instantaneous realignment to an anchor is used.
    """
    speed=initial_states[...,2,None]+torch.cumsum(controls[...,0]*dt,dim=-1)
    yaw=initial_states[...,3,None]+torch.cumsum(controls[...,1]*dt,dim=-1)
    velocity=torch.stack((speed*torch.cos(yaw),speed*torch.sin(yaw)),-1)
    xy=initial_states[...,:2].unsqueeze(-2)+torch.cumsum(velocity*dt,dim=-2)
    return torch.cat((xy,speed.unsqueeze(-1),yaw.unsqueeze(-1)),dim=-1)


def project_guided_controls(action, current_states, data, dt, stage='clean'):
    cfg=(data or {}).get('control_projection')
    if cfg is None:
        return action
    previous=cfg['previous_controls'].repeat_interleave(data['num_samples'],dim=0)
    limits=dict(DEFAULT_LIMITS, **cfg.get('limits', {}))
    backend=cfg.get('backend', 'torch')
    if backend not in ('torch','triton'):
        raise ValueError('Unknown control projection backend: '+str(backend))
    with torch.no_grad():
        if backend=='triton' and action.is_cuda:
            from .control_projection_triton import project_controls_cuda
            projected=project_controls_cuda(action.detach(),current_states,previous,dt,limits)
        else:
            backend='torch'
            projected=project_controls(action.detach(),current_states,previous,dt,limits)
    if 'stats' in cfg:
        stats=cfg['stats']
        stats['calls_by_stage'][stage]=stats['calls_by_stage'].get(stage,0)+1
        stats['calls_by_backend'][backend]=stats['calls_by_backend'].get(backend,0)+1
    # Straight-through derivative keeps later guidance iterations attached to
    # the denoiser while the actual returned values satisfy the hard bounds.
    return projected+(action-action.detach())
