"""Track sparse position references with the Stage 2 unicycle limits.

Targets are references, not executable teleports. Feedback is evaluated against
the projected prefix, so a clipped turn continues toward the target next step.
This helper has no simulator or policy cache; trial restoration is automatic.
"""
import math
import numpy as np
import torch

from tbsim.utils.control_projection import DEFAULT_LIMITS, project_controls
from .geometry import wrap


def previous_executed_control(trace, aid, now, dt):
    if len(trace)<2 or abs(trace[-1]['time_s']-now)>1e-6:
        return [0.,0.]
    a,b=trace[-2]['states'][aid],trace[-1]['states'][aid]
    if abs(trace[-1]['time_s']-trace[-2]['time_s']-dt)>1e-6:
        raise ValueError('Physical waypoint tracking requires consecutive native frames')
    return [(b['speed']-a['speed'])/dt,wrap(b['yaw']-a['yaw'])/dt]


def track_waypoints(actor, waypoints, steps, dt, previous=(0.,0.), limits=None):
    cfg=dict(DEFAULT_LIMITS,**(limits or {}))
    knots=np.asarray([[0.,actor['x'],actor['y']]]+[
        [p['step']*dt,*p['position_world']] for p in waypoints],dtype=float)
    if not np.isfinite(knots).all() or np.any(np.diff(knots[:,0])<=0):
        raise ValueError('Tracking references must be finite with increasing time')
    state=torch.tensor([[actor['x'],actor['y'],actor['speed'],actor['yaw']]],dtype=torch.float64)
    prev=torch.tensor([previous],dtype=torch.float64)
    poses=[];controls=[];references=[]
    for i in range(steps):
        t=(i+1)*dt
        segment=min(len(knots)-2,np.searchsorted(knots[:,0],t,side='left')-1)
        segment=max(0,segment)
        left,right=knots[segment],knots[segment+1]
        velocity=(right[1:]-left[1:])/(right[0]-left[0])
        target=left[1:]+np.clip((t-left[0])/(right[0]-left[0]),0,1)*(right[1:]-left[1:])
        if t>knots[-1,0]+1e-8:
            # No sudden stop when the supplied plan ends before the horizon.
            target=right[1:]+velocity*(t-right[0])
        references.append(target.tolist())
        current=state[0].numpy()
        # Velocity feed-forward plus position-error feedback. Correcting against
        # the actual projected prefix avoids losing a turn after initial clipping.
        desired=velocity+.8*(target-current[:2]-velocity*dt)
        desired_speed=float(np.linalg.norm(desired))
        desired_yaw=math.atan2(desired[1],desired[0]) if desired_speed>.05 else current[3]
        heading_error=wrap(desired_yaw-current[3])
        desired_speed*=max(0.,math.cos(heading_error))
        raw=torch.tensor([[[(desired_speed-current[2])/.5,heading_error/.5]]],dtype=torch.float64)
        control=project_controls(raw,state,prev,dt,cfg)[0,0]
        speed=current[2]+float(control[0])*dt
        yaw=current[3]+float(control[1])*dt
        x=current[0]+speed*math.cos(yaw)*dt
        y=current[1]+speed*math.sin(yaw)*dt
        pose=[x,y,speed,yaw];poses.append(pose);controls.append(control.tolist())
        state=torch.tensor([pose],dtype=torch.float64);prev=control[None]
    poses=np.asarray(poses)
    deviation=np.linalg.norm(poses[:,:2]-references,axis=1)
    return poses,dict(method='projected_unicycle_reference_tracking',limits=cfg,
        previous_control=list(previous),controls=controls,
        reference_positions_world=references,max_reference_deviation_m=float(deviation.max()),
        first_second_reference_deviation_m=float(deviation[:round(1/dt)].max()))
