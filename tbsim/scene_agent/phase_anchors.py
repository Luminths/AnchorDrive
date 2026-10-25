"""Retime saved Stage 1 anchors along their original spatial paths.

Only saved execution and the current online EGO forecast are inputs. The
operator adjusts reference timing, not world geometry or executed controls.
"""
import numpy as np
import torch


def retime_anchor_targets(positions, valid, names, reference_ego, reference_paths,
                         ego_forecast, current_states, step_index, actor_ids,
                         dt=.1, max_time_shift_s=1., max_position_shift_m=4.,
                         interaction_radius_m=15., full_radius_m=6., match_radius_m=5.):
    if (not 0<dt or not 0<max_time_shift_s<=2 or not 0<max_position_shift_m<=6
            or not 0<=full_radius_m<interaction_radius_m or match_radius_m<=0):
        raise ValueError('Invalid anchor phase adaptation bounds')
    if positions.shape[:2]!=valid.shape or len(names)!=positions.shape[0]:
        raise ValueError('Anchor phase actor/time shape mismatch')
    original=positions.detach().cpu().numpy().astype(float)
    mask=valid.detach().cpu().numpy()
    adjusted=original.copy();records={}
    ref_steps=np.array(sorted(reference_ego),dtype=float)
    if not len(ref_steps):return positions.clone(),dict(method='bounded_path_retiming',anchors_by_actor={})
    ref_xy=np.array([reference_ego[int(s)] for s in ref_steps],dtype=float)
    max_steps=max_time_shift_s/dt
    if not np.isfinite(ref_xy).all():raise ValueError('Non-finite EGO phase reference')
    for i,aid in enumerate(names):
        if aid not in actor_ids or aid not in reference_paths:continue
        path=reference_paths[aid];steps=np.array(sorted(path),dtype=float)
        xy=np.array([path[int(s)] for s in steps],dtype=float)
        if not np.isfinite(xy).all():raise ValueError('Non-finite actor phase reference')
        current=np.array([current_states[aid]['x'],current_states[aid]['y']])
        # Do not ask a car that already advanced to return to a past path point.
        candidates=np.flatnonzero(abs(steps-step_index)<=3*max_steps)
        if not len(candidates):continue
        k=candidates[np.argmin(np.linalg.norm(xy[candidates]-current,axis=1))]
        if np.linalg.norm(xy[k]-current)>match_radius_m:continue
        floor=steps[k];previous=floor;entries=[]
        def sample(s):return np.array([np.interp(s,steps,xy[:,axis]) for axis in (0,1)])
        for j in np.flatnonzero(mask[i]):
            absolute=step_index+j+1
            if j+1>=len(ego_forecast) or absolute not in reference_ego or absolute not in path:continue
            future=np.array([ego_forecast[j+1]['x'],ego_forecast[j+1]['y']])
            if not np.isfinite(future).all():raise ValueError('Non-finite current EGO forecast')
            candidates=np.flatnonzero(abs(ref_steps-absolute)<=max_steps+1e-7)
            distances=np.linalg.norm(ref_xy[candidates]-future,axis=1)
            # Prefer the original timestamp for stationary/equally close poses.
            k=candidates[np.argmin(distances+1e-6*abs(ref_steps[candidates]-absolute))]
            if np.linalg.norm(ref_xy[k]-future)>match_radius_m:continue
            gap=np.linalg.norm(original[i,j]-np.asarray(reference_ego[absolute]))
            gain=float(np.clip((interaction_radius_m-gap)/(interaction_radius_m-full_radius_m),0,1))
            if gain==0:continue
            target=float(np.clip(absolute+gain*(ref_steps[k]-absolute),steps[0],steps[-1]))
            if floor>absolute+max_steps:continue
            target=max(target,previous)
            target=min(target,absolute+max_steps)
            # Bound spatial change by reducing the time shift ON the path.
            if np.linalg.norm(sample(target)-original[i,j])>max_position_shift_m:
                low,high=0.,1.
                for _ in range(12):
                    middle=(low+high)/2
                    if np.linalg.norm(sample(absolute+middle*(target-absolute))-original[i,j])<=max_position_shift_m:low=middle
                    else:high=middle
                target=absolute+low*(target-absolute)
            # An already advanced actor may need more than the allowed change;
            # leave that anchor untouched instead of ordering a backward target.
            if target<floor-1e-7:continue
            adjusted[i,j]=sample(target);previous=target
            entries.append(dict(step=int(absolute),effective_reference_step=float(target),
                shift_s=float((target-absolute)*dt),gain=gain,
                original_world_xy=original[i,j].tolist(),effective_world_xy=adjusted[i,j].tolist()))
        if entries:records[aid]=entries
    result=torch.as_tensor(adjusted,device=positions.device,dtype=positions.dtype)
    return result,dict(method='bounded_path_retiming',max_time_shift_s=max_time_shift_s,
        max_position_shift_m=max_position_shift_m,anchors_by_actor=records,
        note='World points remain on the saved actor path. Original anchor validity and absolute collision fade times are unchanged.')
