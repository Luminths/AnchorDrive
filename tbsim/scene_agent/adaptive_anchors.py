"""Bounded interaction-relative correction of offline position anchors.

Only the current independent EGO forecast and saved Stage 1 execution are used.
No unseen dataset future or future closed-loop result is available here.
"""
import numpy as np
import torch


def adapt_anchor_targets(positions, valid, names, reference_ego, ego_forecast, step_index,
                         max_shift_m=2., interaction_radius_m=15., full_radius_m=5.):
    if not 0<max_shift_m<=5. or not 0<=full_radius_m<interaction_radius_m:
        raise ValueError('Invalid bounded anchor adaptation configuration')
    if len(names)!=positions.shape[0] or valid.shape!=positions.shape[:2]:
        raise ValueError('Anchor adaptation actor/time shape mismatch')
    result=positions.clone();records={}
    for i,aid in enumerate(names):
        entries=[]
        for j in torch.nonzero(valid[i],as_tuple=False).flatten().cpu().tolist():
            absolute_step=step_index+j+1
            if absolute_step not in reference_ego or j+1>=len(ego_forecast):continue
            old=np.asarray(reference_ego[absolute_step],dtype=float)
            new=np.asarray([ego_forecast[j+1]['x'],ego_forecast[j+1]['y']],dtype=float)
            anchor=positions[i,j].detach().cpu().numpy().astype(float)
            if not np.isfinite(np.r_[old,new,anchor]).all():
                raise ValueError('Non-finite anchor/EGO reference')
            distance=float(np.linalg.norm(anchor-old))
            gain=float(np.clip((interaction_radius_m-distance)/(interaction_radius_m-full_radius_m),0,1))
            delta=(new-old)*gain
            norm=float(np.linalg.norm(delta))
            if norm>max_shift_m:delta*=max_shift_m/norm
            result[i,j]+=torch.as_tensor(delta,device=positions.device,dtype=positions.dtype)
            entries.append(dict(step=absolute_step,original_world_xy=anchor.tolist(),
                effective_world_xy=result[i,j].detach().cpu().tolist(),shift_xy_m=delta.tolist(),
                stage1_ego_world_xy=old.tolist(),current_ego_forecast_world_xy=new.tolist(),gain=gain))
        if entries:records[aid]=entries
    return result,dict(method='bounded_ego_forecast_translation',max_shift_m=max_shift_m,
        interaction_radius_m=interaction_radius_m,full_radius_m=full_radius_m,anchors_by_actor=records)
