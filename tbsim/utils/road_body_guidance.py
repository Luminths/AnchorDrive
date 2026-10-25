"""Fast differentiable whole-body road loss on a cached signed distance field."""
import numpy as np
import torch
import torch.nn.functional as F


def build_road_field(road, states, device, resolution_m=.5, padding_m=400., weight=12., margin_m=.25):
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy,prepare
    positions=np.array([[s['x'],s['y']] for s in states.values()])
    origin=np.floor((positions.min(axis=0)-padding_m)/resolution_m)*resolution_m
    shape=np.ceil((positions.max(axis=0)+padding_m-origin)/resolution_m).astype(int)+1
    x=origin[0]+np.arange(shape[0])*resolution_m
    y=origin[1]+np.arange(shape[1])*resolution_m
    prepare(road)
    inside=contains_xy(road,x[None,:],y[:,None])
    if not inside.any():raise ValueError('No mapped drivable road in the guidance field')
    signed=(distance_transform_edt(~inside)-distance_transform_edt(inside))*resolution_m
    return dict(signed_distance_m=torch.as_tensor(signed.astype('float32'),device=device),
                origin_world_xy=torch.as_tensor(origin,device=device,dtype=torch.float32),
                resolution_m=float(resolution_m),weight=float(weight),margin_m=float(margin_m),
                source='Cached vector ROAD_AREA and lane polygons, no pedestrian surface')


def body_road_loss(state, params):
    field=params.get('road_body_field')
    if field is None:return state.new_zeros(state.shape[:2])
    batch=params['batch_size'];samples=params['num_samples']
    extent=params['ego_extents'].reshape(batch,1,2).expand(batch,samples,2).reshape(-1,2)
    probes=state.new_tensor([[0,0],[1,1],[1,-1],[-1,1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]])
    offsets=extent[:,None,None,:]/2*probes[None,None]
    c,s=torch.cos(state[...,3:4]),torch.sin(state[...,3:4])
    body=state[...,:2,None].transpose(-1,-2)+torch.stack(
        [c*offsets[...,0]-s*offsets[...,1],s*offsets[...,0]+c*offsets[...,1]],dim=-1)
    transform=params['world_from_agent']
    world=torch.einsum('btpi,bij->btpj',body,transform[:,:2,:2].transpose(-1,-2))+transform[:,None,None,:2,2]
    sdf=field['signed_distance_m'];height,width=sdf.shape
    pixels=(world-field['origin_world_xy'])/field['resolution_m']
    denom=state.new_tensor([width-1,height-1])
    grid=2*pixels/denom-1
    # One field is shared across actors/candidates; expand is a view, not a copy.
    distances=F.grid_sample(sdf[None,None].expand(state.shape[0],1,height,width),
                            grid,mode='bilinear',padding_mode='border',align_corners=True)[:,0]
    outside_grid=(grid.abs()-1).clamp_min(0)*denom*field['resolution_m']/2
    violation=(distances+field['margin_m']).clamp_min(0)+torch.linalg.vector_norm(outside_grid,dim=-1)
    return violation.square().mean(dim=-1)*field['weight']
