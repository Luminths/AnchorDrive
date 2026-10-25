"""Build real-driving W1 reference samples from locally cached nuScenes tracks."""
import argparse
import hashlib
import json
from pathlib import Path
import dill
import numpy as np
import pandas as pd
from tbsim.configs.selected_scene_config import SCENE_IDX_MAP
from tbsim.evaluation.paper_wd import KEYS, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path.home()/'.unified_data_cache')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dt', type=float, default=.1)
    p.add_argument('--first-frame', type=int, default=11)
    p.add_argument('--last-frame', type=int, default=131)
    args = p.parse_args()
    stride = round(args.dt/.1)
    if stride < 1 or not np.isclose(stride*.1, args.dt):p.error('--dt must be a positive multiple of 0.1')
    if args.output.exists():p.error('Output already exists')
    arrays = {k: [] for k in KEYS}; sources = []
    for name, index in sorted(SCENE_IDX_MAP.items(), key=lambda item:item[1]):
        root = args.cache.expanduser()/'nusc_trainval'/name
        meta_file = root/'scene_metadata_dt0.10.dill'
        with meta_file.open('rb') as f:meta = dill.load(f)
        track_file = root/'agent_data_dt0.10.feather'
        df = pd.read_feather(track_file)
        ids = {a.name for a in meta.agents if a.name != 'ego' and a.type.name == 'VEHICLE'}
        df = df[df.scene_ts.between(args.first_frame,args.last_frame) & df.agent_id.isin(ids)]
        for _, track in df.groupby('agent_id'):
            track = track.sort_values('scene_ts')
            for chunk in np.split(np.arange(len(track)),np.flatnonzero(np.diff(track.scene_ts)!=1)+1):
                part = track.iloc[chunk]
                if stride > 1:part = part[part.scene_ts % stride == 0]
                if len(part)<2:continue
                if not np.allclose(np.diff(part.scene_ts)*.1,args.dt):raise ValueError('Nonuniform reference interval')
                v = values(part[['x','y']].to_numpy(float),part.heading.to_numpy(float),args.dt)
                for k in KEYS:arrays[k].append(v[k])
        sources.append(dict(scene_index=index,scene_name=name,sha256=hashlib.sha256(track_file.read_bytes()).hexdigest()))
    packed = {k:np.concatenate(v) for k,v in arrays.items()}
    for a in packed.values():
        if not np.isfinite(a).all() or np.std(a)<=0:raise ValueError('Invalid reference samples')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as f:np.savez_compressed(f,dt=args.dt,**packed)
    args.output.with_suffix('.json').write_text(json.dumps(dict(dt=args.dt,first_frame=args.first_frame,last_frame=args.last_frame,sources=sources,std={k:float(np.std(v)) for k,v in packed.items()}),indent=2))


if __name__ == '__main__':main()
