"""Export execution results and SafeSim-compatible trajectory arrays."""
import json
from pathlib import Path
import numpy as np


def export_episode(toolkit,result):
    root=toolkit.root
    toolkit.write('result.json',result)
    if not toolkit.trace:return
    import h5py
    ids=list(toolkit.trace[0]['states'])
    trace=toolkit.trace
    with h5py.File(root/'data.hdf5','w') as f:
        g=f.create_group(toolkit.world.name)
        g.attrs['schema']='anchordrive_stage1_reactive_agent_v2'
        g.attrs['dt']=toolkit.world.dt
        g.attrs['goal_sha256']=toolkit.goal_hash
        g.create_dataset('centroid',data=np.array([[[r['states'][aid]['x'],r['states'][aid]['y']] for r in trace] for aid in ids]))
        g.create_dataset('yaw',data=np.array([[[r['states'][aid]['yaw']] for r in trace] for aid in ids]))
        g.create_dataset('speed',data=np.array([[r['states'][aid]['speed'] for r in trace] for aid in ids]))
        g.create_dataset('extent',data=np.array([[[r['states'][aid]['length'],r['states'][aid]['width'],1.5] for r in trace] for aid in ids]))
        g.create_dataset('agent_id',data=np.array(ids,dtype=h5py.string_dtype()))
        g.create_dataset('controlled',data=np.array([aid in toolkit.monitor.participants for aid in ids]))
        g.create_dataset('time_s',data=[r['time_s'] for r in trace])
