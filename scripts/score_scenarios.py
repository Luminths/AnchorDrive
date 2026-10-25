"""Score saved rollouts without simulation or API calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import dill
import h5py
import numpy as np
from shapely.prepared import prep
from tbsim.evaluation import paper_geometry as geom
from tbsim.evaluation import paper_wd as wd
from tbsim.scene_agent.roles import is_helper_role


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def average(rows,key):
    v=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def verdict(result, review, folder, goal):
    if 'source_trace_sha256' in review and review['source_trace_sha256']!=sha(folder/'trial_01.jsonl'):
        raise ValueError('Review belongs to a different execution trace')
    if 'goal_sha256' in review and review['goal_sha256']!=goal['goal_sha256']:
        raise ValueError('Review belongs to a different goal')
    status=result.get('status')
    accepted=review.get('accepted',review.get('goal_met'))
    if accepted is None:
        if status=='achieved':accepted=True
        elif status=='failed':accepted=False
        else:raise ValueError('Missing final task verdict')
    origin=review.get('contact_origin',{})
    collision=origin.get('has_new_designated_contact')
    if collision is None:
        collision=review.get('descriptive_measurements',{}).get('metrics',{}).get('target_collision')
    if collision is None:
        collision=result.get('review',{}).get('descriptive_measurements',{}).get('metrics',{}).get('target_collision')
    if collision is None:raise ValueError('Missing target-contact evidence; run Reviewer first')
    return bool(accepted),bool(collision)


def sparse_arrays(folder, attackers):
    """Use only original waypoints within accepted, actually executed intervals."""
    trace=[json.loads(x) for x in (folder/'trial_01.jsonl').read_text().splitlines() if x.strip()]
    actual={round(x['time_s'],6):x['states'] for x in trace};end=max(actual)
    points={a:{} for a in attackers}
    for row in read(folder/'attacker_actions.json'):
        start=row['time_s'];stop=min(start+row['duration_s'],end)
        if stop<=start:continue
        for decision in row.get('legacy_decisions',[]):
            actor=decision.get('actor_id')
            if actor not in points:
                plans=list(row.get('plans_world',{}))
                if len(plans)==1:actor=plans[0]
                else:raise ValueError('Cannot identify original waypoint actor')
            if actor not in points:continue
            if not decision.get('used_llm_output'):raise ValueError('Sparse WD requires executed original LLM waypoints')
            for t in [start,stop]:
                s=actual[round(t,6)][actor];points[actor][round(t,6)]=[s['x'],s['y'],s['yaw']]
            for point in decision['parse_meta']['sparse_waypoints_world']:
                t=round(start+point['step']*.1,6)
                if t>stop+1e-6:continue
                s=actual[t][actor];xy=point['position_world']
                if np.linalg.norm(np.asarray(xy)-[s['x'],s['y']])>1e-3:raise ValueError('Waypoint does not match executed position')
                points[actor][t]=[*xy,s['yaw']]
    arrays=[]
    for actor, rows in points.items():
        if len(rows)<3:continue
        ts=np.array(sorted(rows));v=np.array([rows[t] for t in ts]);arrays.append(wd.sparse_components(ts,v[:,:2],v[:,2]))
    return arrays


def score_scene(folder, review_path, reference, dt, maps, cache, dataset, distribution, sparse):
    result=read(folder/'result.json');goal=read(folder/'goal.json');handoff=read(folder/'handoff.json')
    review=read(review_path) if review_path else result.get('review',{})
    accepted,contact=verdict(result,review,folder,goal)
    with h5py.File(folder/'data.hdf5') as f:
        name=next(iter(f));g=f[name]
        ids=[v.decode() if isinstance(v,bytes) else str(v) for v in g['agent_id'][:]]
        pos,extent,speed=[g[k][:] for k in ['centroid','extent','speed']]
        yaw=g['yaw'][:].reshape(pos.shape[:2]);times=g['time_s'][:]
    if len(times)<2:return dict(accepted=accepted,ego_adv_collision=contact,valid_trajectory=False)
    if not all(np.isfinite(a).all() for a in [pos,extent,speed,yaw,times]) or not np.all(np.diff(times)>0):raise ValueError('Invalid trajectory')
    participants=[];attackers=[]
    for actor in goal['goal']['participants']:
        i=ids.index(handoff['aliases'].get(actor['actor_id'],actor['actor_id']));participants.append(i)
        if not is_helper_role(actor['role']):attackers.append(i)
    if not attackers:raise ValueError('No designated ADV')
    bg=[i for i,a in enumerate(ids) if a!='ego' and i not in participants]
    with (cache/'nusc_trainval'/name/'scene_metadata_dt0.10.dill').open('rb') as f:location=dill.load(f).location
    if location not in maps:
        surface,_=geom.official_drivable_surface(location,dataset);maps[location]=prep(surface.buffer(.25))
    outside=np.zeros(pos.shape[:2],dtype=bool)
    for i in attackers+bg:
        for t in range(len(times)):outside[i,t]=not maps[location].covers(geom.body_polygon(pos[i,t],yaw[i,t],extent[i,t]))
    bg_oob=geom.background_body_oob_summary(outside[bg,0],outside[bg].any(axis=1))
    pairs=geom.collision_pairs(pos,yaw,extent,times)
    inherited={p for p,e in pairs.items() if e['initial_overlap'] and all(handoff['initial_states'][ids[i]].get('source')=='existing' for i in p)}
    remaining=set(pairs)-inherited
    row=dict(accepted=accepted,ego_adv_collision=contact,valid_trajectory=True,scene_name=name,
        adv_bg_collision=any((i in attackers and j in bg) or (j in attackers and i in bg) for i,j in remaining),
        bg_bg_collision=any(i in bg and j in bg for i,j in remaining),
        adv_body_oob=bool(outside[attackers].any()),bg_body_oob_rate=bg_oob['rate'],
        bg_initial_oob_excluded=bg_oob['initial_excluded'],bg_oob_eligible=bg_oob['eligible'],
        inherited_overlap_pairs=len(inherited),simulation_time_s=float(times[-1]),
        simulation_wall_s=result.get('wall_time_s',result.get('wall_s')),
        sampling_wall_s=result.get('sampling_wall_total_s'),reviewer_wall_s=review.get('review_wall_s'))
    peaks={k:[] for k in ['speed','acceleration','jerk','yaw_rate']}
    for i in attackers:
        motion=geom.motion_values(pos[i],times);peaks['speed'].append(float(speed[i].max()))
        peaks['yaw_rate'].append(float(np.max(np.abs(np.diff(np.unwrap(yaw[i]))/np.diff(times)))))
        for k in ['acceleration','jerk']:
            if len(motion[k]):peaks[k].append(float(np.max(motion[k])))
    row.update({'adv_max_'+k:max(v) if v else None for k,v in peaks.items()})
    if sparse:arrays=sparse_arrays(folder,[ids[i] for i in attackers])
    else:
        ix=np.flatnonzero(np.isclose((times-times[0])/dt,np.round((times-times[0])/dt),atol=1e-7))
        if len(ix)>1 and not np.allclose(np.diff(times[ix]),dt,atol=1e-7):raise ValueError('Requested WD interval unavailable')
        arrays=[wd.values(pos[i,ix],yaw[i,ix],dt) for i in attackers] if len(ix)>1 else []
    for k in wd.KEYS:
        samples=np.concatenate([a[k] for a in arrays]) if arrays else np.array([])
        raw=None
        if len(samples):
            if distribution=='all-samples':raw=wd.full_w1(samples,reference[k])
            else:raw=wd.hist_w1(wd.histogram(samples,k)[0],wd.histogram(reference[k],k)[0],k)
        row['raw_w1_'+k]=raw;row['normalized_w1_'+k]=raw/np.std(reference[k]) if raw is not None else None
        row['wd_samples_'+k]=len(samples)
    row['source_sha256']=sha(folder/'data.hdf5')
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path,help='run_150 output or execute_XXX batch directory')
    group.add_argument('--manifest',type=Path,help='JSON rows: scene_index, directory, optional review and pipeline_wall_s; relative paths use manifest directory')
    p.add_argument('--stage',choices=['stage1','stage2'],default='stage2')
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('nuscenes'))
    p.add_argument('--cache',type=Path,default=Path.home()/'.unified_data_cache')
    p.add_argument('--total-scenes',type=int,default=150)
    p.add_argument('--dt',type=float,default=.1)
    p.add_argument('--distribution',choices=['all-samples','histogram'],default='all-samples')
    p.add_argument('--sparse-waypoints',action='store_true',help='Stage 1 sensitivity: use original executed waypoints')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Choose a new output directory')
    if args.total_scenes<1:p.error('--total-scenes must be positive')
    with np.load(args.reference) as ref:
        if not np.isclose(float(ref['dt']),args.dt):p.error('Reference and generated WD intervals must match')
        reference={k:ref[k].copy() for k in wd.KEYS}
    for a in reference.values():
        if not len(a) or not np.isfinite(a).all() or np.std(a)<=0:raise ValueError('Invalid reference distribution')
    entries={}
    if args.manifest:
        for entry in read(args.manifest):
            i=entry['scene_index']
            if i in entries or not 0<=i<args.total_scenes:raise ValueError('Duplicate or out-of-range scene index')
            for key in ['directory','review']:
                if entry.get(key):entry[key]=str((args.manifest.resolve().parent/entry[key]).resolve())
            entries[i]=entry
    else:
        for i in range(args.total_scenes):
            marker=args.run/f'scene_{i:03d}/completion.json'
            entry=dict(scene_index=i)
            if marker.exists():
                completion=read(marker);base=Path(completion['run_directory'])/args.stage
                entry['pipeline_wall_s']=completion.get('elapsed_s')
            else:base=args.run
            entry['directory']=str(base/f'execute_{i:03d}/scene_{i:03d}');entries[i]=entry
    rows=[];maps={};errors=[]
    for i in range(args.total_scenes):
        entry=entries.get(i,{})
        folder=Path(entry.get('directory','__missing__'))
        row=dict(scene_index=i,accepted=False,ego_adv_collision=False,valid_trajectory=False)
        if (folder/'result.json').is_file() and (folder/'data.hdf5').is_file():
            if read(folder/'result.json').get('status')!='error':
                try:row.update(score_scene(folder,entry.get('review'),reference,args.dt,maps,args.cache.expanduser(),args.dataset,args.distribution,args.sparse_waypoints))
                except Exception as exc:errors.append(dict(scene_index=i,error=str(exc)))
        row['pipeline_wall_s']=entry.get('pipeline_wall_s');rows.append(row)
        print(f'Scored {i+1}/{args.total_scenes}',flush=True)
    args.output.mkdir(parents=True)
    if errors:
        (args.output/'errors.json').write_text(json.dumps(errors,indent=2))
        raise RuntimeError('Scoring errors; no aggregate published. Inspect errors.json.')
    quality=[r for r in rows if r['valid_trajectory']]
    keys=['adv_bg_collision','bg_bg_collision','adv_body_oob','bg_body_oob_rate',
          'adv_max_speed','adv_max_acceleration','adv_max_yaw_rate','adv_max_jerk',
          *['normalized_w1_'+k for k in wd.KEYS],*['raw_w1_'+k for k in wd.KEYS],
          'simulation_wall_s','sampling_wall_s','reviewer_wall_s']
    summary=dict(total_scenes=args.total_scenes,valid_trajectories=len(quality),
        task_success_count=sum(r['accepted'] for r in rows),ego_adv_collision_count=sum(r['ego_adv_collision'] for r in rows),
        task_success_rate=sum(r['accepted'] for r in rows)/args.total_scenes,
        ego_adv_collision_rate=sum(r['ego_adv_collision'] for r in rows)/args.total_scenes,
        **{k:average(quality,k) for k in keys},metric_scene_counts={k:sum(r.get(k) is not None for r in quality) for k in keys},
        pipeline_wall_s=average(rows,'pipeline_wall_s'),pipeline_wall_scene_count=sum(r.get('pipeline_wall_s') is not None for r in rows),
        protocol=dict(wd_dt=args.dt,distribution=args.distribution,sparse_waypoints=args.sparse_waypoints,
            reference_sha256=sha(args.reference),reference_std={k:float(np.std(v)) for k,v in reference.items()},
            road_buffer_m=.25,rates='Fractions, not percentages; task/contact over all declared scenes; quality over valid trajectories.',
            initial_exclusions='Initially offroad BG excluded from BG OOB denominator; initially overlapping original vehicle pairs excluded from non-target collisions.',
            runtime='Pipeline time from completion marker; stage wall time and reviewer time reported separately. Missing time remains null. Hardware/concurrency must match for timing comparisons.'))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (args.output/'scene_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
