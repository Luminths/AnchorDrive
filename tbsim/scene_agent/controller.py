"""Reactive behavior commands -> bicycle-model state updates.

The Stage-1 agent owns behavior choice in the closed loop.  This adapter does
not clip acceleration, jerk, steering, steering rate, lateral acceleration, or
maximum speed.  It only converts the requested behavior into the state format
used by the simulator.
"""
import math
import numpy as np
from .geometry import project, point_at, relative, wrap


class BehaviorController:
    def __init__(self, config):
        self.config=config
        self.previous={}
        self.progress={}
        self.events=[]

    def set_program(self, participants, now, preserve=True):
        old=self.progress if preserve else {}
        self.progress={}
        for p in participants:
            aid=p['actor_id']
            if aid in old and old[aid]['program']==p:
                self.progress[aid]=old[aid]
            else:
                self.progress[aid]=dict(program=p,index=0,stage_since=None,plan_since=now,done=False)

    def command(self, actor, ego, now, routes):
        runtime=self.progress[actor['id']]
        p=runtime['program']
        dependency=p.get('after_role')
        waiting=now<runtime['plan_since']+p.get('delay_s',0)
        if dependency:
            waiting=waiting or not any(r['program']['role']==dependency and r['done'] for r in self.progress.values())
        if waiting:
            aligned=[]
            for rid,route in routes.items():
                _,_,h,d=project(route['points'],[actor['x'],actor['y']])
                if abs(wrap(h-actor['yaw']))<.7:aligned.append((d,rid))
            rid=min(aligned)[1] if aligned else p['stages'][0]['route_id']
            return dict(primitive='route_follow',route_id=rid,target_speed_mps=actor['speed'])
        if runtime['done']:
            return dict(primitive='route_follow',route_id=p['stages'][-1]['route_id'],target_speed_mps=p['stages'][-1]['target_speed_mps'])
        if runtime['stage_since'] is None:
            runtime['stage_since']=now
        stage=p['stages'][runtime['index']]
        if now-runtime['stage_since'] >= stage['duration_s']-1e-8:
            self.events.append(dict(time_s=now,actor_id=actor['id'],role=p['role'],event='stage_duration_complete',stage=runtime['index']))
            runtime['index']+=1
            runtime['stage_since']=now
            if runtime['index']>=len(p['stages']):
                runtime['done']=True
                stage=p['stages'][-1]
            else:
                stage=p['stages'][runtime['index']]
        if stage['primitive']=='intercept':
            return dict(stage,remaining_time_s=max(self.config.get('dt',.1),stage['deadline_s']-now))
        return stage

    def step(self, actor, ego, command, routes, dt):
        route=routes[command['route_id']]['points']
        s,_,_,_=project(route,[actor['x'],actor['y']])
        lookahead=max(5.,actor['speed']*.8)
        target,heading=point_at(route,s+lookahead)
        offset=command.get('offset_m',0.) if command['primitive']=='lateral_offset' else 0.
        if command['primitive'] in ('align_ego_lateral','side_overlap','staged_cut_in'):
            _,ego_nearest,ego_heading,_=project(route,[ego['x'],ego['y']])
            offset=-(ego['x']-ego_nearest[0])*math.sin(ego_heading)+(ego['y']-ego_nearest[1])*math.cos(ego_heading)
            if command['primitive']=='side_overlap':
                offset+= (1. if command['side']=='left' else -1.)*command['target_lateral_separation_m']
            elif command['primitive']=='staged_cut_in':
                longitudinal=relative(actor,ego)['longitudinal_m']
                separation=(command['approach_lateral_separation_m']
                    if longitudinal < -command['trigger_gap_m'] else
                    command['contact_lateral_separation_m'])
                offset+=(1. if command['side']=='left' else -1.)*separation
        target=target+offset*np.array([-math.sin(heading),math.cos(heading)])
        alpha=wrap(math.atan2(target[1]-actor['y'],target[0]-actor['x'])-actor['yaw'])
        wheelbase=max(1.8,actor['length']*.6)
        steering=math.atan2(2*wheelbase*math.sin(alpha),max(lookahead,1.))
        desired=command.get('target_speed_mps',0.)
        if command['primitive']=='intercept':
            remaining=max(0.,command['target_route_s_m']-s)
            desired=remaining/command['remaining_time_s']
            # Prevent crossing the selected point in one integration step.
            desired=min(desired,remaining/dt)
        elif command['primitive']=='hold':
            desired=0.
        elif command['primitive']=='gap':
            error=command['target_gap_m']-relative(actor,ego)['longitudinal_m']
            desired=max(0.,float(ego['speed']+.5*error))
        previous=self.previous.get(actor['id'],dict(acceleration=0.,steering=0.))
        acceleration=float((desired-actor['speed'])/dt)
        speed=max(0.,float(actor['speed']+acceleration*dt))
        acceleration=(speed-actor['speed'])/dt
        yaw_rate=speed*math.tan(steering)/wheelbase
        yaw_mid=actor['yaw']+.5*yaw_rate*dt
        new=dict(actor,x=actor['x']+speed*math.cos(yaw_mid)*dt,y=actor['y']+speed*math.sin(yaw_mid)*dt,
                 yaw=wrap(actor['yaw']+yaw_rate*dt),speed=speed)
        control=dict(acceleration=acceleration,steering=steering,yaw_rate=yaw_rate,
                     lateral_acceleration=speed*yaw_rate,jerk=(acceleration-previous['acceleration'])/dt,
                     steering_rate=(steering-previous['steering'])/dt,primitive=command['primitive'],route_id=command['route_id'])
        self.previous[actor['id']]=control
        return new,control
