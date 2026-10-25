import math
import numpy as np


def wrap(x):
    return (x+math.pi) % (2*math.pi)-math.pi


def relative(actor, ego):
    d = np.array([actor['x']-ego['x'], actor['y']-ego['y']])
    c, s = math.cos(ego['yaw']), math.sin(ego['yaw'])
    longitudinal, lateral = float(d @ [c, s]), float(d @ [-s, c])
    dh = abs(wrap(actor['yaw']-ego['yaw']))
    if dh > 2.35:
        relation = 'opposing'
    elif dh > .78:
        relation = 'crossing'
    elif abs(lateral) > 2:
        relation = 'left_adjacent' if lateral > 0 else 'right_adjacent'
    else:
        relation = 'ahead' if longitudinal >= 0 else 'behind'
    return dict(longitudinal_m=round(longitudinal, 3), lateral_m=round(lateral, 3),
                distance_m=round(float(np.linalg.norm(d)), 3), heading_delta_rad=round(dh, 3), relation=relation)


def corners(a, margin=0.):
    c, s = math.cos(a['yaw']), math.sin(a['yaw'])
    body = np.array([[1,1],[1,-1],[-1,-1],[-1,1]]) * np.array([a['length']/2+margin, a['width']/2+margin])
    return body @ np.array([[c,s],[-s,c]]) + [a['x'], a['y']]


def overlap(a, b, margin=0.):
    pa, pb = corners(a, margin), corners(b, margin)
    for yaw in [a['yaw'], a['yaw']+math.pi/2, b['yaw'], b['yaw']+math.pi/2]:
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        va, vb = pa@axis, pb@axis
        if va.max() < vb.min() or vb.max() < va.min():
            return False
    return True


def clearance(a, b):
    if overlap(a,b):
        return 0.
    pa, pb = corners(a), corners(b)
    result = float('inf')
    for points, polygon in [(pa,pb),(pb,pa)]:
        for p in points:
            for q,r in zip(polygon,np.roll(polygon,-1,axis=0)):
                d=r-q
                t=np.clip(np.dot(p-q,d)/max(np.dot(d,d),1e-12),0,1)
                result=min(result,float(np.linalg.norm(p-q-t*d)))
    return result


def swept_overlap(a0,a1,b0,b1):
    # Adaptive sub-frame overlap sampling: <=0.2 m translational motion per sample.
    distance=max(math.hypot(a1['x']-a0['x'],a1['y']-a0['y']),math.hypot(b1['x']-b0['x'],b1['y']-b0['y']))
    n=max(5,int(math.ceil(distance/.2)))
    for t in np.linspace(0,1,n+1):
        pair=[]
        for old,new in [(a0,a1),(b0,b1)]:
            pair.append(dict(new,x=old['x']+t*(new['x']-old['x']),y=old['y']+t*(new['y']-old['y']),yaw=old['yaw']+t*wrap(new['yaw']-old['yaw'])))
        if overlap(*pair):
            return float(t)
    return None


def project(points, xy):
    p=np.asarray(points,float)[:,:2]
    d=np.diff(p,axis=0)
    lengths=np.linalg.norm(d,axis=1)
    t=np.clip(np.sum((np.asarray(xy)-p[:-1])*d,axis=1)/np.maximum(lengths**2,1e-12),0,1)
    q=p[:-1]+t[:,None]*d
    index=int(np.argmin(np.linalg.norm(q-np.asarray(xy),axis=1)))
    arc=np.r_[0,np.cumsum(lengths)]
    return float(arc[index]+t[index]*lengths[index]),q[index],math.atan2(d[index,1],d[index,0]),float(np.linalg.norm(q[index]-xy))


def point_at(points, distance):
    p=np.asarray(points,float)[:,:2]
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    s=np.clip(distance,0,arc[-1])
    i=min(int(np.searchsorted(arc,s,side='right')-1),len(p)-2)
    t=(s-arc[i])/max(arc[i+1]-arc[i],1e-9)
    xy=p[i]+t*(p[i+1]-p[i])
    return xy,math.atan2(p[i+1,1]-p[i,1],p[i+1,0]-p[i,0])
