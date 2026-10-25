"""SafeSim component convention and exact full-sample normalized W1.

The acceleration magnitude is projected using world heading, matching the
upstream convention; these are not body-frame vector projections.
"""
import numpy as np
from scipy.stats import wasserstein_distance
KEYS = ('lon_accel', 'lat_accel', 'jerk')
BINS = {k: np.linspace(0., 20. if k == 'jerk' else 10., 21) for k in KEYS}

def values(pos, yaw, dt):
    """Float64 equivalent of SafeSim's imported trajdata calc_stats."""
    if len(pos) < 2:
        raise ValueError('At least two regularly spaced measured positions required')
    vel = np.diff(pos, axis=0, prepend=(2 * pos[0] - pos[1])[None]) / dt
    acc = np.diff(vel, axis=0, prepend=(2 * vel[0] - vel[1])[None]) / dt
    mag = np.linalg.norm(acc, axis=-1)
    jerk = np.diff(mag, prepend=2 * mag[0] - mag[1]) / dt
    return dict(lon_accel=mag * np.cos(yaw), lat_accel=mag * np.sin(yaw), jerk=jerk)

def histogram(x, key):
    if not np.isfinite(x).all():
        raise ValueError('Non-finite motion sample')
    counts = np.histogram(x, BINS[key])[0]
    return counts, dict(samples=len(x), retained=int(counts.sum()),
                        below=int((x < BINS[key][0]).sum()),
                        above=int((x > BINS[key][-1]).sum()))

def hist_w1(left, right, key):
    if left.sum() == 0 or right.sum() == 0:
        return None
    support = BINS[key][:-1]
    return float(wasserstein_distance(support, support, left, right))

def full_w1(samples, reference):
    """Exact empirical W1; no clipping, absolute value or range exclusion."""
    if not len(samples) or not len(reference) or not np.isfinite(samples).all() or not np.isfinite(reference).all():
        raise ValueError('W1 requires nonempty finite samples')
    return float(wasserstein_distance(samples, reference))

def sparse_components(t,p,yaw):
    t,p,yaw = np.asarray(t),np.asarray(p),np.asarray(yaw)
    dt=np.diff(t)
    assert np.all(dt>0)
    v=np.diff(p,axis=0)/dt[:,None]
    tv=(t[:-1]+t[1:])/2
    a=np.diff(v,axis=0)/np.diff(tv)[:,None]
    ta=(tv[:-1]+tv[1:])/2
    mag=np.linalg.norm(a,axis=1)
    # Retain existing SafeSim component convention and endpoint heading.
    return dict(lon_accel=mag*np.cos(yaw[2:]),lat_accel=mag*np.sin(yaw[2:]),
                jerk=np.diff(mag)/np.diff(ta))
