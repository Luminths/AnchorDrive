"""Full-body road, contact and motion metrics used by the paper scorer."""
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

def read_json(path):
    return json.loads(Path(path).read_text())

def body_polygon(pos, yaw, size):
    local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * size[:2] / 2
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return Polygon(local @ rotation.T + pos)

def background_body_oob_summary(initial_oob, ever_oob):
    """Exclude initially outside BG actors from both numerator and denominator.

    An empty eligible population has no defined rate. Do not turn it into a
    zero-OOB observation or remove that scene from other experimental metrics.
    """
    initial, ever = [bool(x) for x in initial_oob], [bool(x) for x in ever_oob]
    if len(initial) != len(ever):
        raise ValueError('Initial and ever-OOB flags must describe the same BG actors')
    if any(start and not later for start, later in zip(initial, ever)):
        raise ValueError('Ever-OOB must include the initial observation')
    eligible = [i for i, start in enumerate(initial) if not start]
    counted = sum(ever[i] for i in eligible)
    return dict(total=len(initial), initial_excluded=sum(initial), eligible=len(eligible),
                counted=counted, rate=counted/len(eligible) if eligible else None,
                scene=bool(counted) if eligible else None,
                including_initial_rate=sum(ever)/len(ever) if ever else None)

def motion_values(pos, times):
    velocity = np.diff(pos, axis=0) / np.diff(times)[:, None]
    vtimes = (times[1:] + times[:-1]) / 2
    acceleration = np.diff(velocity, axis=0) / np.diff(vtimes)[:, None]
    atimes = (vtimes[1:] + vtimes[:-1]) / 2
    jerk = np.diff(acceleration, axis=0) / np.diff(atimes)[:, None]
    return {"velocity": np.linalg.norm(velocity, axis=-1),
            "acceleration": np.linalg.norm(acceleration, axis=-1),
            "jerk": np.linalg.norm(jerk, axis=-1)}

def official_drivable_surface(location, dataset_path=None):
    """Load raw annotated road surface, including holes and None-token areas.

    Cached polylines can acquire self-intersections from coordinate rounding.
    Discarding those shapes removes large real roads in two Singapore maps.
    Raw annotations are the scoring source; no lane approximation is added.
    """
    roots = ([Path(dataset_path)] if dataset_path is not None else
             [Path.cwd() / "nuscenes"] +
             [parent / "nuscenes" for parent in Path(__file__).resolve().parents])
    candidates = [root / "maps" / "expansion" / (location + ".json") for root in roots]
    filename = next((path for path in candidates if path.is_file()), None)
    if filename is None:
        raise FileNotFoundError("Raw nuScenes drivable_area map required; provide dataset_path. "
                                "Do not silently fall back to the lossy cached road surface.")
    raw = read_json(filename)
    nodes = {row["token"]: (row["x"], row["y"]) for row in raw["node"]}
    polygons = {row["token"]: row for row in raw["polygon"]}
    tokens = {token for area in raw["drivable_area"] for token in area["polygon_tokens"]}
    parts = []
    for token in tokens:
        row = polygons[token]
        poly = Polygon([nodes[t] for t in row["exterior_node_tokens"]],
                       [[nodes[t] for t in hole["node_tokens"]] for hole in row.get("holes", [])])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_valid or poly.is_empty:
            raise ValueError("Unusable drivable_area polygon: %r in %s" % (token, filename))
        parts.append(poly)
    road = unary_union(parts)
    if road.is_empty or not road.is_valid:
        raise ValueError("Invalid or empty annotated road surface: %s" % filename)
    return road, filename

def collision_pairs(pos, yaw, extent, times):
    """Bounding-circle broad phase, then exact oriented-body intersections."""
    n, t, _ = pos.shape
    radius = np.linalg.norm(extent[:, :, :2], axis=-1) / 2
    delta = pos[:, None] - pos[None, :]
    candidate = (np.sum(delta * delta, axis=-1) <= (radius[:, None] + radius[None, :]) ** 2)
    candidate &= np.triu(np.ones((n, n), dtype=bool), 1)[:, :, None]
    candidates = np.argwhere(candidate)
    pairs = {}
    polygons = {}
    for i, j, k in candidates:
        pair = (int(i), int(j))
        for a in [i, j]:
            if (a, k) not in polygons:
                polygons[a, k] = body_polygon(pos[a, k], yaw[a, k], extent[a, k])
        if polygons[i, k].intersects(polygons[j, k]):
            if pair not in pairs:
                pairs[pair] = {"first_time_s": float(times[k]), "after_initial": False,
                               "initial_overlap": bool(k==0),"last_contact_frame":int(k),
                               "contact_episodes":0,"recontact_after_initial_separation":False}
            if k==0 or k>pairs[pair]['last_contact_frame']+1 or pairs[pair]['contact_episodes']==0:
                pairs[pair]['contact_episodes']+=1
                if k>0 and pairs[pair]['initial_overlap']:
                    pairs[pair]['recontact_after_initial_separation']=True
            pairs[pair]['last_contact_frame']=int(k)
            pairs[pair]["after_initial"] |= k > 0
    return pairs
