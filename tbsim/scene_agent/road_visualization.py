"""Map evidence and proposed body poses for display only; no action gating."""
import copy
import json
from functools import lru_cache
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .anchor_controller import AnchorController


@lru_cache(maxsize=4)
def drivable_geometry(map_path):
    """Read the annotated drivable layer, preserving holes and joining overlaps."""
    raw = json.loads(Path(map_path).read_text())
    nodes = {r['token']: (r['x'], r['y']) for r in raw['node']}
    polygons = {r['token']: r for r in raw['polygon']}
    tokens = {t for area in raw['drivable_area'] for t in area['polygon_tokens']}
    shapes = []
    for token in tokens:
        row = polygons[token]
        shape = Polygon([nodes[t] for t in row['exterior_node_tokens']],
                        [[nodes[t] for t in hole['node_tokens']] for hole in row.get('holes', [])])
        if not shape.is_valid:
            shape = shape.buffer(0)
        if not shape.is_empty:
            shapes.append(shape)
    return unary_union(shapes)


def polygon_parts(geometry):
    if geometry.geom_type == 'Polygon':
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def preview_body_poses(actor, plan, dt=.1):
    """Use the direct executor's poses, including stopped segments and yaw jumps."""
    if 'body_poses' in plan:
        # Legacy execution supplies its parsed action poses. Do not reconstruct
        # headings with AnchorController's different convention.
        return [(sample['t_s'],copy.deepcopy(sample['pose_world'])) for sample in plan['body_poses']]
    controller = AnchorController()
    controller.set_plans({actor['id']: plan})
    state = dict(actor)
    poses = [(0., state)]
    for step in range(1, round(plan['points'][-1]['t_s'] / dt) + 1):
        state, _ = controller.step(state, plan['created_at_s'] + (step-1)*dt, dt)
        poses.append((round(step*dt, 8), state))
    return poses
