"""Road footprint measurements for LLM review; never an execution gate."""
from shapely.geometry import Polygon
from .geometry import corners


def footprint(actor, road, tolerance_road):
    body = Polygon(corners(actor))
    return dict(outside_fraction=round(body.difference(road).area/max(body.area,1e-9),4),
                tolerance_outside_fraction=round(body.difference(tolerance_road).area/max(body.area,1e-9),4))


def trial_road_evidence(world, trial):
    if not hasattr(world, '_road_union'):
        return dict(available=False,reason='No road geometry provider')
    road=world._road_union()
    if road.is_empty:
        return dict(available=False,reason='Mapped road surface unavailable')
    tolerance=road.buffer(.25)
    result=dict(available=True,source='mapped road areas and lane polygons; outside does not prove a physical barrier',
                tolerance_m=.25,actors={})
    for aid,plan in trial['plans'].items():
        actual=[dict(t_s=round(r['time_s']-trial['start_s'],3),**footprint(r['states'][aid],road,tolerance)) for r in trial['trace']]
        future=[dict(t_s=round(p['t_s']-trial['start_s'],3),**footprint(p['pose'],road,tolerance)) for p in plan if p['t_s']>trial['end_s']+1e-8]
        initial=getattr(world,'initial_states',{}).get(aid) or trial['trace'][0]['states'][aid]
        result['actors'][aid]=dict(initial=footprint(initial,road,tolerance) if initial else None,
            initial_source=initial.get('source'),
            actual_start=actual[0],actual_end=actual[-1],actual_worst=max(actual,key=lambda r:r['outside_fraction']),
            actual_tolerance_worst=max(actual,key=lambda r:r['tolerance_outside_fraction']),
            first_outside_t_s=next((r['t_s'] for r in actual if r['tolerance_outside_fraction']>0),None),
            actual_sample_count=len(actual),future_plan_worst=max(future,key=lambda r:r['outside_fraction']) if future else None)
    return result
