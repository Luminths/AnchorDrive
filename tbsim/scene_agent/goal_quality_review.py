"""Independent, evidence-based LLM review of a construction proposal."""
import json

SYSTEM="You are an independent goal feasibility reviewer. Check only the current initial facts, passage space, and qualitative concept; do not plan waypoints, and do not accept or guarantee success. Measured vehicles, directed lanes, and images take precedence over the rationale; the EGO's tentative IDM prediction and old execution are not guarantees of the future.\nRead each vehicle's x/y, speed, and heading, then check the goal claims. Movement cannot be written as initially stationary, and rear cannot be written as front; own left/right is determined by the measured heading. The left/right unit vectors of construction_geometry only represent the current geometric direction; check them together with the EGO's tentative path, and do not treat a direction away from it as an intrusion. Reject concepts that must cross a clear isolation/disconnected road surface, clearly lack turning time, require a U-turn/reversing/lateral sliding to complete, or depend on EGO cooperation. Do not suggest modifying existing vehicles' initial states; if the layout is unsuitable, change the participating vehicles or legally respawn them.\nSame-direction same-lane straight rear-end collisions directly from behind are prohibited, and cutting in front of the EGO and stopping to wait for a rear-end collision is prohibited. An adjacent-lane vehicle may approach continuously from the side-rear and then cut diagonally, as long as the initial relationship is truthful, the road is continuous, and there is reasonable arrival and turning time; \"center is behind\" cannot be directly treated as a same-lane rear-end collision. Reasonable deceleration waiting/starting early to prepare may pass, and a one-second collision is not required; a stationary vehicle cannot wait until the EGO is already side by side before instantly starting and moving laterally. Auxiliary vehicles need not be added just to make up numbers, and they must not block the main attack corridor in advance or depend on a blocked EGO reaching a distant location.\nmapped_surface_geometry is a measurement of a straight line passing outside the map-annotated surface, not a physical obstacle or proof of an executable route. Adjacent-lane lateral movement does not require a C connection, but there must be continuous road surface and full vehicle-body space; intersections may be reached via a truly continuous curved road. Unknown separations cannot be fabricated, and clear separations cannot be crossed directly. BG collisions are allowed, and precise parts, specific times, or guaranteed EGO reactions are not required. Insufficient evidence to guarantee the future is not the same as a contradiction; if there is a clear contradiction in facts/direction/passage/time, then feasible=false.\nOutput only JSON: initial_fact_checks as an array per vehicle (actor_id, measured_xy_speed_heading, goal_claim, comparison in short English); then feasible as a boolean, issues as a short English array, and revision_advice as a short English string. Do not point out a clear contradiction and still pass; passing only means it is initially credible, and the actual result is judged by the closed loop and independent acceptance."


def review_goal(toolkit,language_goal,rationale,participants):
    from .client import QwenClient
    if not hasattr(toolkit,'goal_quality_client'):
        c=dict(toolkit.world.config)
        c.update(response_max_tokens=1400,enable_thinking=False,max_requests=8,max_total_tokens=150000)
        c['model']=c.get('goal_quality_model',c['model'])
        toolkit.goal_quality_client=QwenClient(c,toolkit.root/'goal_quality_calls.jsonl')
    client=toolkit.goal_quality_client
    vehicles=toolkit.get_vehicles(['EGO']+[p['actor_id'] for p in participants])['vehicles']
    ids=list(dict.fromkeys(v['current_lane']['lane_id'] for v in vehicles if v.get('current_lane')))
    lanes=[toolkit.get_lane(lid) for lid in ids]
    forecast=toolkit.world.forecast_ego(horizon_s=toolkit.world.config['max_simulation_s'])
    from .construction_cues import construction_cues,surface_cues
    for v in vehicles:
        if v['actor_id']=='EGO':continue
        v['construction_geometry']=construction_cues(toolkit.world.states[toolkit._actor(v['actor_id'])],
            toolkit.world.states['ego'],forecast['samples'])
        v['mapped_surface_geometry']=surface_cues(toolkit.world,toolkit.world.states[toolkit._actor(v['actor_id'])],forecast['samples'])
    payload=dict(goal_text=language_goal,rationale=rationale,participants=participants,vehicles=vehicles,lanes=lanes,
        execution_budget_s=toolkit.world.config['max_simulation_s'],ego_model='IDM .1s closed loop',
        ego_nominal_forecast=[{k:p[k] for k in ('t_s','ego_frame_xy_m','speed_mps','yaw_change_rad')}
            for p in forecast['samples'] if abs(p['t_s']-round(p['t_s']))<1e-6],
        forecast_note="Only current online inference, does not read recorded EGO future. Attack changes response, cannot be treated as actual success.")
    image=None
    for row in reversed(toolkit.trace):
        obs=row.get('observation') or {}
        if row.get('tool')=='view_scene' and obs.get('ok') and obs.get('image_name'):
            image=toolkit.root/obs['image_name'];break
    if image is None and (toolkit.root/'goal_selection.png').exists():image=toolkit.root/'goal_selection.png'
    sent_image=image if toolkit.world.config.get('goal_quality_send_image',True) else None
    system=SYSTEM
    if toolkit.world.config.get('physical_waypoint_tracking'):
        from tbsim.utils.control_projection import DEFAULT_LIMITS
        payload['execution_physics']=dict(DEFAULT_LIMITS)
        system+="\nThe execution vehicles in this group move continuously according to execution_physics. In particular, verify the timing of starting from rest and lateral movement: you cannot simply assume immediate reachability using maximum speed multiplied by time; you need to accelerate gradually from the measured speed, and cannot turn in place when stationary. When there is not enough lead time in the same-direction adjacent lane, you cannot write that you start from zero only after waiting for the EGO to be side by side; before a high-speed oncoming vehicle passes, you need to leave actual lateral movement time. Judge whether the initial conception is credible according to these bounds; do not plan numerical actions, and do not require guaranteeing a collision."
    if sent_image is None:
        system+="\nThis request contains only textual numerical values and directed road geometry; no image was sent. Check heading and intrusion according to the numerical direction components; unknown isolation cannot be used to assert presence/absence out of thin air."
    number=client.requests+1
    path=toolkit.root/('goal_quality_%02d.json'%number)
    request=dict(system=system,payload=payload,image_name=sent_image.name if sent_image else None,
                 preview_image_name=image.name if image else None,model=client.config['model'])
    path.write_text(json.dumps(dict(request=request),ensure_ascii=False,indent=2))
    verdict=client.complete(system,payload,image_path=sent_image)
    if isinstance(verdict,dict) and 'revision_advice' not in verdict:
        # Optional advice is not an approval criterion. Keep the raw response in
        # provider logs and preserve feasible=false without inventing advice.
        verdict=dict(verdict,revision_advice='')
    if not isinstance(verdict,dict) or type(verdict.get('feasible')) is not bool or not isinstance(verdict.get('issues'),list) or not isinstance(verdict.get('revision_advice'),str):
        raise ValueError('Goal feasibility reviewer returned malformed verdict')
    path.write_text(json.dumps(dict(request=request,verdict=verdict),ensure_ascii=False,indent=2))
    return verdict
