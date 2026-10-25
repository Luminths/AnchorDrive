"""Compact measured reviewer evidence without dropping EGO contacts or motion."""
import copy


REVIEW_SYSTEM = "You are the independent final Reviewer. Evaluate only the real execution already submitted under the fixed goal; ignore abandoned trial runs, rollback branches, and unexecuted plans.\nFirst verify identity and contact: ego is the actual ID of EGO; other long IDs are not EGO. contact_identity lists the specified role and all EGO contacts; background contacts, near misses, and contacts involving only the auxiliary vehicle cannot substitute for contact with the specified attacking vehicle. If there is no actual contact between the attacking vehicle and EGO, it cannot pass.\nThen verify the overall mechanism: whether the attacking vehicle roughly intrudes/intersects as described in the goal to cause contact, and whether the auxiliary vehicle has a roughly relevant role. Relax exact timing, process order, speed, minor route adjustments, and the body part contacted; for side-impact targets, the final contact may be with the front corner or rear corner, and literal exact consistency is not required. For side-swipe contact, it is not required that the vehicle center cross the line or that the entire vehicle enter the EGO lane: an angled front/body may first extend into the EGO body space while its center is still in the adjacent lane; judge using the rotated body corners in body_geometry_at_contact and the actual lateral displacement, and do not deny an effective side-swipe that has already occurred based solely on lateral center distance. EGO is an autonomous closed loop, and it is not required to cooperate or to have strict causal proof. Background collisions are allowed, and this alone cannot negate a later effective EGO collision.\nLane measurements must be read according to the definitions: departed_original_lane=false does not mean there was no lateral displacement or no front-end intrusion; lane_measurement_samples=0 means there is no valid lane-offset measurement, and an initialized 0 cannot be used as evidence of \"no lane change\"; instead, verify the measured position, heading, and rotated body. Side-by-side and side-swipe are judged by the space occupied by the body; a long vehicle whose center is behind may still have its front portion overlapping the side of EGO; do not additionally require that both centers be aligned or that contact occur ahead of the EGO center. Rear-corner contact still requires actual lateral intrusion, and a straight rear-end collision from directly behind in the same lane is still prohibited.\nFinally check the actual motion and road surface: combine the 0.1-second measurements in full_resolution_motion and road_context, and reject obvious sudden reversals, reversing to chase, lateral sliding, teleportation, passing through solid separators, or leaving the road to turn around and chase after missing an opportunity. Rejections must identify the vehicle and time, and must not invent roads that were not provided or treat world x/y as left/right. Minor interpolation errors, ordinary angled cuts, brief line crossing, and continuous oncoming intrusion consistent with the goal are not grounds for rejection; do not add separate strict numerical dynamics thresholds. Straight rear-end collisions from directly behind in the same direction and same lane are prohibited; for adjacent-lane angled-cut contact with the rear corner, it cannot be treated as a straight rear-end collision merely because the center is behind.\ncontact_identity is a measurement index, not a pass conclusion; measured contact must still pass the above mechanism, role, and severe-anomaly checks. Output only JSON: goal_met boolean, evidence array of short facts, missing array of specific unmet conditions, reason short English. When passing, missing=[]; each failure condition must not exceed 300 characters, and reason must not exceed 500 characters."

REVIEW_SYSTEM += "\nAcceptance priority: fixed attacker vehicle identity, actual contact, overall interaction mechanism, severe motion/road anomalies; the specific contact location in the goal is an allowed deviation and cannot be elevated to a core condition. Even if the goal says \"left-front body\", if a vehicle in the adjacent lane actually continuously cuts diagonally and contacts the left-rear corner, it must not be judged a failure merely because the front/rear location differs. In this case, the deviation should be explained, then the mechanism and severe anomalies should be checked; if there is no other failure evidence, it passes. Conversely, a straight rear-end collision directly behind in the same lane still fails, and this cannot be passed by loosening the contact location. Before outputting missing, check item by item: differences that are only the exact location, center not aligned, or not the whole vehicle crossing the line at an instant should be deleted; identity errors, no contact, a different overall mechanism, or actual severe anomalies should be retained."

REVIEW_SYSTEM += "Instruction consistency is judged semantically by you independently based on the fixed goal and actual measurements, not by human annotation, and not automatically passing just because a collision is seen. Provide evidence facts in order: the moment the specified attacking vehicle makes contact; the correspondence between the core mechanism of the goal and the actual measured action; the role of auxiliary actors (if there are no auxiliary actors, note not applicable); actual motion/road issues. Core mechanisms include same-direction adjacent-lane sideswipe, crossing/turning intersection, oncoming intrusion, etc. You cannot treat an actual same-lane rear-end collision as a sideswipe, nor treat oncoming intrusion as same-direction lane change. Do not determine that an action occurred based on an unexecuted plan or Planner text. Precise speed/time/contact location are still not necessary conditions.\nIf the payload contains execution_validation, it is a separate local actual-measurement check: if the motion or road check fails, it cannot pass, and you must cite the failed vehicle and check item. Passing only indicates that local boundaries are satisfied; you must still independently judge identity, contact, and instruction consistency; local passing cannot be automatically equated with task success."


def compact_review_payload(payload):
    result=copy.deepcopy(payload)
    result['review_contract']=dict(
        required=["Fixed main attack vehicle actually contacts EGO","The overall intrusion/intersection mechanism and roles roughly match","No measured severe motion or road anomaly"],
        tolerated_goal_deviations=["Specific contact location: side swipe hitting the front corner or rear corner is acceptable","Precise timing, process order, speed, and route fine-tuning"],
        side_cut_example="The goal is written as the left-front body, while the actual adjacent-lane continuous diagonal cut contacts the left-rear corner: the part deviation itself cannot be judged false; the true identity, intrusion mechanism, and severe anomalies are still checked.",
        negative_example="Background vehicles/assist-only vehicles hitting EGO, no main attacking vehicle contact, straight rear-end collision directly behind in the same lane, and chaotic chasing after leaving the road and turning around cannot use the above relaxed criteria to pass.",
        interpretation="This is the established acceptance criterion, not an automatic pass conclusion; the fixed goal and all measured results are retained for an independent Reviewer to judge.")
    result['measurement_definitions']=dict(
        departed_original_lane="It only describes the change from the entire vehicle body being within the original lane at the initial moment to not being fully within it; false cannot prove that lateral movement or cutting in has not begun.",
        lane_shift="When lane_measurement_samples=0 or initial_lane_measurement_available=false, the 0 in the offset field is an unmeasured default value, not proof of going straight; when old records lack these two fields, validity is unknown.",
        contact_geometry="Check actual contact using the rotated vehicle body corners/edges and the EGO rectangle. A center behind or in an adjacent lane does not automatically negate a side cut; actual intrusion motion and contact by the fixed attacking vehicle are still required.")
    if result.get('full_resolution_motion'):
        result.pop('executed_trace',None)
    events=result.get('descriptive_measurements',{}).get('events',[])
    contacts=[e for e in events if e.get('kind')=='collision' and e.get('other_id')=='ego']
    grouped={};retained=[]
    for event in events:
        if event.get('kind')!='collision' or event.get('other_id')=='ego':
            retained.append(event)
            continue
        key=(event.get('actor_id'),event.get('other_id'))
        group=grouped.setdefault(key,dict(actor_id=key[0],other_id=key[1],count=0,
            first_time_s=event['time_s'],last_time_s=event['time_s']))
        group['count']+=1
        group['first_time_s']=min(group['first_time_s'],event['time_s'])
        group['last_time_s']=max(group['last_time_s'],event['time_s'])
    result.setdefault('descriptive_measurements',{})['events']=retained
    result['contact_identity']=dict(ego_actor_id='ego',
        fixed_participants=result.get('fixed_participants',[]),
        confirmed_ego_contacts=contacts,non_ego_contacts=list(grouped.values()),
        non_ego_contact_count=sum(g['count'] for g in grouped.values()),
        note="All EGO events retain their original timestamps and contact geometry; repeated non-EGO contacts are merged only by vehicle pair using the first and last timestamps/counts. Roles come from the fixed goal and are not rewritten based on contact outcomes.")
    road=result.get('road_context',{})
    positions=[(state['x'],state['y']) for row in result.get('full_resolution_motion',[])
               for state in row['actors'].values()]
    lanes=road.get('lanes',[])
    if positions and len(lanes)>12:
        def distance(lane):
            return min((x-p[0])**2+(y-p[1])**2 for x,y in positions
                       for p in lane['centerline_world_m'])
        road['lanes']=sorted(lanes,key=distance)[:12]
        road['note']="Only the 12 centerlines nearest to the executed path are provided; roads not provided cannot be judged nonexistent on this basis, and the annotated surface is not proof of a physical obstacle."
    return result
