"""Descriptive measurements for the independent language reviewer.

Nothing in this module accepts or rejects a scene. It records what happened;
the independent Qwen reviewer is the sole goal judge.
"""
import copy
import math
from .geometry import clearance, corners, relative, swept_overlap, wrap


def interpolated_pose(start, end, fraction):
    pose=dict(start)
    for key in ('x','y','speed'):
        pose[key]=start[key]+fraction*(end[key]-start[key])
    pose['yaw']=wrap(start['yaw']+fraction*wrap(end['yaw']-start['yaw']))
    return pose


def contact_body_geometry(attacker, ego):
    """Describe the oriented vehicle bodies at first contact in EGO coordinates."""
    c, s = math.cos(ego['yaw']), math.sin(ego['yaw'])
    origin = (ego['x'], ego['y'])
    attacker_corners = []
    for x, y in corners(attacker):
        dx, dy = x-origin[0], y-origin[1]
        attacker_corners.append([round(dx*c+dy*s, 3), round(-dx*s+dy*c, 3)])
    rear_x=-ego['length']/2
    front_a,front_b=attacker_corners[:2]
    edge_intersection_y=None
    if ((front_a[0]-rear_x)*(front_b[0]-rear_x)<=0 and
            abs(front_b[0]-front_a[0])>1e-9):
        fraction=(rear_x-front_a[0])/(front_b[0]-front_a[0])
        edge_intersection_y=front_a[1]+fraction*(front_b[1]-front_a[1])
    return dict(ego_half_length_m=round(ego['length']/2, 3),
                ego_half_width_m=round(ego['width']/2, 3),
                attacker_corners_in_ego_frame_m=attacker_corners,
                attacker_front_edge_crosses_ego_rear_boundary=bool(
                    edge_intersection_y is not None and abs(edge_intersection_y)<=ego['width']/2),
                front_edge_rear_boundary_intersection_lateral_m=(
                    None if edge_intersection_y is None else round(edge_intersection_y,3)),
                interpretation='Coordinates are [forward, left] in EGO frame. Compare the oriented '
                               'attacker corners with the EGO body rectangle; center offsets alone '
                               'do not identify the contacted side.')


class Monitor:
    def __init__(self, goal, states, config, lane_observer=None):
        self.goal=copy.deepcopy(goal)
        self.initial=copy.deepcopy(states)
        self.config=config
        self.participants={p['actor_id'] for p in goal['participants']}
        self.events=[];self.samples=0;self.lane_observer=lane_observer
        self.initial_lanes={aid:lane_observer(aid,states[aid]) for aid in self.participants} if lane_observer else {}
        self.ego_initial_speed_mps=states['ego']['speed']
        self.ego_min_speed_mps=states['ego']['speed'];self.ego_max_speed_mps=states['ego']['speed']
        self.actor_measurements={}
        for aid in self.participants:
            initial_gap=clearance(states[aid],states['ego'])
            self.actor_measurements[aid]=dict(initial_body_clearance_m=initial_gap,
                initial_relative_to_ego=relative(states[aid],states['ego']),
                min_body_clearance_m=initial_gap,max_gap_closure_m=0.,
                max_speed_mps=states[aid]['speed'],min_speed_mps=states[aid]['speed'],
                initial_lane_measurement_available=bool(self.initial_lanes.get(aid)),
                lane_measurement_samples=0,
                max_absolute_lane_offset_change_m=0.,max_rightward_lane_shift_m=0.,
                max_leftward_lane_shift_m=0.,departed_original_lane=False)
        self.metrics=dict(min_clearance_m=min(v['initial_body_clearance_m'] for v in self.actor_measurements.values()),
                          target_collision=False,background_collision=False)

    def update(self, before, after, controls, now, on_road):
        self.samples+=1;ego=after['ego']
        self.ego_min_speed_mps=min(self.ego_min_speed_mps,ego['speed'])
        self.ego_max_speed_mps=max(self.ego_max_speed_mps,ego['speed'])
        target_contact=False
        for aid in sorted(self.participants):
            distance=clearance(after[aid],ego)
            self.metrics['min_clearance_m']=min(self.metrics['min_clearance_m'],distance)
            measured=self.actor_measurements[aid]
            measured['min_body_clearance_m']=min(measured['min_body_clearance_m'],distance)
            measured['max_gap_closure_m']=max(measured['max_gap_closure_m'],measured['initial_body_clearance_m']-distance)
            measured['max_speed_mps']=max(measured['max_speed_mps'],after[aid]['speed'])
            measured['min_speed_mps']=min(measured['min_speed_mps'],after[aid]['speed'])
            if self.lane_observer:
                lane=self.lane_observer(aid,after[aid]);initial=self.initial_lanes.get(aid)
                if lane and initial:
                    measured['lane_measurement_samples']+=1
                    change=lane['signed_center_offset_m']-initial['signed_center_offset_m']
                    measured['current_signed_lane_offset_change_m']=change
                    measured['max_absolute_lane_offset_change_m']=max(measured['max_absolute_lane_offset_change_m'],abs(change))
                    measured['max_rightward_lane_shift_m']=max(measured['max_rightward_lane_shift_m'],-change)
                    measured['max_leftward_lane_shift_m']=max(measured['max_leftward_lane_shift_m'],change)
                    measured['departed_original_lane'] |= initial['footprint_in_original_lane'] and not lane['footprint_in_original_lane']
            for other_id in sorted(after):
                if other_id==aid or other_id in self.participants and other_id<aid:continue
                contact=swept_overlap(before[aid],after[aid],before[other_id],after[other_id])
                if contact is None:continue
                event=dict(id='contact_'+str(len(self.events)),
                    time_s=round(now-self.config['dt']+contact*self.config['dt'],4),
                    kind='collision',actor_id=aid,other_id=other_id)
                if other_id=='ego':
                    attacker_at_contact=interpolated_pose(before[aid],after[aid],contact)
                    ego_at_contact=interpolated_pose(before['ego'],after['ego'],contact)
                    contact_relation=relative(attacker_at_contact,ego_at_contact)
                    event['attacker_relative_to_ego_at_contact']=contact_relation
                    event['body_geometry_at_contact']=contact_body_geometry(attacker_at_contact,ego_at_contact)
                    centered_limit=min(ego_at_contact['width'],attacker_at_contact['width'])/2
                    event['narrow_rear_ban_geometry']=dict(
                        center_behind_rear_half=contact_relation['longitudinal_m']<-ego_at_contact['length']/2+.5,
                        centerline_aligned=abs(contact_relation['lateral_m'])<=centered_limit,
                        heading_aligned=abs(contact_relation['heading_delta_rad'])<=.25,
                        lateral_centerline_threshold_m=round(centered_limit,3),
                        applies=(contact_relation['longitudinal_m']<-ego_at_contact['length']/2+.5 and
                            abs(contact_relation['lateral_m'])<=centered_limit and
                            abs(contact_relation['heading_delta_rad'])<=.25))
                    event['all_attackers_at_contact']={pid:dict(
                        relative_to_ego=relative(interpolated_pose(before[pid],after[pid],contact),ego_at_contact),
                        body_clearance_m=round(clearance(
                            interpolated_pose(before[pid],after[pid],contact),ego_at_contact),3))
                        for pid in sorted(self.participants)}
                self.events.append(event)
                if other_id=='ego':self.metrics['target_collision']=True;target_contact=True
                else:self.metrics['background_collision']=True
        # Background/participant contacts remain evidence, not a stop signal.
        return target_contact

    def report(self):
        return dict(metrics=dict(self.metrics,
                ego_initial_speed_mps=self.ego_initial_speed_mps,
                ego_min_speed_mps=self.ego_min_speed_mps,
                ego_max_speed_mps=self.ego_max_speed_mps),
            events=self.events,actor_measurements=self.actor_measurements,samples=self.samples,
            interpretation=('Measurements describe the executed trajectory and are not pass/fail criteria. '
                'Contact uses sub-frame swept oriented bodies; only participant contact with EGO stops simulation. '
                'Background and participant-to-participant contacts are recorded without stopping. '
                'EGO responses are observations, not controlled actions or causal proof.'))
