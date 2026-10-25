"""Schema validation for one fixed goal and reactive attacker actions."""
import copy
import hashlib
import json
import math
import re

PRIMITIVES=['speed','gap','lane_change','lateral_offset','align_ego_lateral','side_overlap','staged_cut_in','route_follow','intercept','hold']
BASE_ACTION={'actor_id','primitive','route_id'}
LEGACY_ACTION=BASE_ACTION|{'target_speed_mps','target_gap_m','offset_m'}
ACTION_FIELDS={
    'speed':{'target_speed_mps'},'route_follow':{'target_speed_mps'},
    'lane_change':{'target_speed_mps'},
    'gap':{'target_gap_m'},
    'lateral_offset':{'target_speed_mps','offset_m'},
    'align_ego_lateral':{'target_speed_mps'},
    'side_overlap':{'target_speed_mps','side','target_lateral_separation_m'},
    'staged_cut_in':{'target_speed_mps','side','approach_lateral_separation_m',
                     'contact_lateral_separation_m','trigger_gap_m'},
    'intercept':{'target_forward_distance_m','target_arrival_time_s'},
    'hold':set(),
}


def exact(value, keys, label):
    if not isinstance(value,dict) or set(value)!=set(keys):
        raise ValueError(label+' requires exactly '+', '.join(sorted(keys)))


def number(value, low, high, label):
    if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError('{} must be finite in [{},{}]'.format(label,low,high))
    return float(value)


def finite_number(value, label, nonnegative=False):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(label+' must be a finite number')
    if nonnegative and value<0:raise ValueError(label+' must be nonnegative')
    return float(value)


def text(value, label, maximum=1200):
    if not isinstance(value,str) or not 1<=len(value.strip())<=maximum:
        raise ValueError(label+' must be bounded nonempty text')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def requires_body_contact(language):
    """A threatened collision is not a promise that two bodies actually touch."""
    risk_only=re.sub("(?:\u63a5\u89e6|\u78b0\u649e|\u8ffd\u5c3e|\u649e\u51fb)(?:\u7684)?(?:\u98ce\u9669|\u5371\u9669|\u53ef\u80fd\u6027)",
                     "\u98ce\u9669",language,flags=re.IGNORECASE)
    return bool(re.search("\u63a5\u89e6|\u78b0\u649e|\u649e\u4e0a|\u649e\u51fb|\u8ffd\u5c3e|\u522e\u64e6|\u5250\u8e6d|impact|collision|rear.end",
                          risk_only,re.IGNORECASE))


def validate_goal(raw, snapshot, config):
    """Validate one model-authored goal without scoring or diversity rules."""
    exact(raw,{'language_goal','rationale','participants'},'goal')
    text(raw['language_goal'],'language goal',400);text(raw['rationale'],'rationale',1200)
    language=raw['language_goal']
    if re.search("(?:\u63a5\u89e6|\u78b0\u649e|\u8ffd\u5c3e|\u649e\u51fb)(?:\u7684)?(?:\u98ce\u9669|\u5371\u9669|\u53ef\u80fd\u6027)",language) and not re.search(
            "(?:\u8f66\u8eab|\u8f66\u8f86|\u4e24\u8f66)?(?:\u6700\u5c0f)?\u95f4\u9699.{0,10}(?:\u5c0f\u4e8e|\u4f4e\u4e8e|\u4e0d\u5927\u4e8e|\u4e0d\u8d85\u8fc7|\u81f3|\u8fbe\u5230|<|\u2264)\\s*\\d+(?:\\.\\d+)?\\s*(?:\u7c73|m)",
            language,re.IGNORECASE):
        raise ValueError("A vague collision risk is not an observable acceptance target. Specify the attacker maneuver and a measurable body clearance, for example \"body clearance less than 0.5 meters\", or require actual body contact if the route supports it.")
    ego_name="(?:EGO|ego vehicle|host vehicle|subject vehicle)"
    ego_behavior="(?:braking|brake|deceleration|acceleration|lane change|avoidance|stop|stopping|halt|response|reaction)"
    ego_response=("(?:force|require|make|cause).{0,10}"+ego_name+r'|'
                  +ego_name+r'.{0,16}'+ego_behavior+"|(?:is|has|will|will)?"+ego_behavior+".{0,6}\u7684\\s*"+ego_name+r'|'
                  r'(?:force|require)\s+(?:the\s+)?EGO')
    if re.search(ego_response,language,re.IGNORECASE):
        raise ValueError("language_goal must be one observable attacker goal. Remove claims about EGO braking, accelerating, lane changing, or avoiding; put reasoning in rationale. Example: \"spawn_1 cuts in from the right adjacent lane and makes body contact with EGO\"")
    if re.search("(?:\u9884\u6d4b|\u9884\u62a5|\u9884\u8ba1|\u9884\u671f|\u89c4\u5212).{0,8}(?:\u8f68\u8ff9|\u8def\u7ebf|\u8def\u5f84)|(?:predicted|forecast)\\s+(?:trajectory|path|route)",language,re.IGNORECASE):
        raise ValueError('A forecast is planning context, not executed evidence; state the actual attacker maneuver and measured EGO interaction in the language goal')
    if (re.search("(?:\u63a5\u89e6|\u78b0\u649e|\u649e\u4e0a|\u8ffd\u5c3e)",language) and
        re.search("(?:\u524d\u65b9|\u540e\u65b9|\u5de6\u4fa7|\u53f3\u4fa7).{0,4}(?:\u7ea6|\u5927\u7ea6)?\\s*\\d+(?:\\.\\d+)?\\s*\u7c73\u5904",language)):
        raise ValueError("A predicted conflict-point distance is planning context, not a contact condition. Remove \"about N meters ahead\" from the fixed contact goal and keep it in rationale; describe the actual maneuver and contact instead")
    if (re.search("\u63a5\u89e6|\u78b0\u649e|\u649e\u4e0a|\u649e\u51fb|impact|collision",language,re.IGNORECASE) and
        re.search("(?:\u5728|\u4e8e)(?:\u7ea6|\u5927\u7ea6)?\\s*\\d+(?:\\.\\d+)?\\s*\u79d2(?:\u5de6\u53f3)?(?:\u65f6|\u540e|\u5185|\u4e4b\u524d|\u4ee5\u524d)",language)):
        raise ValueError('A predicted collision time belongs in rationale, not in the fixed goal. '
            'EGO replans during execution; describe the observable maneuver and contact without '
            'requiring contact at a guessed second.')
    for entry_side,move,far_side in (("\u53f3","\u5de6","\u5de6"),("\u5de6","\u53f3","\u53f3")):
        if re.search("\u4ece"+entry_side+"\u4fa7.{0,25}(?:\u5411"+move+r'|'+move+"\u53d8\u9053|"+move+"\u8f6c|"+move+"\u5207).{0,45}"+ego_name+far_side+"\u4fa7",language,re.IGNORECASE):
            raise ValueError('A one-step cut-in from one side cannot directly strike EGO on its far side. Name body contact without an unsupported struck side, or describe a genuinely different maneuver')
    for entry_side,move in (("\u53f3","\u5de6"),("\u5de6","\u53f3")):
        if re.search(r'(?:'+entry_side+"\u4fa7\u76f8\u90bb\u8f66\u8f86|\u4ece"+entry_side+"\u4fa7\u76f8\u90bb\u8f66\u9053).{0,70}(?:\u5411"+move+r'|'+move+"\u53d8\u9053|"+move+"\u5207\u5165).{0,70}(?:\u5176|\u8be5\u8f66|\u653b\u51fb\u8f66)\u8f66\u8eab"+entry_side+"\u4fa7",language,re.IGNORECASE):
            raise ValueError('An adjacent attacker cutting toward EGO cannot strike it with '
                'the body side facing away from EGO. Remove the unsupported attacker-side '
                'contact label and request observable body contact instead.')
    actors={actor['id']:actor for actor in snapshot['actors']};participants=raw['participants']
    if not isinstance(participants,list) or not 1<=len(participants)<=config['max_controlled_vehicles']:
        raise ValueError('Need 1..max_controlled_vehicles participants')
    seen=set();roles=set()
    for participant in participants:
        exact(participant,{'actor_id','role'},'goal participant')
        aid=participant['actor_id'];text(participant['role'],'role',80)
        if aid in seen or participant['role'] in roles:raise ValueError('Duplicate actor or role')
        if aid not in actors or not actors[aid]['controllable']:
            raise ValueError('Unknown/uncontrollable actor; EGO is never controllable')
        if aid not in raw['language_goal']:
            raise ValueError('Language goal must explicitly name every controlled actor ID')
        seen.add(aid);roles.add(participant['role'])
    contact=requires_body_contact(language)
    if len(seen)>1 and contact:
        collision_verb="(?:contact|collision|hit|impact|scrape|graze|impact|collision)"
        collective="(?:two vehicles|two cars|two vehicles|multiple vehicles|all attacking vehicles|all attacking vehicles|each vehicle|respectively|respectively|all|all)"
        repeated_contact=len(re.findall("(?:\u4e0e|\u548c|\u78b0\u649e\u4e8e)\\s*(?:EGO|\u81ea\u8f66|\u672c\u8f66|\u4e3b\u8f66).{0,12}"+collision_verb,
            language,re.IGNORECASE))>1
        if (re.search(collective+".{0,45}(?:EGO|\u81ea\u8f66|\u672c\u8f66|\u4e3b\u8f66).{0,16}"+collision_verb,
                language,re.IGNORECASE) or repeated_contact):
            raise ValueError('The simulator stops at the first body contact. A multi-car goal '
                'cannot require every attacker to collide with EGO. Give one attacker the '
                'primary contact and each support car a separate observable role completed '
                'before that contact, or choose a non-contact cooperative goal.')
        if not re.search("(?:\u9996\u6b21)?(?:\u63a5\u89e6|\u78b0\u649e|\u649e\u51fb)(?:\u53d1\u751f)?(?:\u4e4b)?\u524d|before.{0,15}(?:first )?(?:contact|collision)",
                language,re.IGNORECASE):
            raise ValueError('A multi-car contact goal must explicitly say what the support car '
                'does before the first EGO contact. The simulator stops at that contact; '
                'do not leave both attackers implied as colliding.')
        if re.search("(?:\u672a|\u5c1a\u672a)(?:\u5230\u8fbe|\u8fdb\u5165|\u9a76\u5165).{0,8}(?:\u51b2\u7a81\u533a\u57df|\u51b2\u7a81\u70b9|\u4ea4\u6c47\u533a\u57df|EGO.{0,4}\u8def\u5f84)|(?:\u4ec5|\u53ea\u662f).{0,12}(?:\u6cbf\u8def\u7ebf\u884c\u9a76|\u5411EGO\u65b9\u5411\u884c\u9a76|\u903c\u8fd1)",language,re.IGNORECASE):
            raise ValueError('A support car that stays outside the conflict area or merely drives '
                'toward it has no demonstrated cooperative role. Specify a distinct observable '
                'occupation or maneuver in the relevant corridor before first EGO contact, '
                'or use the single attacker alone.')
    if len(seen)==1:
        only=actors[next(iter(seen))]
        rel=only.get('relative_to_ego',{})
        same_ego_corridor=(rel.get('longitudinal_m',0)<-2 and
            abs(rel.get('lateral_m',math.inf))<
                (only.get('width',0)+actors['ego'].get('width',0))/2 and
            rel.get('heading_delta_rad',math.inf)<.25)
        has_adjacent_route=any(r.get('kind') in ('left','right') for r in only.get('route_options',[]))
        if (contact and same_ego_corridor and not has_adjacent_route and
            re.search("(?:\u53d8\u9053|\u5207\u5165).{0,20}"+ego_name+"\\s*(?:\u8f66\u9053|\u884c\u9a76\u8def\u5f84|\u8def\u7ebf|\u8def\u5f84)",language,re.IGNORECASE)):
            raise ValueError('This attacker starts directly behind EGO in the same directed corridor and has no offered adjacent lane. It cannot truthfully cut in from a side lane. Choose a genuine side/crossing actor or spawn slot; do not relabel a following approach as a cut-in.')
        if (rel.get('longitudinal_m',0)>5 and abs(rel.get('lateral_m',float('inf')))<1.5 and
            rel.get('heading_delta_rad',math.inf)<.35 and
            re.search("\u4ece[\u5de6\u53f3]\u4fa7(?:\u76f8\u90bb)?\u8f66\u9053",language)):
            raise ValueError('Goal says the attacker starts in a side lane, but its measured position '
                'is ahead in EGO\'s corridor. Choose a factual maneuver or a different actor.')
    lateral_maneuver=bool(re.search("\u5207\u5165|\u53d8\u9053|\u5e76\u7ebf|\u6a2a\u7a7f|\u6a2a\u5411|\u4fa7\u5411|\u8f6c\u5f2f|\u6389\u5934|\u7ed5\u884c|cut.in|lane.change|merge|cross|turn",language,re.IGNORECASE))
    same_lane_rear_end=bool(re.search("(?:\u540c\u8f66\u9053|\u672c\u8f66\u9053|\u5f53\u524d\u8f66\u9053).{0,35}(?:\u8ffd\u5c3e|\u649e\u51fb.{0,8}(?:\u5c3e\u90e8|\u8f66\u5c3e)|\u4ece\u540e\u65b9.{0,12}(?:\u63a5\u89e6|\u78b0\u649e|\u649e\u4e0a))|(?:\u8ffd\u5c3e|\u649e\u51fb.{0,8}(?:\u5c3e\u90e8|\u8f66\u5c3e)).{0,35}(?:\u540c\u8f66\u9053|\u672c\u8f66\u9053|\u5f53\u524d\u8f66\u9053)|(?:\u8ffd\u5c3e|\u649e\u51fb.{0,8}(?:\u5c3e\u90e8|\u8f66\u5c3e))",language,re.IGNORECASE))
    if same_lane_rear_end:
        raise ValueError('Rear-ending EGO is prohibited regardless of starting lane; choose a side or crossing interaction')
    if contact and not lateral_maneuver:
        for aid in seen:
            rel=actors[aid].get('relative_to_ego',{})
            if rel.get('relation')=='behind' and abs(rel.get('lateral_m',float('inf')))<2:
                raise ValueError('Attacker '+aid+' starts directly behind EGO in the same corridor. A contact goal must specify a real lateral/crossing maneuver; same-lane rear-end contact is prohibited')
    return copy.deepcopy(raw)


def validate_reactive_actions(actions, participants, snapshot):
    """Validate the behavior interface, without imposing physical limits."""
    expected={participant['actor_id'] for participant in participants}
    supplied={action.get('actor_id') for action in actions if isinstance(action,dict)} if isinstance(actions,list) else set()
    if supplied!=expected or len(actions)!=len(expected):
        raise ValueError('Provide exactly one action for every fixed attacker')
    actors={actor['id']:actor for actor in snapshot['actors']};checked=[]
    for action in actions:
        if not isinstance(action,dict):raise ValueError('Each reactive action must be an object')
        primitive=action.get('primitive')
        if primitive not in PRIMITIVES:raise ValueError('Unknown primitive')
        required=BASE_ACTION|ACTION_FIELDS[primitive]
        if set(action)!=required and not (primitive in ('speed','route_follow','lane_change','gap','lateral_offset') and set(action)==LEGACY_ACTION):
            raise ValueError('{} action requires {} (legacy six-field actions remain valid)'.format(primitive,', '.join(sorted(required))))
        aid=action['actor_id']
        options={route['route_id']:route for route in actors[aid]['route_options']}
        if action['route_id'] not in options:raise ValueError('Route '+str(action['route_id'])+' is not offered to '+aid)
        if primitive=='lane_change' and options[action['route_id']]['kind'] not in ('left','right'):
            raise ValueError('lane_change must use an offered adjacent route')
        if 'target_speed_mps' in action:finite_number(action['target_speed_mps'],'target_speed_mps',nonnegative=True)
        if 'target_gap_m' in action:finite_number(action['target_gap_m'],'target_gap_m')
        if 'offset_m' in action:finite_number(action['offset_m'],'offset_m')
        if primitive in ('side_overlap','staged_cut_in'):
            if action['side'] not in ('left','right'):
                raise ValueError(primitive+'.side must be left or right in the attacker route frame')
            if primitive=='side_overlap':
                finite_number(action['target_lateral_separation_m'],'target_lateral_separation_m',nonnegative=True)
            else:
                for field in ('approach_lateral_separation_m','contact_lateral_separation_m','trigger_gap_m'):
                    finite_number(action[field],field,nonnegative=True)
        if primitive=='intercept':
            number(action['target_forward_distance_m'],0.,100.,'target_forward_distance_m')
            number(action['target_arrival_time_s'],.1,8.,'target_arrival_time_s')
        checked.append(copy.deepcopy(action))
    return checked
