"""Review sparse waypoint proposals visually before committing a simulation step."""
import base64
import copy
from pathlib import Path

from .anchor_execution_agent import AnchorExecutionTools, SYSTEM as ANCHOR_SYSTEM, TOOLS as ANCHOR_TOOLS
from .simple_goal_agent import _function

SYSTEM = ANCHOR_SYSTEM + "\nact now only submits a plan and generates a purple waypoint preview, without advancing the simulation. First observe to see the current state, then act; after seeing the preview image, verify the vehicle heading direction, speed, continuity of front and rear wheel motion, road direction, and collision timing. If reasonable, confirm_act(preview_id,reason) to execute; if unreasonable, act again to modify and view the image again, or use rollback for your own mistakes. You cannot confirm in the same reply that submits act, and do not submit multiple tools at once; first look at the image returned by the tool before deciding. The confirmation reason only needs to briefly state the verification conclusion, no lengthy analysis required.\nThe light-colored area of the map is the complete drivable surface, and the thick brown line is its boundary (including hole boundaries); the light brown area is not marked as drivable by the map and is not equivalent to a known guardrail. Thin gray lines are lane lines or connecting centerlines, and do not represent the outer boundary of the road. The purple semi-transparent vehicle body and direction line are the vehicle body extent and orientation sampled according to the plan to be executed, not a safety certification; check the complete vehicle body and the path between points, do not only look at the anchor center or replace the road check with a first-segment angle difference \u226490\u00b0. If the vehicle body enters the light brown area, it must be verified and modified; you cannot claim without basis that \"the road surface is continuous with no boundary crossing.\"\nEach request displays at most three images: the state at the starting point of the previous round's decision, the current state, and the latest unexecuted plan of this round. The two state images each use the EGO at that time as the coordinate origin; the displacement on the images cannot be directly treated as world displacement. The old plan image is replaced, while the text history is retained; after rollback, discard the abandoned branch images."
TOOLS = copy.deepcopy(ANCHOR_TOOLS)
TOOLS[2]['function']['description'] = "Submit or modify the waypoint plan, and only a preview image that has not been executed is returned; after inspecting the image, another call to confirm_act is required to execute."
TOOLS.append(_function('confirm_act', "After viewing the latest preview and considering it reasonable, execute the second before that plan; return the measured state.", {
    'preview_id': {'type':'string','description':"The preview_id returned by the latest act; old plans are invalid."},
    'reason': {'type':'string','description':"Briefly describe the conclusions from inspecting the images for motion continuity, road, and timing."}}, ['preview_id','reason']))


class PreviewAnchorExecutionTools(AnchorExecutionTools):
    system_prompt = SYSTEM
    tool_definitions = TOOLS
    bounded_image_context = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_preview = None
        self.preview_serial = 0
        self.seen_preview_id = None
        self.observed_time = None

    def observe(self):
        result = super().observe()
        self.observed_time = self.world.time
        return result

    def act(self, actions, reason):
        if self.observed_time != self.world.time:
            raise ValueError('Call observe and inspect the current state before proposing waypoints')
        plans, interval = self._build_action_plans(actions, reason)
        image = self.view.view_scene(highlight_actor_ids=self.selected,
            ego_plan=self.world.ego_current_plan(), anchor_preview=plans)
        self.preview_serial += 1
        preview_id = 'preview_%03d'%self.preview_serial
        self.pending_preview = dict(preview_id=preview_id,time_s=self.world.time,
            actions=copy.deepcopy(actions),reason=reason,plans_world=plans)
        self.seen_preview_id = None
        self.episode.write(preview_id+'.json',self.pending_preview)
        return dict(preview_id=preview_id,time_s=self.world.time,executed=False,
            execution_interval_s=interval,image_path=image['image_path'],image_name=image['image_name'],
            action_reminder=self.action_reminder(),
            next="First look at the purple preview; if reasonable, confirm_act; if not, act again to replace the plan.")

    def mark_preview_seen(self):
        # Called only after a model response to a request containing this preview.
        if self.pending_preview:
            self.seen_preview_id = self.pending_preview['preview_id']

    def confirm_act(self, preview_id, reason):
        proposal = self.pending_preview
        if (not proposal or proposal['preview_id'] != preview_id
                or proposal['time_s'] != self.world.time or self.seen_preview_id != preview_id):
            raise ValueError('Inspect the latest act preview in a separate model turn before confirming')
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('Briefly explain your visual check')
        self.pending_preview = None
        self.seen_preview_id = None
        result = super().act(proposal['actions'],proposal['reason'])
        self.episode.action_history[-1].update(preview_id=preview_id,confirmation_reason=reason)
        self.episode._save_trace()
        result.update(executed=True,confirmed_preview_id=preview_id)
        return result

    def rollback(self, to_time_s, reason):
        result = super().rollback(to_time_s,reason)
        self.pending_preview = None
        self.seen_preview_id = None
        return result


class DecisionImageWindow:
    """Keep image payloads outside persistent text/tool history."""
    def __init__(self):
        self.previous = self.current = self.preview = None

    def update(self, tool, observation, image_path):
        frame = dict(path=str(image_path),time_s=observation['time_s'])
        if tool == 'rollback':
            self.previous = self.preview = None
            self.current = frame
        elif tool == 'act':
            self.preview = frame
        else:
            if self.current and self.current['time_s'] != frame['time_s']:
                self.previous = self.current
                self.preview = None
            if tool == 'confirm_act':
                self.preview = None
            self.current = frame

    def message(self):
        parts = []
        for label,frame in (("\u4e0a\u4e00\u8f6e\u51b3\u7b56\u8d77\u70b9\u72b6\u6001",self.previous),("\u5f53\u524d\u72b6\u6001",self.current),
                            ("Latest waypoint plan, not yet executed",self.preview)):
            if frame:
                parts.append(dict(type='text',text="%s, simulation t=%.1fs"%(label,frame['time_s'])))
                encoded=base64.b64encode(Path(frame['path']).read_bytes()).decode()
                parts.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+encoded)))
        return dict(role='user',content=parts) if parts else None
