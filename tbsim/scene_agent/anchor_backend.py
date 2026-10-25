"""Direct anchor trajectories executed through SafeSim's EnvUnifiedSimulation.step."""
import copy
import math

import numpy as np

from .anchor_controller import AnchorController
from .backend import ForbiddenRearEnd, NuscStage1World, forbidden_rear_end
from .geometry import wrap


class NuscAnchorWorld(NuscStage1World):
    def begin_anchor_execution(self, participant_ids):
        from trajdata import AgentType
        from tbsim.utils.rollout_logger import RolloutLogger

        if self.time != 0:
            raise ValueError('Initialize the native environment before execution')
        env, sim = self._safesim_env, self._simulation_scene
        sim.agents = [a for a in sim.agents if a.name in self.states]
        # SafeSim expects history for newly inserted agents. Initialize only the
        # past with constant velocity; no recorded EGO future is used by the agent.
        generated = []
        for aid in self.spawned:
            actor = self.states[aid]
            times = np.arange(self.start_frame + 1)
            offsets = (times - self.start_frame)*self.dt*actor['speed']
            history = np.column_stack((actor['x']+offsets*math.cos(actor['yaw']),
                                       actor['y']+offsets*math.sin(actor['yaw']),
                                       np.full(len(times), actor['yaw'])))
            generated.append((aid, history, 0, AgentType.VEHICLE,
                              [actor['length'], actor['width'], 1.5]))
        if generated:
            sim.add_new_agents(generated)
        sim.scene.agents = sim.agents
        sim.scene.agent_presence[sim.scene_ts] = sim.agents
        sim.initialize_ref_dict()
        env._current_scenes = [sim]
        env._current_scene_indices = [self.scene_index]
        env._num_scenes = 1
        env._frame_index = 0
        env._done = False
        env._cached_observation = None
        env._cached_raw_observation = None
        env._last_collision_info = None
        env._log_data = False  # Stage1Tools stores the full executed branch.
        env.logger = RolloutLogger()
        env._env_config.unlock()
        env._env_config.simulation.num_simulation_steps = round(self.config['max_simulation_s']/self.dt)
        env._env_config.lock()
        env.ego_indices = [env.current_agent_names.index('ego')]
        env.batch_ego_indices = np.array([aid == 'ego' for aid in env.current_agent_names])
        env.batch_ctrl_indices = np.array([aid in participant_ids for aid in env.current_agent_names])
        self.anchor_controller = AnchorController()
        self.anchor_participants = set(participant_ids)

    def snapshot_execution_state(self):
        sim, env = self._simulation_scene, self._safesim_env
        return dict(cache=copy.deepcopy(sim.cache), scene_ts=sim.scene_ts,
                    agent_presence=copy.deepcopy(sim.scene.agent_presence),
                    agent_ref_dict=copy.deepcopy(sim.agent_ref_dict),
                    frame_index=env._frame_index, done=env._done,
                    anchors=copy.deepcopy(self.anchor_controller))

    def restore_execution_state(self, checkpoint):
        sim, env = self._simulation_scene, self._safesim_env
        sim.cache = copy.deepcopy(checkpoint['cache'])
        sim.cache.scene = sim.scene
        sim.scene_ts = checkpoint['scene_ts']
        sim.scene.agent_presence = copy.deepcopy(checkpoint['agent_presence'])
        sim.agent_ref_dict = copy.deepcopy(checkpoint['agent_ref_dict'])
        env._frame_index = checkpoint['frame_index']
        env._done = checkpoint['done']
        env._last_collision_info = None
        env._cached_observation = None
        env._cached_raw_observation = None
        self.anchor_controller = copy.deepcopy(checkpoint['anchors'])

    def step(self):
        from tbsim.policies.common import Action, RolloutAction

        before = copy.deepcopy(self.states)
        old_plan = copy.deepcopy(self._ego_plan)
        old_anchor_controls = copy.deepcopy(self.anchor_controller.previous)
        old_bg_controller = copy.deepcopy(self.bg_controller)
        old_fallbacks = set(self.background_fallbacks)
        updated = {'ego': self._ego_step()}
        next_frame = self.start_frame + round(self.time/self.dt) + 1
        for aid, actor in before.items():
            if aid == 'ego':
                continue
            updated[aid] = self._non_ego_step(aid, actor, before, next_frame)
        # Reject before advancing native simulation, retaining the one-step transaction.
        for aid in self.anchor_participants:
            relation = forbidden_rear_end(before[aid], updated[aid], before['ego'], updated['ego'])
            if relation is not None:
                self._ego_plan = old_plan
                self.anchor_controller.previous = old_anchor_controls
                self.bg_controller = old_bg_controller
                self.background_fallbacks = old_fallbacks
                raise ForbiddenRearEnd(aid, relation, self.time+self.dt)
        env = self._safesim_env
        env.lightweight_native_steps = bool(self.config.get('lightweight_native_steps', False))
        obs = (env.get_step_observation() if env.lightweight_native_steps else env.get_observation())['agents']
        positions, yaws = [], []
        for i, aid in enumerate(env.current_agent_names):
            current = np.asarray(obs['centroid'][i], dtype=float)
            yaw = float(np.asarray(obs['yaw'][i]).item())
            target = updated[aid]
            delta = np.array([target['x'], target['y']])-current
            c, s = math.cos(yaw), math.sin(yaw)
            positions.append([[c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]]])
            yaws.append([[wrap(target['yaw']-yaw)]])
        env.step(RolloutAction(agents=Action(np.asarray(positions, dtype=np.float64),
                                            np.asarray(yaws, dtype=np.float64))),
                 num_steps_to_take=1, render=False)
        self.time = round(env._frame_index*self.dt, 6)
        sim = self._simulation_scene
        sim.cache.reset_obs_frame()
        sim.cache.reset_obs_format()
        # Read back native simulated positions and finite-difference speeds, rather
        # than claiming that the requested states were necessarily executed.
        for aid, actor in updated.items():
            row = sim.cache.persistent_data_df.loc[(aid, sim.scene_ts)]
            actor.update(x=float(row.x), y=float(row.y), yaw=float(row.heading),
                         speed=float(np.hypot(row.vx, row.vy)))
        self.anchor_controller.previous = old_anchor_controls
        controls = {aid: self.anchor_controller.measured_control(before[aid], updated[aid], self.dt)
                    for aid in self.anchor_participants}
        self.states = updated
        self._forecast_cache.clear()
        return before, copy.deepcopy(updated), controls

    def _non_ego_step(self, aid, actor, before, next_frame):
        if aid in self.anchor_participants:
            target, _ = self.anchor_controller.step(actor, self.time, self.dt)
        elif (aid, next_frame) in self._recorded.index:
            row = self._recorded.loc[(aid, next_frame)]
            target = dict(actor, x=float(row.x), y=float(row.y), yaw=float(row.heading),
                                speed=float(np.hypot(row.vx, row.vy)))
        else:
            self.background_fallbacks.add(aid)
            options = self.actor_routes[aid]
            if options:
                target, _ = self.bg_controller.step(actor, before['ego'],
                    dict(primitive='route_follow', route_id=options[0]['route_id'],
                         target_speed_mps=actor['speed']), self.routes, self.dt)
            else:
                target = dict(actor, x=actor['x']+actor['speed']*math.cos(actor['yaw'])*self.dt,
                                    y=actor['y']+actor['speed']*math.sin(actor['yaw'])*self.dt)
        return target
