import numpy as np
from copy import deepcopy

from tbsim.configs.config import Dict


class EvaluationConfig(Dict):
    def __init__(self):
        super(EvaluationConfig, self).__init__()
        self.name = None
        self.env = "nusc"  # [l5kit, nusc]
        self.dataset_path = None
        self.eval_class = ""
        self.seed = 0
        self.num_scenes_per_batch = 2
        self.num_scenes_to_evaluate = 100

        self.num_episode_repeats = 1
        self.start_frame_index_each_episode = None  # if specified, should be the same length as num_episode_repeats
        self.seed_each_episode = None  # if specified, should be the same length as num_episode_repeats

        # self.num_episode_repeats = 3
        # self.start_frame_index_each_episode = None  # if specified, should be the same length as num_episode_repeats
        # self.seed_each_episode = [0,1,2]  # if specified, should be the same length as num_episode_repeats

        self.ego_only = False
        self.agent_eval_class = None

        self.ckpt_root_dir = "checkpoints/"
        self.experience_hdf5_path = None
        self.results_dir = "results/"


        self.ckpt.policy.ckpt_dir = None
        self.ckpt.policy.ckpt_key = None

        self.ckpt.planner.ckpt_dir = None
        self.ckpt.planner.ckpt_key = None

        self.ckpt.predictor.ckpt_dir = None
        self.ckpt.predictor.ckpt_key = None

        self.ckpt.cvae_metric.ckpt_dir = None
        self.ckpt.cvae_metric.ckpt_key = None

        self.ckpt.occupancy_metric.ckpt_dir = None
        self.ckpt.occupancy_metric.ckpt_key = None

        self.policy.mask_drivable = True
        self.policy.num_plan_samples = 50
        self.policy.num_action_samples = 10
        self.policy.pos_to_yaw = True
        self.policy.yaw_correction_speed = 1.0
        self.policy.diversification_clearance = None
        self.policy.sample = True


        self.policy.cost_weights.collision_weight = 15.0
        self.policy.cost_weights.lane_weight = 1.0
        self.policy.cost_weights.lane_dir_weight = 1.0
        self.policy.cost_weights.likelihood_weight = 0.0  # 0.1
        self.policy.cost_weights.progress_weight = 0.01  # 0.005

        self.metrics.compute_analytical_metrics = True
        self.metrics.compute_agent2ego_metrics = False
        self.metrics.compute_learned_metrics = False

        self.perturb.enabled = False
        self.perturb.OU.theta = 0.8
        self.perturb.OU.sigma = [0.0, 0.1,0.2,0.5,1.0,2.0,4.0]
        self.perturb.OU.scale = [1.0,1.0,0.2]

        self.rolling_perturb.enabled = False
        self.rolling_perturb.OU.theta = 0.8
        self.rolling_perturb.OU.sigma = 0.5
        self.rolling_perturb.OU.scale = [1.0,1.0,0.2]

        self.occupancy.rolling = True
        self.occupancy.rolling_horizon = [5,10,20]

        self.cvae.rolling = True
        self.cvae.rolling_horizon = [5,10,20]

        # self.nusc.eval_scenes = np.arange(0,100).tolist()
        self.nusc.eval_scenes = np.arange(0,150).tolist()
        self.nusc.n_step_action = 5
        self.nusc.num_simulation_steps = 200
        self.nusc.skip_first_n = 0
        
        self.drivesim.eval_scenes = np.arange(0,40).tolist()
        self.drivesim.n_step_action = 3
        self.drivesim.num_simulation_steps = 200
        self.drivesim.skip_first_n = 9

        self.l5kit.eval_scenes = [9058, 5232, 14153, 8173, 10314, 7027, 9812, 1090, 9453, 978, 10263, 874, 5563, 9613, 261, 2826, 2175, 9977, 6423, 1069, 1836, 8198, 5034, 6016, 2525, 927, 3634, 11806, 4911, 6192, 11641, 461, 142, 15493, 4919, 8494, 14572, 2402, 308, 1952, 13287, 15614, 6529, 12, 11543, 4558, 489, 6876, 15279, 6095, 5877, 8928, 10599, 16150, 11296, 9382, 13352, 1794, 16122, 12429, 15321, 8614, 12447, 4502, 13235, 2919, 15893, 12960, 7043, 9278, 952, 4699, 768, 13146, 8827, 16212, 10777, 15885, 11319, 9417, 14092, 14873, 6740, 11847, 15331, 15639, 11361, 14784, 13448, 10124, 4872, 3567, 5543, 2214, 7624, 10193, 7297, 1308, 3951, 14001]
        self.l5kit.n_step_action = 5
        self.l5kit.num_simulation_steps = 100
        self.l5kit.skip_first_n = 1
        self.l5kit.skimp_rollout = False
        
        
        
        

        self.adjustment.enabled = False #True
        self.adjustment.random_init_plan=False # True
        self.adjustment.remove_existing_neighbors = False #True
        
        self.init_recipe.num_target_agents = 1
        self.init_recipe.predefined_scene_init = None
        self.init_recipe.auto_adv_selection_mode = "none"
        self.init_recipe.auto_adv_track_index = None
        self.init_recipe.auto_adv_agent_name = None
        self.init_recipe.auto_ego_name = "ego"
        self.render_track_ids = False
        
        #vec_map params
        self.vec_map_params = Dict()
        self.vec_map_params.NUM_FUTURE_LANES = 6
        self.vec_map_params.FIND_ALL_POS_REFS = True
        self.vec_map_params.CENTERLINE_LENGTH = 150
        self.vec_map_params.CENTERLINE_LOOKAHEAD = 150
        self.vec_map_params.CENTERLINE_LOOKBACK = 50
        
        self.vec_map_params.FIND_CLOSEST_DIST_THRESHOLD = 2.5
        self.vec_map_params.FIND_CLOSEST_MAX_THRESHOLD = 5.0
        self.vec_map_params.FIND_ALL_DIST_THRESHOLD = 10.0
        self.vec_map_params.FIND_ALL_MAX_THRESHOLD = 20.0
        
        self.vec_map_params.DEFAULT_MAX_HEADING_ERROR = np.pi / 4.0
        self.vec_map_params.DFS_THRESHOLD_FUTURE = 150
        self.vec_map_params.DFS_THRESHOLD_PAST = 50
        self.vec_map_params.DFS_DEFAULT_THRESHOLD = 30
        
        self.vec_map_params.EXTEND_DISTANCE = 100.0

        self.llm.api_keys_path = ""
        self.llm.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.llm.api_key_env = "DEEPSEEK_API_KEY"
        self.llm.api_key_json_field = None
        self.llm.http_referer = None
        self.llm.app_title = "safe-sim-anchordrive"
        self.llm.model = "qwen3.6-plus"
        self.llm.prompt_version = "baselinec_waypoint"
        self.llm.temperature = 0.2
        self.llm.reasoning_effort = ""
        self.llm.enable_thinking = True
        self.llm.thinking_budget = 1024
        self.llm.timeout_s = 300.0
        self.llm.max_retries = 2
        self.llm.trajectory_horizon = 32
        self.llm.future_context_steps = 12
        self.llm.max_nearby_agents = 6
        self.llm.max_reference_points = 20
        self.llm.max_step_distance = 4.0
        self.llm.min_start_move_distance = 0.08
        self.llm.min_heading_update_distance = 0.15
        self.llm.enable_macro_actions = True
        self.llm.default_lane_width_m = 3.6
        self.llm.default_lane_change_duration_steps = 12
        self.llm.default_macro_hold_steps = 6
        self.llm.max_abs_accel_mps2 = 4.0
        self.llm.max_abs_yaw_rate_rps = 0.8
        self.llm.max_speed_mps = 25.0
        self.llm.validation_max_abs_accel_mps2 = 12.0
        self.llm.validation_max_abs_yaw_rate_rps = 2.0
        self.llm.validation_max_speed_mps = 40.0
        self.llm.validation_reject_dynamics_exceeded = False
        self.llm.validation_reject_waypoint_segment_jumps = True
        self.llm.validation_max_waypoint_speed_jump_mps = 6.0
        self.llm.validation_max_waypoint_heading_jump_rad = 0.8
        self.llm.llmscenario_disable_online_local_guard = True
        self.llm.llmscenario_enable_local_reality_assessor = True
        self.llm.llmscenario_local_reality_max_replans = 1
        self.llm.nearby_radius_m = 20.0
        self.llm.map_info_mode = "local_radius"
        self.llm.map_lane_radius_m = 20.0
        self.llm.send_route_information = True
        self.llm.route_num_points = 6
        self.llm.route_spacing_m = 5.0
        self.llm.send_nearby_agents = True
        self.llm.send_nearby_future_trajectory = True
        self.llm.enable_plan_assessor = True
        self.llm.plan_assessor_max_replans = 2
        self.llm.enable_local_collision_termination = True
        self.llm.call_interval_steps = None
        self.llm.send_bev_to_llm = True
        self.llm.bev_max_side = 512
        self.llm.bev_image_format = "jpeg"
        self.llm.bev_jpeg_quality = 85
        self.llm.save_bev_debug = False
        self.llm.bev_debug_dir = None
        self.llm.scenario_goal = "adv collide with ego"
        self.llm.objective = "Use the ego planning trajectory to generate a reasonable but aggressive ADV closed-loop trajectory, actively creating high-risk interactions."
        self.llm.manual_response_path = ""
        self.llm.results_dir = None
        self.stage2.enabled = False
        self.stage2.mode = "case_anchor"
        self.stage2.case_dir = None
        self.stage2.require_anchor = True
        self.stage2.dump_runtime_plans = True
        # self.adjustment.enabled = True
        # self.adjustment.random_init_plan= True
        # self.adjustment.remove_existing_neighbors = True
        # self.adjustment.initial_num_neighbors = 5
        # self.adjustment.num_frame_per_new_agent = 2000

    def clone(self):
        return deepcopy(self)


class TrainTimeEvaluationConfig(EvaluationConfig):
    def __init__(self):
        super(TrainTimeEvaluationConfig, self).__init__()

        self.num_scenes_per_batch = 4
        self.nusc.eval_scenes = np.arange(0, 100, 10).tolist()
        self.l5kit.eval_scenes = self.l5kit.eval_scenes[:20]

        self.policy.sample = False
