"""Construct the native SafeSim nuScenes environment for AnchorDrive."""
from pathlib import Path
import torch
from trajdata import MapAPI
from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.evaluation.env_builders import EnvNuscBuilder


class NoCollisionSceneBrowser:
    def __init__(self, dataset_path: str):
        cfg = EvaluationConfig()
        cfg.env = "nusc"
        cfg.dataset_path = str(dataset_path)
        cfg.num_scenes_per_batch = 1
        cfg.num_simulation_steps = 40
        cfg.nusc.num_simulation_steps = 40
        cfg.train.load_cache = True
        cfg.train.data_filter = None
        cfg.metrics.compute_analytical_metrics = False
        cfg.metrics.compute_agent2ego_metrics = False
        cfg.metrics.compute_learned_metrics = False
        exp_cfg = get_registered_experiment_config("nusc_bc")
        self.env = EnvNuscBuilder(
            eval_config=cfg,
            exp_config=exp_cfg,
            device=torch.device("cpu"),
        ).get_env(split_ego=False, parse_obs=True)
        self.map_api = MapAPI(Path("~/.unified_data_cache").expanduser(), keep_in_memory=True)
