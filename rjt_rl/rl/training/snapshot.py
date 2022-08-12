from __future__ import annotations

import logging
import pickle
import random
from typing import Optional

import numpy as np
import torch
from pytorch_pfn_extras.training.extensions.snapshot_writers import SimpleWriter

from rjt_rl.rl.agents.rjt_ppo import RJTPPO
from rjt_rl.rl.envs.mol_env_base import MolEnvBase

logger = logging.getLogger(__name__)


class SnapshotHook:
    def __init__(self, freq: Optional[int], file_name: str, out_dir: str):
        self.freq = freq
        self.file_name = file_name
        self.out_dir = out_dir
        self.writer = SimpleWriter()

    def __call__(self, env: MolEnvBase, agent: RJTPPO, t: int) -> None:
        if self.freq is None or t % self.freq != 0:
            return
        target = {k: getattr(agent, k) for k in agent.saved_attributes}
        for k in agent.saved_attributes:
            if not hasattr(agent, k):
                continue
            v = getattr(agent, k)
            if v is None:
                continue
            target[k] = v

        target["random_states"] = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        )

        target["current_step"] = t
        target["current_episode"] = env.episode
        if hasattr(env, "history"):
            target["env_history"] = pickle.dumps(env.history)
        self.writer(self.file_name, self.out_dir, target)


def load_snap_file(env: MolEnvBase, agent: RJTPPO, file_name: str) -> int:
    logger.info(f"Load snapshot: {file_name}")
    state_dict = torch.load(file_name)  # type: ignore

    step_offset = state_dict.pop("current_step")
    assert type(step_offset) is int
    if "env_history" in state_dict:
        env.history = pickle.loads(state_dict.pop("env_history"))
    env.episode = state_dict.pop("current_episode")

    if "random_states" in state_dict:
        random.setstate(state_dict["random_states"][0])
        np.random.set_state(state_dict["random_states"][1])
        torch.set_rng_state(state_dict["random_states"][2].cpu())
        if torch.cuda.is_available() and state_dict["random_states"][3] is not None:
            torch.cuda.set_rng_state(state_dict["random_states"][3].cpu())
        del state_dict["random_states"]
        logger.info("loaded random state")

    for k in agent.saved_attributes:
        if not hasattr(agent, k):
            continue
        v = getattr(agent, k)
        if v is None:
            continue
        logger.info(f"loaded attribute {k}")
        setattr(agent, k, state_dict[k])

    return step_offset
