from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import torch
from omegaconf import MISSING

from rjt_rl.rjt.vocab import Vocab, load_vocab
from rjt_rl.rl.envs.rjt_mol_env import MolEnvBase, RJTMolEnv
from rjt_rl.rl.models.rjt_policy_net import RJTPolicyNet, RJTPolicyNetConfig
from rjt_rl.rl.rewards import get_reward_class
from rjt_rl.utils.config_wrapper import load_class, obj_from_config

from .ppo_trainer import PPOTrainer, PPOTrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class RJTPPOTrainerConfig(PPOTrainerConfig):
    reward: Optional[Any] = None
    init_smiles: str = "CC"
    model: RJTPolicyNetConfig = MISSING
    history: Optional[Any] = None
    standardize_mol: bool = False
    neutralize_mol: bool = False


class RJTPPOTrainer(PPOTrainer):
    @staticmethod
    def get_config_class() -> type[RJTPPOTrainerConfig]:
        return RJTPPOTrainerConfig

    config: RJTPPOTrainerConfig

    def __init__(self, config: RJTPPOTrainerConfig) -> None:
        super().__init__(config)

        self.init_smiles = config.init_smiles

        # reward config
        logger.debug(f"config.reward: {config.reward}")
        self.reward, _ = obj_from_config(get_reward_class, config.reward)

        # history config
        logger.info(f"config.history: {config.history}")
        if config.history is not None:
            self.history, _ = obj_from_config(
                partial(load_class, default_module_name="rjt_rl.rl.envs"),
                config.history,
            )
        else:
            self.history = None

    def create_env(self, idx: int) -> MolEnvBase:

        init_smiles: str | list[str] = self.init_smiles
        if init_smiles == "random":
            vocab = load_vocab(self.vocab)
            init_smiles = [vocab.get_smiles(i) for i in range(len(vocab))]

        return RJTMolEnv(
            reward=self.reward,
            init_smiles=init_smiles,
            history=self.history,
            standardize_mol=self.config.standardize_mol,
            neutralize_mol=self.config.neutralize_mol,
        )

    def create_model(self, vocab: Vocab, env: MolEnvBase) -> torch.nn.Module:
        return RJTPolicyNet(
            vocab,
            env.action_space,
            env.observation_space,
            self.config.model,
        )
