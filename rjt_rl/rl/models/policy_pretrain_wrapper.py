from __future__ import annotations

import logging

import pytorch_pfn_extras as ppe
import torch

from rjt_rl.rjt.utils import Module
from rjt_rl.rjt.vocab import Vocab
from rjt_rl.rl.datasets.expert_dataset_collator import ListFromMolTree
from rjt_rl.rl.envs.mol_env_base import MolEnvBase

from .rjt_policy_net import RJTPolicyNet, RJTPolicyNetConfig

logger = logging.getLogger(__name__)


class PolicyPretrainWrapper(Module):
    def __init__(self, vocab: Vocab, env: MolEnvBase, hidden_size: int = 512) -> None:
        super().__init__()
        self.vocab = vocab

        self.policy = RJTPolicyNet(
            vocab,
            env.action_space,
            env.observation_space,
            config=RJTPolicyNetConfig(hidden_size=hidden_size),
        )
        self.policy.expert_learn = True

    def forward(self, s: ListFromMolTree, a: torch.Tensor) -> torch.Tensor:

        distrib, _ = self.policy(s)
        log_prob = distrib.log_prob(a)
        expert_loss: torch.Tensor = -log_prob.sum()

        ppe.reporting.report({"loss": expert_loss}, self)

        return expert_loss
