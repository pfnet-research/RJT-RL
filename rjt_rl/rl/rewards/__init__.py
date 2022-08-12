from __future__ import annotations

from rjt_rl.utils.config_wrapper import load_class

from .penalized_logp_reward import PenalizedLogPReward  # NOQA
from .similarity_reward import SimilarityReward  # NOQA


def get_reward_class(clsnm: str) -> type:
    return load_class(clsnm, default_module_name="rjt_rl.rl.rewards")
