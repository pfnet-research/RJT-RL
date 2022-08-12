from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from rjt_rl.rl.envs.states import State

from .base_reward import BaseReward

logger = logging.getLogger(__name__)


@dataclass
class BaseMolRewardConfig:
    valid_step_reward: float = 0.1
    invalid_step_reward: float = 0.0
    use_final_reward: bool = False
    init_smiles: str = "CC"
    score_until_dup_count: Optional[int] = None
    reuse_step_score: bool = False


class BaseMolReward(BaseReward):
    @staticmethod
    def get_config_class() -> type[BaseMolRewardConfig]:
        return BaseMolRewardConfig

    @classmethod
    def from_config(cls, config: BaseMolRewardConfig) -> "BaseMolReward":
        return cls(config)

    def __init__(self, config: BaseMolRewardConfig):

        if (
            isinstance(config.score_until_dup_count, int)
            and config.score_until_dup_count < 0
        ):
            dup_count = None
        else:
            dup_count = config.score_until_dup_count

        super().__init__(score_until_dup_count=dup_count)
        self._init_smiles = config.init_smiles
        self.valid_step = config.valid_step_reward
        self.invalid_step = config.invalid_step_reward
        self.use_final_reward = config.use_final_reward
        self.reuse_step_score = config.reuse_step_score
        logger.info(f"score_until_dup_count: {self.score_until_dup_count}")
        self.known_smiles = {}

    def step_reward(self, current_state: State, next_state: Optional[State]) -> float:
        if self.use_final_reward:
            if next_state is not None and next_state.valid:
                return self.valid_step
            else:
                return self.invalid_step

        if not hasattr(current_state, "score"):
            current_state.score = 0

        if next_state is not None and next_state.valid:
            next_score = self.calc_score(next_state)
            next_state.score = next_score
            if self.reuse_step_score:
                curr_score = current_state.score
            else:
                curr_score = self.calc_score(current_state)
            del_score = next_score - curr_score
            step_rew = self.valid_step + del_score
            logger.info(f"{curr_score=} --> {next_score=} delta: {del_score}")
            logger.info(f"{step_rew=}")
            return step_rew
        else:
            return self.invalid_step

    def final_reward(self, state: State) -> float:
        cur_score = self.calc_score(state)

        assert state.smiles is not None
        if self.use_final_reward:
            before_score = cur_score
            penal_score = cur_score
            after_score = self.calc_dup_penalty(
                state.smiles, score=penal_score, default_score=before_score
            )
            return after_score
        else:
            before_score = 0.0
            penal_score = cur_score
            after_score = self.calc_dup_penalty(
                state.smiles, score=penal_score, default_score=before_score
            )
            return after_score

    def calc_score(self, state: State) -> float:
        raise NotImplementedError()
