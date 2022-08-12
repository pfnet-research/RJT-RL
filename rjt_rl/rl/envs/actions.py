from __future__ import annotations

import gym
import gym.spaces
import numpy as np

from rjt_rl.rjt.consts import MAX_SITES
from rjt_rl.rjt.vocab import Vocab
from rjt_rl.rl.consts import MAX_ACTIONS


class Action2:
    TARGET_PRED_ID = 0
    WORD_PRED_ID = 1
    DIR_PRED_ID = 2
    SITE1_PRED_ID = 3
    SITE2_PRED_ID = 4
    STOP_PRED_ID = 5

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.nvocab = len(vocab)
        self.nsite = MAX_SITES

        self.action_space = gym.spaces.MultiDiscrete(
            [MAX_ACTIONS, self.nvocab, 3, self.nsite, self.nsite, 2]
        )

    def get_action_space(self) -> gym.spaces.MultiDiscrete:
        return self.action_space

    def sample(self) -> np.ndarray:
        result: np.ndarray = self.action_space.sample()
        return result

    def target_node(self, action: list[int]) -> int:
        return action[self.TARGET_PRED_ID]

    def add_node(self, action: list[int]) -> int:
        wid = action[self.WORD_PRED_ID]
        return wid

    def site_info1(
        self, action: list[int], max_site: int = MAX_SITES
    ) -> tuple[int, int]:
        return (
            int(action[self.SITE1_PRED_ID] % max_site),
            int(action[self.DIR_PRED_ID]) - 1,
        )

    def site_info2(
        self, action: list[int], max_site: int = MAX_SITES
    ) -> tuple[int, int]:
        return (
            int(action[self.SITE2_PRED_ID] % max_site),
            int(action[self.DIR_PRED_ID]) - 1,
        )

    def end(self, action: list[int]) -> bool:
        return action[self.STOP_PRED_ID] == 1

    def dump_str(self, ac: list[int]) -> str:
        res = ""

        if self.end(ac):
            res += "<END> "

        res += f"target:{self.target_node(ac)} "

        wid = self.add_node(ac)
        res += f"wid:{wid:d}/{self.vocab.get_smiles(wid)} "

        res += f"site1:{self.site_info1(ac)} "
        res += f"site2:{self.site_info2(ac)}"

        return res
