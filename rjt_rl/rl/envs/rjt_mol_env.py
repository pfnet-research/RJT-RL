from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from rdkit import Chem

from rjt_rl.rjt import chemutils
from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rl.rewards.base_reward import BaseReward

from .actions import Action2
from .base_history import BaseHistory
from .mol_env_base import MolEnvBase, check_moltree

logger = logging.getLogger(__name__)


class RJTMolEnv(MolEnvBase):
    def __init__(
        self,
        reward: Optional[BaseReward] = None,
        init_smiles: str | list[str] = "CC",
        history: Optional[BaseHistory] = None,
        standardize_mol: bool = False,
        neutralize_mol: bool = False,
    ):
        super().__init__(
            reward,
            init_smiles,
            history=history,
            standardize_mol=standardize_mol,
            neutralize_mol=neutralize_mol,
        )
        logger.info(f"Init smiles: {self.init_smiles}")

    def reset(self) -> MolTree:
        result = super().reset()
        logger.info(f"=== NEW EPISODE {self.episode} ===")
        return result

    def step(self, action: list[int]) -> tuple[MolTree, float, bool, dict[str, Any]]:
        assert self.reward is not None
        logger.info(f"-- step {self.counter}; action={self.action.dump_str(action)}")

        info = {}

        assert self.state.mol_tree is not None
        nnodes = len(self.state.mol_tree.nodes)

        next_state = copy.deepcopy(self.state)

        assert next_state.mol_tree is not None
        add_ok = self._add_node(action, next_state.mol_tree)
        next_state.valid = False

        stop = False
        if self.action.end(action) and self.counter >= self.min_steps:
            logger.info(f"EOE generated at {self.counter}")
            stop = True

        if self.counter >= self.max_steps:
            logger.info("XXX: max step exceed")
            stop = True
        if nnodes >= self.max_nodes:
            logger.info("XXX: max node exceed")
            stop = True

        reward_step = 0.0
        if add_ok:
            if check_moltree(next_state.mol_tree):
                next_idx_mol, next_nmap = self.decode_mol_tree(next_state.mol_tree)
                if next_idx_mol is not None:
                    next_mol = copy.deepcopy(next_idx_mol)
                    chemutils.set_atommap(next_mol)
                    next_smiles = Chem.MolToSmiles(next_mol)
                    word_ac = action[Action2.WORD_PRED_ID]
                    word_size = self.vocab.get_word_size(word_ac)
                    if next_smiles != self.state.smiles or word_size == 1:
                        next_state.valid = True
                        next_state.mol = next_mol
                        next_state.smiles = next_smiles
                        next_state.idx_mol = next_idx_mol
                        next_state.node_map = next_nmap
                    else:
                        logger.info(
                            f"state not changed by action"
                            f" word:{word_ac} size: {word_size } smiles:{next_smiles}"
                        )
                        add_ok = False
                else:
                    logger.info("decode_mol_tree failed")

        if add_ok and next_state.valid:
            reward_step = self.reward.step_reward(self.state, next_state)
        else:
            reward_step = self.reward.step_reward(self.state, None)

        if add_ok and next_state.valid:
            self.state = next_state
        else:
            self.state.num_decode_fail += 1
        self.state_history.append(self.state)

        if stop:
            reward_final = self.reward.final_reward(self.state)
            reward = reward_step + reward_final
        else:
            reward = reward_step
            self.counter += 1

        ob = self.get_observation()

        logger.info(f"smiles: {self.state.smiles} Reward: {reward:.2f}")
        info = self.state.calc_mol_info()

        self.episode_reward += reward
        logger.info(f"reward: {reward}, episode_reward: {self.episode_reward}")

        if stop:
            logger.info(f"=== END EPISODE {self.episode} ===")
            self.register_entry(self.state, info)

        return ob, reward, stop, info
