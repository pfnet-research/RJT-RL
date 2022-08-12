from __future__ import annotations

import copy
import logging
import random
from collections import deque
from typing import Any, Optional

import gym
import gym.spaces
import numpy as np
import torch
from rdkit import Chem

from rjt_rl.rjt import chemutils
from rjt_rl.rjt.assemble_mol import NodeMap
from rjt_rl.rjt.consts import MAX_NB
from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.mol_tree_node import MolTreeNode, NodeIDFunc
from rjt_rl.rjt.vocab import Vocab
from rjt_rl.rl.consts import MAX_ACTIONS
from rjt_rl.rl.rewards.base_reward import BaseReward

from .actions import Action2
from .base_history import BaseHistory
from .moltree_builder import MolTreeBuilder
from .states import State

logger = logging.getLogger(__name__)


def assign_depth(
    root_node: MolTreeNode, id_fn: NodeIDFunc = lambda x: x.safe_nid
) -> None:
    queue = deque([root_node])
    visited = set([id_fn(root_node)])
    root_node.depth1 = 0
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if id_fn(y) not in visited:
                queue.append(y)
                visited.add(id_fn(y))
                assert x.depth1 is not None
                y.depth1 = x.depth1 + 1


class MolEnvBase(gym.Env):  # type: ignore

    metadata = {"render.modes": ["human"]}

    observation_space: dict[Any, Any]
    state_history: list[State]
    mtbuild: Optional[MolTreeBuilder]
    init_smiles: str | list[str]

    def __init__(
        self,
        reward: Optional[BaseReward] = None,
        init_smiles: str | list[str] = "CC",
        history: Optional[BaseHistory] = None,
        standardize_mol: bool = False,
        neutralize_mol: bool = False,
    ):

        self.episode = -1
        self.episode_reward = 0.0

        self.reward = reward
        self.init_smiles = init_smiles

        self.mtbuild = None
        self.history = history
        self.standardize_mol = standardize_mol
        self.neutralize_mol = neutralize_mol

    def init(
        self, vocab: Vocab, max_steps: int = MAX_ACTIONS, max_nodes: int = MAX_ACTIONS
    ) -> None:
        self.action = Action2(vocab)

        self.action_space = self.action.get_action_space()
        self.observation_space = {}

        self.min_nodes = 4
        self.min_steps = 5
        self.max_nodes = max_nodes
        self.max_steps = max_steps
        self.counter = 0

        self.state_history = []
        self.vocab = vocab

        if self.mtbuild is None:
            self.mtbuild = MolTreeBuilder(
                vocab, self.standardize_mol, self.neutralize_mol
            )

    def seed(self, seed: int) -> None:
        np.random.seed(seed=seed)
        random.seed(seed)
        torch.manual_seed(0)

    def reset(self) -> MolTree:
        self.episode += 1
        self.episode_reward = 0.0
        self.counter = 0

        smiles = self.init_smiles
        if isinstance(smiles, list):
            smiles = random.choice(smiles)

        self.state = State(smiles, self.vocab, valid=True)
        logger.info(f"smiles: {smiles}")
        self.state_history = [self.state]

        assert self.state.mol_tree is not None
        assign_depth(self.state.mol_tree.get_root_node())

        self.counter += 1
        return self.get_observation()

    def set_state(self, mol_tree: MolTree) -> None:
        self.state.mol_tree = mol_tree
        mol, nmap = self.decode_mol_tree(mol_tree)
        if mol is None:
            mol_tree.dump_tree()
            raise RuntimeError("decode mol tree failed")

        self.state.idx_mol = copy.deepcopy(mol)
        self.state.node_map = nmap

        chemutils.set_atommap(mol)
        self.state.mol = mol
        self.state.smiles = Chem.MolToSmiles(mol)
        self.state.valid = True

    def render(self, mode: str = "human", close: bool = False) -> None:
        return

    def get_observation(self) -> MolTree:
        assert self.state.mol_tree is not None
        return copy.deepcopy(self.state.mol_tree)

    def get_current_smiles(self) -> str:
        assert self.state.smiles is not None
        return self.state.smiles

    def decode_mol_tree(
        self, mol_tree: MolTree
    ) -> tuple[Optional[Chem.Mol], Optional[NodeMap]]:
        assert self.mtbuild is not None
        return self.mtbuild.decode_mol_tree(mol_tree)

    def _add_node(self, action: list[int], mol_tree: MolTree) -> bool:
        assert self.mtbuild is not None
        return self.mtbuild.add_node(action, mol_tree)

    def check_and_backup_history(self) -> None:
        if self.history is not None:
            self.history.check_and_backup_history(self.episode)
        return

    def register_entry(self, state: State, info: dict[str, Any]) -> None:
        info["episode"] = self.episode
        info["episode_reward"] = self.episode_reward
        if self.history is not None:
            self.history.register_entry(state, info)
        self.check_and_backup_history()
        return

    def flush_history(self) -> None:
        if self.history is not None:
            self.history.flush_history()
        return


def check_moltree(mol_tree: MolTree) -> bool:
    reachable_nodes: set[MolTreeNode] = set()

    def _collect_reachable_nodes(
        nodes: set[MolTreeNode],
        node: MolTreeNode,
        fa_node: Optional[MolTreeNode] = None,
    ) -> None:
        nodes.add(node)
        for nei in node.neighbors:
            if nei is not fa_node:
                _collect_reachable_nodes(nodes, nei, fa_node=node)

    _collect_reachable_nodes(reachable_nodes, mol_tree.nodes[0])

    max_num_neis = max([len(n.neighbors) for n in mol_tree.nodes])

    if max_num_neis > MAX_NB:
        logger.error(f"Too many (>{MAX_NB}) neighbor nodes")
        return False
    unique_nodes = set(mol_tree.nodes)
    if len(reachable_nodes) != len(unique_nodes):
        logger.error("orphan node found")
        logger.error(f"unique nodes: {len(unique_nodes)} {unique_nodes}")
        logger.error(f"reachable nodes: {len(reachable_nodes)} {reachable_nodes}")
        return False
    return True
