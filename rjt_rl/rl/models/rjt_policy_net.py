from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
import torch.nn as nn
from pfrl.nn.mlp import MLP
from torch.distributions import Categorical

from rjt_rl.nn.site_tree.embed_site_info import EmbedSiteInfoBidir1
from rjt_rl.nn.site_tree.site_tree_encoder import SiteTreeEncoder
from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.utils import (
    Module,
    check_node_idx,
    filter_logits,
    index_tensor,
    mask_invalid_nodes,
    set_batch_node_id,
)
from rjt_rl.rjt.vocab import Vocab
from rjt_rl.rl.datasets.expert_dataset_collator import ListFromMolTree
from rjt_rl.rl.envs.mol_action_distr import MolActionDistr3

logger = logging.getLogger(__name__)


@dataclass
class RJTPolicyNetConfig:
    stop_ent_coef: float = 1e-20
    word_ent_coef: float = 1.0
    targ_ent_coef: float = 1.0

    site1_ent_coef: float = 0.0
    site2_ent_coef: float = 0.0
    sitedir_ent_coef: float = 0.0

    hidden_size: int = 512
    freeze_actions: List[int] = field(default_factory=list)

    depth_limit: Optional[int] = None


class RJTPolicyNet(Module):
    @staticmethod
    def get_config_class() -> type[RJTPolicyNetConfig]:
        return RJTPolicyNetConfig

    @classmethod
    def from_config(
        cls,
        vocab: Vocab,
        action_space: Any,
        observation_space: Any,
        config: RJTPolicyNetConfig,
    ) -> RJTPolicyNet:
        return cls(vocab, action_space, observation_space, config)

    def __init__(
        self,
        vocab: Vocab,
        action_space: Any,
        observation_space: Any,
        config: RJTPolicyNetConfig,
    ) -> None:
        self.config = config
        self.depth_limit = config.depth_limit

        self._dump = False
        self.vocab = vocab
        self.action_space = action_space
        self.hidden_size = config.hidden_size

        self.expert_learn = False

        self.node_feature_size = config.hidden_size
        self.edge_feature_size = config.hidden_size

        self.freeze_actions = config.freeze_actions

        super().__init__()

        vocab_size = len(vocab)
        self.embedding = nn.Embedding(
            vocab_size + 1, self.node_feature_size, padding_idx=vocab_size
        )
        self.embed_slot = EmbedSiteInfoBidir1(out_size=self.edge_feature_size)

        self.tree_encoder = SiteTreeEncoder(
            vocab,
            self.hidden_size,
            self.node_feature_size,
            self.edge_feature_size,
            self.embedding,
            self.embed_slot,
        )

        hidden_size_list = [self.hidden_size, self.hidden_size]

        self.Wa1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.Wns = nn.Linear(self.hidden_size, 1)

        self.W_node = MLP(
            in_size=self.hidden_size,
            out_size=self.action_space.nvec[1],
            hidden_sizes=hidden_size_list,
        )

        self.W_site_dir = MLP(
            in_size=self.hidden_size,
            out_size=self.action_space.nvec[2],
            hidden_sizes=hidden_size_list,
        )

        self.W_site1 = MLP(
            in_size=self.hidden_size,
            out_size=self.action_space.nvec[3],
            hidden_sizes=hidden_size_list,
        )

        self.W_site2 = MLP(  # type: ignore
            in_size=self.hidden_size,
            out_size=self.action_space.nvec[4],
            hidden_sizes=hidden_size_list,
        )

        self.Waggr = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wstop2 = MLP(
            in_size=self.hidden_size,
            out_size=self.action_space.nvec[5],
            hidden_sizes=hidden_size_list,
        )

        self.W_value = MLP(
            in_size=self.hidden_size, out_size=1, hidden_sizes=hidden_size_list
        )

    def make_depth_mask(
        self,
        mol_batch: Sequence[MolTree],
        node_vecs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        shape = node_vecs.shape[:-1]
        mask = torch.full(shape, fill_value=-1.0, device=device)
        for i, mol_tree in enumerate(mol_batch):
            for j, node in enumerate(mol_tree.nodes):
                if node.depth1 is not None:
                    mask[i, j] = node.depth1
        return mask

    def forward(self, batch: ListFromMolTree) -> tuple[MolActionDistr3, torch.Tensor]:
        device = self.device()

        mol_batch = batch.get_mol_batch()
        set_batch_node_id(mol_batch)
        check_node_idx(mol_batch)

        try:
            _, z = self.tree_encoder(batch.get_encoder_data(), aggregate_all=True)
            if self._dump:
                self._dump = False
        except Exception:
            logger.error("Failed: tree_encoder")
            for i, mt in enumerate(mol_batch):
                logger.error(f"{i}: len(nodes): {mt.dump_str()}")
                mt.dump_tree()
            raise

        node_sizes = index_tensor(
            [len(mol_tree.nodes) for mol_tree in mol_batch], device
        )
        node_vecs = nn.ReLU()(self.Wa1(z))
        aggr_vec = self._aggregate_nodes(node_vecs, node_sizes)
        stop_distrib = self._stop_prediction(aggr_vec)
        if self.depth_limit is not None and not self.training:
            depth_info = self.make_depth_mask(mol_batch, node_vecs, device)
        else:
            depth_info = None
        targ_distrib = self._target_prediction(node_vecs, node_sizes, depth_info)
        word_distrib = self._word_prediction(node_vecs)
        sitedir_distrib = self._sitedir_prediction(node_vecs)
        site1_scores = self._site1_prediction(node_vecs)
        site2_scores = self._site2_prediction(node_vecs)
        vpred = self.W_value(aggr_vec)
        distr = MolActionDistr3(
            mol_batch,
            self.vocab,
            stop_distrib,
            targ_distrib,
            word_distrib,
            sitedir_distrib,
            site1_scores,
            site2_scores,
            stop_ent_coef=self.config.stop_ent_coef,
            word_ent_coef=self.config.word_ent_coef,
            targ_ent_coef=self.config.targ_ent_coef,
            freeze_actions=self.config.freeze_actions,
            site1_ent_coef=self.config.site1_ent_coef,
            site2_ent_coef=self.config.site2_ent_coef,
            sitedir_ent_coef=self.config.sitedir_ent_coef,
        )

        return (distr, vpred)

    def _aggregate_nodes(
        self, node_vecs: torch.Tensor, node_sizes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h_vecs = nn.ReLU()(self.Waggr(node_vecs))
        if node_sizes is not None:
            h_vecs = mask_invalid_nodes(h_vecs, node_sizes)
        aggr_vec: torch.Tensor = h_vecs.sum(dim=1)
        return aggr_vec

    def _stop_prediction(self, aggr_vec: torch.Tensor) -> Categorical:
        stop_scores = self.Wstop2(aggr_vec)
        return Categorical(logits=stop_scores)  # type: ignore

    def _target_prediction(
        self,
        node_vecs: torch.Tensor,
        node_sizes: torch.Tensor,
        depth_info: Optional[torch.Tensor] = None,
        padding: float = -1.0e10,
    ) -> Categorical:
        targ_scores = self.Wns(node_vecs)
        targ_scores = targ_scores[:, :, 0]
        targ_scores = filter_logits(targ_scores, node_sizes)
        if depth_info is not None and self.depth_limit is not None:
            dummy_scores = torch.full_like(
                targ_scores, fill_value=padding, device=targ_scores.device
            )
            bool_mask = depth_info <= self.depth_limit
            targ_scores = torch.where(bool_mask, targ_scores, dummy_scores)

        return Categorical(logits=targ_scores)  # type: ignore

    def _word_prediction(self, node_vecs: torch.Tensor) -> Categorical:
        word_scores = self.W_node(node_vecs)
        return Categorical(logits=word_scores)  # type: ignore

    def _sitedir_prediction(self, node_vecs: torch.Tensor) -> Categorical:
        sitedir_scores = self.W_site_dir(node_vecs)
        return Categorical(logits=sitedir_scores)  # type: ignore

    def _site1_prediction(self, node_vecs: torch.Tensor) -> torch.Tensor:
        site1_scores: torch.Tensor = self.W_site1(node_vecs)
        return site1_scores

    def _site2_prediction(self, node_vecs: torch.Tensor) -> torch.Tensor:
        site2_scores: torch.Tensor = self.W_site2(node_vecs)
        return site2_scores
