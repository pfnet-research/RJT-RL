from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from rjt_rl.nn.tree_rnn import TreeGRU
from rjt_rl.rjt.consts import MAX_NB
from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.mol_tree_node import MolTreeNode, PropOrder
from rjt_rl.rjt.utils import get_root_batch, index_tensor, pad_nodes
from rjt_rl.rjt.vocab import Vocab

logger = logging.getLogger(__name__)
MsgDict = Dict[Tuple[int, int], torch.Tensor]


class TreeEncoder(nn.Module):

    gru: nn.Module

    def __init__(
        self,
        vocab: Vocab,
        hidden_size: int,
        node_feature_size: int,
        embedding: Optional[nn.Embedding] = None,
        gru: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = len(vocab)
        self.vocab = vocab

        self.node_feature_size = node_feature_size
        self._pad_size_error = -1

        if embedding is None:
            # use padding index (=vocab_size)
            self.embedding = nn.Embedding(
                self.vocab_size + 1, node_feature_size, padding_idx=self.vocab_size
            )
        else:
            self.embedding = embedding

        self.W = nn.Linear(node_feature_size + hidden_size, hidden_size)

        # GRU
        if gru is None:
            self.gru = TreeGRU(hidden_size, node_feature_size)
        else:
            self.gru = gru

    def calc_prop_order(
        self, mol_batch: Iterable[MolTree], up_only: bool = True
    ) -> list[PropOrder]:
        orders = []
        for mt in mol_batch:
            order = mt.calc_prop_order(up_only=up_only, id_fn=lambda x: x.safe_idx)
            orders.append(order)
        return orders

    def device(self) -> torch.device:
        if not hasattr(self, "device_"):
            self.device_ = torch.device(next(self.parameters()).device)
        return self.device_

    def forward(self, mol_batch: Sequence[MolTree]) -> tuple[MsgDict, torch.Tensor]:
        device = self.device()
        root_batch = get_root_batch(mol_batch)
        orders = self.calc_prop_order(mol_batch)

        h: MsgDict = {}
        max_depth = max([len(x) for x in orders])
        padding = torch.zeros(self.hidden_size, device=device)

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = []
            cur_h_nei = []
            for node_x, node_y in prop_list:
                x, y = node_x.safe_idx, node_y.safe_idx
                cur_x.append(node_x.safe_wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.safe_idx
                    if z == y:
                        continue
                    h_nei.append(h[(z, x)])

                h_nei = pad_nodes(h_nei, padding, MAX_NB)
                cur_h_nei.extend(h_nei)
                assert len(cur_h_nei) % MAX_NB == 0

            cur_x_t = index_tensor(cur_x, device)
            cur_x_te = self.embedding(cur_x_t)
            h_nei_cat = torch.cat(cur_h_nei, dim=0).view(-1, MAX_NB, self.hidden_size)

            new_h = self.gru(cur_x_te, h_nei_cat)

            for i, m in enumerate(prop_list):
                h[(m[0].safe_idx, m[1].safe_idx)] = new_h[i]

        root_vecs = self.node_aggregate(root_batch, h)

        return h, root_vecs

    def node_aggregate(
        self,
        nodes: Iterable[Optional[MolTreeNode]],
        h: MsgDict,
    ) -> torch.Tensor:
        device = self.device()
        x_idx = []
        h_nei = []
        hidden_size = self.hidden_size
        padding = torch.zeros(hidden_size, device=device)

        for node_x in nodes:
            if node_x is None:
                x_idx.append(self.embedding.padding_idx)
                nei = []
            else:
                x_idx.append(node_x.safe_wid)
                nei = [
                    h[node_y.safe_idx, node_x.safe_idx] for node_y in node_x.neighbors
                ]
            nei = pad_nodes(nei, padding, MAX_NB)
            h_nei.extend(nei)
            assert len(h_nei) % MAX_NB == 0

        h_cat = torch.cat(h_nei, dim=0)
        if len(h_cat) % (MAX_NB * hidden_size) != 0:
            logger.error(
                f"{len(h_cat)} / {MAX_NB} / {hidden_size} = {len(h_cat) / MAX_NB / hidden_size}"
            )
        try:
            h_cat = h_cat.view(-1, MAX_NB, hidden_size)
        except Exception as e:
            logger.error(f"reshape failed: {e}")
            logger.error(f"nodes: {nodes}")
            raise

        sum_h_nei = h_cat.sum(dim=1)
        x_vec = index_tensor(x_idx, device)
        x_vec = self.embedding(x_vec)
        node_vec = torch.cat([x_vec, sum_h_nei], dim=1)

        result: torch.Tensor = nn.ReLU()(self.W(node_vec))
        return result
