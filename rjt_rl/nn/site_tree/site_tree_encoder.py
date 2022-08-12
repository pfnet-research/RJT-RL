from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional, Protocol, Type

import torch
import torch.nn as nn

from rjt_rl.nn.site_tree.edge_tree_gru import EdgeTreeGRU
from rjt_rl.nn.site_tree.embed_site_info import EmbedSiteInfoBidir1
from rjt_rl.nn.tree.tree_encoder import MsgDict, TreeEncoder
from rjt_rl.rjt.consts import MAX_NB
from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.mol_tree_node import MolTreeNode
from rjt_rl.rjt.utils import get_root_batch, index_tensor, mask_invalid_nodes, pad_nodes
from rjt_rl.rjt.vocab import Vocab

logger = logging.getLogger(__name__)


class DataProto(Protocol):
    def get_mol_batch(self) -> Sequence[MolTree]:
        pass


class SiteTreeEncoder(TreeEncoder):
    """
    JunctionTree encoder with site information ver.2
    Use embed_site's method for site info encoding
    """

    def __init__(
        self,
        vocab: Vocab,
        hidden_size: int,
        node_feature_size: int,
        edge_feature_size: int,
        embedding: Optional[nn.Embedding] = None,
        embed_site: Optional[EmbedSiteInfoBidir1] = None,
        rnn_cls: Type[EdgeTreeGRU] = EdgeTreeGRU,
    ):
        self._debug = False
        self._dump = False

        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size

        # GRU (with edge feature)
        gru: nn.Module = rnn_cls(
            hidden_size, self.node_feature_size, self.edge_feature_size
        )

        super().__init__(
            vocab, hidden_size, node_feature_size, embedding=embedding, gru=gru
        )

        if embed_site is None:
            self.embed_site = EmbedSiteInfoBidir1(out_size=edge_feature_size)
        else:
            self.embed_site = embed_site

        self.We = nn.Linear(edge_feature_size, hidden_size)

    def forward(
        self,
        data: Sequence[MolTree] | DataProto,
        aggregate_all: bool = False,
    ) -> tuple[MsgDict, torch.Tensor]:
        device = self.device()

        if isinstance(data, Sequence):
            mol_batch = data
        else:
            mol_batch = data.get_mol_batch()
        root_batch = get_root_batch(mol_batch)

        orders = self.calc_prop_order(mol_batch, up_only=not aggregate_all)

        h: MsgDict = {}
        max_depth = max([len(x) for x in orders])
        padding = torch.zeros(self.hidden_size, device=device)

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            if self._debug:
                xxx = [(m[0].idx, m[1].idx) for m in prop_list]
                yyy = set(xxx)
                if len(xxx) != len(yyy):
                    raise RuntimeError(
                        f"inconsistent prop list size: len {len(xxx)} != set{len(yyy)}"
                    )

            cur_x = []
            cur_h_nei = []
            cur_site = []
            for node_x, node_y in prop_list:
                x, y = node_x.safe_idx, node_y.safe_idx
                cur_x.append(node_x.safe_wid)

                h_nei = []
                site = []
                for node_z in node_x.neighbors:
                    z = node_z.safe_idx
                    if z == y:
                        continue
                    h_nei.append(h[(z, x)])
                    site.append(self.embed_site.encode_site_node(node_z, node_x))

                h_nei = pad_nodes(h_nei, padding, MAX_NB)
                cur_h_nei.extend(h_nei)

                site_t = pad_nodes(site, self.embed_site.padding, MAX_NB)
                cur_site.append(site_t)

            cur_x_t = index_tensor(cur_x, device)
            cur_x_te = self.embedding(cur_x_t)
            cur_h_nei_t = torch.cat(cur_h_nei, dim=0).view(-1, MAX_NB, self.hidden_size)

            cur_site_t = index_tensor(cur_site, device)
            cur_site_te = self.embed_site(cur_site_t)

            new_h = self.gru(cur_x_te, cur_site_te, cur_h_nei_t)

            for i, m in enumerate(prop_list):
                x, y = m[0].safe_idx, m[1].safe_idx
                h[(x, y)] = new_h[i]

        if aggregate_all:
            max_nodes = max([len(mol_tree.nodes) for mol_tree in mol_batch])
            all_nodes: list[Optional[MolTreeNode]] = []
            for mol_tree in mol_batch:
                all_nodes.extend(mol_tree.nodes)
                pad_len = max_nodes - len(mol_tree.nodes)
                all_nodes.extend([None] * pad_len)

            node_sizes = index_tensor(
                [len(mol_tree.nodes) for mol_tree in mol_batch], device
            )

            out_vecs = self.node_aggregate(all_nodes, h)

            out_vecs = out_vecs.view(len(mol_batch), max_nodes, self.hidden_size)
            out_vecs = mask_invalid_nodes(out_vecs, node_sizes)
        else:
            out_vecs = self.node_aggregate(root_batch, h)

        return h, out_vecs
