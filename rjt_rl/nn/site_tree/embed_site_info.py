from __future__ import annotations

import logging

import torch
import torch.nn as nn

from rjt_rl.rjt.consts import MAX_SITES
from rjt_rl.rjt.mol_tree_node import MolTreeNode, SiteType

logger = logging.getLogger(__name__)

PAD_IDX = MAX_SITES * 3

use_bond_site = False


class EmbedSiteInfoBidir1(nn.Module):

    padding = (PAD_IDX, PAD_IDX)

    def __init__(self, out_size: int):
        super().__init__()
        logger.debug(f"PAD_IDX: {PAD_IDX}")
        num_emb = MAX_SITES * 3 + 1
        logger.debug("num_emb: {num_emb}")
        self.embed_site = nn.Embedding(num_emb, out_size // 2, padding_idx=PAD_IDX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embed_site(x)

        shape = list(y.shape)
        shape = shape[:-1]
        shape[-1] = -1
        result: torch.Tensor = y.reshape(*shape)

        return result

    @staticmethod
    def _encode_site_info_impl(sval: SiteType) -> int:
        if isinstance(sval, tuple):
            return sval[0] + (sval[1] + 1) * MAX_SITES
        else:
            return sval

    @classmethod
    def _encode_site_node_impl(cls, node_z: MolTreeNode, node_x: MolTreeNode) -> int:
        try:
            sval = node_z.node_slots[node_x.safe_nid]
        except Exception:
            logger.error(f"node_z.node_slots: {node_z.node_slots}")
            logger.error(f"node_x.nid: {node_x.nid}")
            raise
        res = cls._encode_site_info_impl(sval)
        return res

    @classmethod
    def encode_site_node(
        cls, node_z: MolTreeNode, node_x: MolTreeNode
    ) -> tuple[int, int]:
        res1 = PAD_IDX
        res2 = PAD_IDX
        if use_bond_site or node_z.is_ring():
            res1 = cls._encode_site_node_impl(node_z, node_x)
        if use_bond_site or node_x.is_ring():
            res2 = cls._encode_site_node_impl(node_x, node_z)

        if res1 == PAD_IDX:
            res1 = res2
        elif res2 == PAD_IDX:
            res2 = res1

        return (res1, res2)

    @staticmethod
    def targ_size(node_z: MolTreeNode, node_x: MolTreeNode) -> int:
        return max(node_z.size(), node_x.size())
