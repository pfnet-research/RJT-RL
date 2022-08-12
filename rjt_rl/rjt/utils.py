from __future__ import annotations

import itertools
import logging
from typing import Any, Iterable, Optional, TypeVar

import torch
import torch.nn as nn

from .consts import MAX_NB
from .mol_tree import MolTree
from .mol_tree_node import MolTreeNode

logger = logging.getLogger(__name__)
T = TypeVar("T")


def index_tensor(data: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None:
        tensor = torch.tensor(data, dtype=torch.long)
    else:
        tensor = torch.tensor(data, device=device, dtype=torch.long)
    return tensor


def dump_tensor(h: torch.Tensor, check_nan: bool = False) -> str:
    if check_nan:
        out = []
        sizes = list(h.shape[:-1])
        sizes2 = [range(i) for i in sizes]
        for it in itertools.product(*sizes2):
            if torch.isnan(h[it].mean()):
                avg = h[it].mean()
                std = h[it].std()
                out.append(f"{it} m:{avg:.3f} s:{std:.3f}")
        return "\n".join(out)
    else:
        if len(h.shape) == 1:
            avg = h.mean()
            std = h.std()
            return f"m:{avg:.3f} s:{std:.3f}"
        else:
            out = []
            sizes = list(h.shape[:-1])
            sizes2 = [range(i) for i in sizes]
            for it in itertools.product(*sizes2):
                avg = h[it].mean()
                std = h[it].std()
                out.append(f"{it} m:{avg:.3f} s:{std:.3f}")
            return "\n".join(out)


def dump_msgs(msgs: dict[Any, torch.Tensor]) -> str:
    if len(msgs) == 0:
        return "(none)"
    out = ""
    for k in msgs.keys():
        h = dump_tensor(msgs[k])
        out += f"{k} {h}\n"
    return out


def get_root_batch(mol_batch: Iterable[MolTree]) -> list[MolTreeNode]:
    return [mol_tree.get_root_node() for mol_tree in mol_batch]


def set_batch_node_id(mol_batch: Any) -> None:
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            tot += 1


def check_node_idx(mol_batch: Any) -> None:
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            if not hasattr(node, "idx"):
                logger.error("ERROR!! mol tree node does not have idx attribute.")
                logger.error(mol_tree.dump_str())
                return


def filter_logits(
    h: torch.Tensor,
    nsizes: torch.Tensor,
    device: Optional[torch.device] = None,
    padding: float = -1.0e10,
) -> torch.Tensor:
    if device is None:
        # Use the same device as "h"
        device = h.device

    nsizes = nsizes[:, None]

    mask = torch.arange(h.shape[1], device=device)[None, :]
    mask = mask.repeat((h.shape[0], 1))
    mask = mask < nsizes

    dummy_scores = torch.full_like(h, fill_value=padding, device=device)

    h = torch.where(mask, h, dummy_scores)

    return h


def mask_invalid_nodes(
    h: torch.Tensor, nsizes: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    if device is None:
        # Use the same device as "h"
        device = h.device

    nsizes = nsizes[:, None]

    mask = torch.arange(h.shape[1], device=device)[None, :]
    mask = mask.repeat((h.shape[0], 1))
    mask = mask < nsizes
    mask = mask[:, :, None]

    dummy_scores = torch.zeros_like(h, device=device)
    h = torch.where(mask, h, dummy_scores)

    return h


def pad_nodes(nei: list[T], padding: T, max_nb: int = MAX_NB) -> list[T]:
    pad_len = max_nb - len(nei)
    if pad_len > 0:
        nei.extend([padding] * pad_len)
    elif pad_len < 0:
        logger.error(f"pad_len < 0 !!! -- pad_len: {pad_len}, len(nei): {len(nei)}")
        nei = nei[:max_nb]
        logger.error(f"truncate to MAX_NB: {len(nei)}")
    assert len(nei) == max_nb
    return nei


class Module(nn.Module):
    def device(self) -> torch.device:
        if not hasattr(self, "device_"):
            self.device_ = torch.device(next(self.parameters()).device)
        return self.device_
