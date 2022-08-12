from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.utils import index_tensor, set_batch_node_id
from rjt_rl.rjt.vocab import Vocab

from .expert_dataset_builder2 import FragDataType

logger = logging.getLogger(__name__)


class ListFromMolTree:
    actions: Optional[torch.Tensor]

    def __init__(self, batch: Sequence[MolTree]):
        self.mol_batch = batch
        self.actions = None

    def get_mol_batch(self) -> Sequence[MolTree]:
        return self.mol_batch

    def get_encoder_data(self) -> "ListFromMolTree":
        return self

    def get_decoder_data(self) -> None:
        raise NotImplementedError()

    def build_batch(self) -> None:
        pass

    def to(self, device: torch.device) -> "ListFromMolTree":
        if self.actions is not None:
            self.actions = self.actions.to(device)
        return self


class PretrainDatasetCollator:
    def __call__(self, batch: Sequence[FragDataType]) -> ListFromMolTree:
        if batch[0] is None:
            return None  # type: ignore

        moltrees = [i["state"] for i in batch]
        actions = np.stack([i["action"] for i in batch])
        actions_t = index_tensor(actions)
        set_batch_node_id(moltrees)
        bd = ListFromMolTree(moltrees)
        bd.actions = actions_t
        return bd


class RLDatasetCollator:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def __call__(
        self, states: Iterable[Any], device: Any, phi: Callable[[Any], MolTree]
    ) -> ListFromMolTree:
        mol_batch = [phi(s) for s in states]
        for m in mol_batch:
            m.set_vocab_wid(self.vocab)

        bd = ListFromMolTree(mol_batch)

        return bd
