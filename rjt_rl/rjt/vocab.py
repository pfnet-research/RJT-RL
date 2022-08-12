from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import rdkit.Chem as Chem

from rjt_rl.utils.path_like import AnyPathLikeType

logger = logging.getLogger(__name__)


Valences = List[Tuple[int]]


class Vocab(object):
    word_sizes: np.ndarray
    valences: Optional[list[Valences]]

    def __init__(self, smiles_list: list[str], valences: Optional[str] = None):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        if valences is None:
            self.valences = None
        else:
            self.valences = [eval(v) for v in valences]

        word_sizes = [Chem.MolFromSmiles(w).GetNumAtoms() for w in self.vocab]
        self.word_sizes = np.asarray(word_sizes)
        self.not_bond_mask = self.word_sizes != 2
        self.singleton_mask = self.word_sizes == 1
        logger.info(f"vocab size: {len(word_sizes)}")

    def get_index(self, smiles: str) -> int:
        return self.vmap[smiles]

    def get_smiles(self, idx: int) -> str:
        return self.vocab[idx]

    def get_word_size(self, idx: int) -> int:
        result: int = self.word_sizes[idx]
        return result

    def __len__(self) -> int:
        return len(self.vocab)

    def get_valence(self, idx: int) -> Valences:
        if self.valences is None:
            raise NotImplementedError("valence not available")
        return self.valences[idx]


def load_vocab(vocab_file: AnyPathLikeType) -> Vocab:
    vocab_df = pd.read_csv(vocab_file, index_col=0)
    if "valence" in vocab_df.columns:
        vocab = Vocab(vocab_df.vocab.values, vocab_df.valence.values)
    else:
        logger.warning(f"no valence column in vocab: {vocab_file}")
        vocab = Vocab(vocab_df.vocab.values, None)
    return vocab
