from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors

from rjt_rl.rjt.assemble_mol import NodeMap
from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.vocab import Vocab

logger = logging.getLogger(__name__)


class State:
    smiles: Optional[str]
    mol_tree: Optional[MolTree]
    mol: Optional[Chem.Mol]
    idx_mol: Optional[Chem.Mol]
    score_dict: dict[str, Any]
    node_map: Optional[NodeMap]

    def __init__(
        self,
        smiles: Optional[str] = None,
        vocab: Optional[Vocab] = None,
        valid: bool = False,
    ):
        self.smiles = smiles
        if smiles is not None and vocab is not None:
            self.mol_tree = MolTree(smiles)
            self.mol_tree.set_wid_and_valences(vocab)
            self.mol_tree.nodes[0].nid = 0
            self.mol_tree.setup_attach_sites()
            self.mol = Chem.MolFromSmiles(smiles)
        else:
            self.mol_tree = None
            self.mol = None

        self.valid = valid
        self.num_decode_fail = 0
        self.node_map = None
        self.score = 0.0
        self.score_dict = {}
        self.idx_mol = None

    def dump(self, out: Callable[[str], None]) -> None:
        assert self.mol_tree is not None
        out(f"nnodes: {len(self.mol_tree.nodes)}")
        out(f"smiles: {self.smiles}")
        self.mol_tree.dump_tree(out)

    def show_mol_info(self) -> None:
        logger.info("final mol: %s", self.smiles)
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            molw = rdMolDescriptors._CalcMolWt(mol)
            logger.info("molw: %.2f", molw)
            logp = Crippen.MolLogP(mol)
            logger.info("logp: %.2f", logp)
            qed = QED.qed(mol)
            logger.info("qed: %.2f", qed)
        except Exception as e:
            logger.warning(f"calc mol props failed: {e}", exc_info=True)

        return

    def calc_mol_info(self) -> dict[str, Any]:
        assert self.mol_tree is not None
        assert self.mol is not None

        info = {}
        info["nnodes"] = len(self.mol_tree.nodes)

        if self.smiles is None and self.mol is None:
            logger.warning("smiles and mol is None")
            return info

        if self.mol is None:
            mol = Chem.MolFromSmiles(self.smiles)
        else:
            mol = self.mol

        try:
            info["molw"] = rdMolDescriptors._CalcMolWt(mol)
            info["logp"] = Crippen.MolLogP(mol)
            info["qed"] = QED.qed(mol)
        except Exception as e:
            logger.warning(f"calc mol props failed: {e}", exc_info=True)
            return info

        return info
