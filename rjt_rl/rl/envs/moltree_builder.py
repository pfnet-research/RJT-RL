from __future__ import annotations

import logging
import traceback
from typing import Optional

from rdkit import Chem

from rjt_rl.rjt.assemble_mol import NodeMap, decode_mol
from rjt_rl.rjt.chemutils import set_atommap
from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.mol_tree_node import MolTreeNode
from rjt_rl.rjt.vocab import Vocab
from rjt_rl.utils.mol_utils import standardize, uncharge

from .actions import Action2

logger = logging.getLogger(__name__)


def _mol_to_smiles(mol: Chem.Mol) -> str:
    mol = Chem.Mol(mol)
    set_atommap(mol)
    result: str = Chem.MolToSmiles(mol)
    return result


class MolTreeBuilder:
    def __init__(
        self,
        vocab: Vocab,
        standardize_mol: bool = False,
        neutralize_mol: bool = False,
    ):
        self.vocab = vocab
        self.action = Action2(vocab)
        self.strict_site_match = False
        self.show_decode_error = False

        self.standardize_mol = standardize_mol
        self.neutralize_mol = neutralize_mol

    def get_target_node(self, action: list[int], mol_tree: MolTree) -> MolTreeNode:
        node_x_nid = self.action.target_node(action)
        return mol_tree.nodes[node_x_nid]

    def get_word_sizes(self, action: list[int], mol_tree: MolTree) -> tuple[int, int]:
        node_x = self.get_target_node(action, mol_tree)
        sz_x = node_x.size()
        assert sz_x == self.vocab.get_word_size(node_x.safe_wid)

        wid = self.action.add_node(action)
        sz_y = self.vocab.get_word_size(wid)
        return sz_x, sz_y

    def check_word_sizes(self, action: list[int], mol_tree: MolTree) -> bool:
        sz_x, sz_y = self.get_word_sizes(action, mol_tree)
        if sz_x == 1 and sz_y == 1:
            return False

        return True

    def add_node(self, action: list[int], mol_tree: MolTree) -> bool:
        node_x_nid = self.action.target_node(action)

        node_x = mol_tree.nodes[node_x_nid]
        wid = self.action.add_node(action)
        wid_smi = self.vocab.get_smiles(wid)
        node_y = MolTreeNode(smiles=wid_smi)
        node_y.wid = wid
        node_y.valence = self.vocab.get_valence(wid)

        new_nid = max([i.safe_nid for i in mol_tree.nodes]) + 1
        node_y.nid = new_nid

        sz_x = node_x.size()
        sz_y = node_y.size()

        if sz_x == 1 and sz_y == 1:
            logger.warning("connect singleton-singleton nodes")
            return False

        if sz_x == 1 and sz_y > 2:
            logger.debug("connect singleton-ring nodes")
        if sz_x > 2 and sz_y == 1:
            logger.debug("connect ring-singleton nodes")

        is1 = self.action.site_info1(action, sz_x)
        is2 = self.action.site_info2(action, sz_y)

        try:
            node_x.neighbors.append(node_y)
            node_y.neighbors.append(node_x)
            mol_tree.nodes.append(node_y)

            if sz_x <= 2 and sz_y <= 2:
                node_x.node_slots[node_y.safe_nid] = 0
                node_y.node_slots[node_x.safe_nid] = 0
            elif sz_x > 2 and sz_y > 2:
                node_x.node_slots[node_y.safe_nid] = is1
                node_y.node_slots[node_x.safe_nid] = is2
            elif sz_x <= 2:
                node_x.node_slots[node_y.safe_nid] = 0
                node_y.node_slots[node_x.safe_nid] = is2[0]
            elif sz_y <= 2:
                node_x.node_slots[node_y.safe_nid] = is1[0]
                node_y.node_slots[node_x.safe_nid] = 0

        except Exception as e:
            logger.error(f"add_node failed: {e}")
            traceback.print_exc()
            return False

        return True

    def add_node_and_decode(
        self, action: list[int], mol_tree: MolTree
    ) -> Optional[tuple[Optional[Chem.Mol], Optional[NodeMap]]]:
        if not self.add_node(action, mol_tree):
            return None
        return self.decode_mol_tree(mol_tree)

    def decode_mol_tree(
        self, mol_tree: MolTree
    ) -> tuple[Optional[Chem.Mol], Optional[NodeMap]]:
        mol_tree.dump_tree(logger.debug)
        mol: Optional[Chem.Mol]
        nmap: Optional[NodeMap]
        try:
            mol, nmap = decode_mol(
                mol_tree,
                strict_site_match=self.strict_site_match,
                raise_all=True,
                require_nodemap=True,
            )
            if self.standardize_mol:
                orig_smi = _mol_to_smiles(mol)
                mol = standardize(mol)
                res_smi = _mol_to_smiles(mol)
                if res_smi != orig_smi:
                    logger.info(f"standardized: {orig_smi} ==> {res_smi}")
            if self.neutralize_mol:
                orig_smi = _mol_to_smiles(mol)
                mol = uncharge(mol)
                res_smi = _mol_to_smiles(mol)
                if res_smi != orig_smi:
                    logger.info(f"neutralized: {orig_smi} ==> {res_smi}")

        except Exception as e:
            if self.show_decode_error:
                logger.error("decode_mol_tree failed", exc_info=e)
                mol_tree.dump_tree()
            return None, None
        mol_tree.smiles = Chem.MolToSmiles(mol)

        return mol, nmap

    def dump_failed_action_info(self, action: list[int], mt: MolTree) -> None:
        logger.info(f"Action {self.action.dump_str(action)} failed")
        nodex = self.get_target_node(action, mt)
        logger.info(f"target node: {nodex}")
        mt.dump_tree()
