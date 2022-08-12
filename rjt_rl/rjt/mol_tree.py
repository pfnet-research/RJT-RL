from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from rdkit import Chem

from .chemutils import get_clique_mol, get_kekule_mol, set_atommap
from .consts import MAX_SITES
from .mol_tree_node import MolTreeNode, NodeIDFunc, PropOrder, TraceOrder
from .tree_decomp import tree_decomp
from .tree_utils import calc_prop_order2
from .vocab import Vocab

logger = logging.getLogger(__name__)


class MolTree(object):

    smiles: str
    mol: Chem.Mol
    nodes: list[MolTreeNode]
    prop_order: Optional[PropOrder]
    trace_order: Optional[TraceOrder]

    def __init__(
        self,
        smiles: str,
        cliques: Optional[list[list[int]]] = None,
        edges: Optional[list[tuple[int, int]]] = None,
    ):
        self.smiles = smiles
        self.mol = get_kekule_mol(smiles)
        self.prop_order = None

        if cliques is None or edges is None:
            cliques, edges = tree_decomp(self.mol)

        self.nodes = []
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(mol=cmol, clique=c)

            node.parent = self
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        self.setup_nodes(root)

    @classmethod
    def from_nodes(
        cls,
        root_node: MolTreeNode,
        pred_nodes: list[MolTreeNode],
        set_nid: bool = True,
        reset_slots: bool = False,
    ) -> MolTree:
        obj = cls(smiles="")
        obj.nodes = pred_nodes
        root = pred_nodes.index(root_node)
        obj.setup_nodes(root, set_nid)
        if reset_slots:
            for node in obj.nodes:
                slt_new = {}
                for nei in node.neighbors:
                    assert nei.nid is not None
                    assert nei.idx is not None
                    slt_new[nei.nid] = node.node_slots[nei.idx]
                node.node_slots = slt_new
        return obj

    @classmethod
    def from_smiles(cls, sm: str, vocab: Vocab, attach_sites: bool = True) -> MolTree:
        mol_tree = cls(sm)
        mol_tree.set_vocab_wid(vocab)
        mol_tree.set_valences(vocab)
        if attach_sites:
            mol_tree.setup_attach_sites()
        return mol_tree

    def setup_nodes(self, root: int = 0, set_nid: bool = True) -> None:
        if root > 0:
            self.set_root_node(root)

        if set_nid:
            for i, node in enumerate(self.nodes):
                node.nid = i + 1
                if len(node.neighbors) > 1:
                    set_atommap(node.mol, node.nid)
                node.is_leaf = len(node.neighbors) == 1

    def get_root_node(self) -> MolTreeNode:
        return self.nodes[0]

    def set_root_node(self, root: int) -> None:
        if root != 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

    def set_vocab_wid(self, vocab: Vocab, overwrite: bool = False) -> None:
        for node in self.nodes:
            if not overwrite and node.wid is not None:
                continue
            try:
                node.wid = vocab.get_index(node.smiles)
            except KeyError:
                raise KeyError(f"node {node.smiles} not in vocab")

    def set_valences(self, vocab: Vocab) -> None:
        for node in self.nodes:
            node.valence = vocab.get_valence(node.safe_wid)

    def set_wid_and_valences(self, vocab: Vocab) -> None:
        for node in self.nodes:
            node.wid = vocab.get_index(node.smiles)
            logger.debug("smiles: %s wid: %s", node.smiles, node.wid)
            node.valence = vocab.get_valence(node.wid)

    def setup_attach_sites(self, max_sites: int = MAX_SITES) -> None:
        for node in self.nodes:
            node.setup_attach_sites(self.mol, max_sites)

    def size(self) -> int:
        return len(self.nodes)

    def dump_tree(self, out: Optional[Callable[[Any], None]] = None) -> None:
        self.nodes[0].dump_tree(out)

    def dump(self, out: Optional[Callable[[Any], None]] = None) -> None:
        if out is None:
            out = logger.debug
        out(self.smiles)
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            bt = str(bond.GetBondType())
            if bt == "SINGLE":
                bs = "-"
            elif bt == "DOUBLE":
                bs = "="
            elif bt == "TRIPLE":
                bs = "#"
            else:
                bs = "."
            out(f"{a1.GetSymbol()} {a1.GetIdx()} {bs} {a2.GetSymbol()} {a2.GetIdx()}")

        for i, node in enumerate(self.nodes):
            out(i)
            node.dump()

    def dump_str(self) -> str:
        ss = []
        ss.append(f"smiles: {self.smiles}")

        for i, node in enumerate(self.nodes):
            ss.append(f"{i} {node.dump_str()}")

        return "\n".join(ss)

    def calc_prop_order(
        self,
        up_only: bool,
        id_fn: NodeIDFunc = lambda x: x.safe_nid,
    ) -> PropOrder:
        order1, order2 = calc_prop_order2(self.get_root_node(), id_fn)
        if up_only:
            result = order2[::-1]
        else:
            result = order2[::-1] + order1
        return result

    def set_prop_order(
        self,
        up_only: bool,
        id_fn: NodeIDFunc = lambda x: x.safe_nid,
    ) -> None:
        self.prop_order = self.calc_prop_order(up_only, id_fn)
        self.depth = len(self.prop_order)

    def get_prop_order(self) -> PropOrder:
        assert self.prop_order is not None
        return self.prop_order


def sort_nodes(node: MolTreeNode) -> tuple[int, int]:
    return (len(node.neighbors), node.safe_wid)


def is_tree_match(
    node1: MolTreeNode,
    node2: MolTreeNode,
    fa_node1: Optional[MolTreeNode] = None,
    fa_node2: Optional[MolTreeNode] = None,
    nwmatch: int = 0,
) -> tuple[bool, int]:
    nch1 = len(node1.neighbors)
    nch2 = len(node2.neighbors)
    if nch1 != nch2:
        logger.debug("node1.len(%d) != node2.len(%d)", nch1, nch2)
        logger.debug("  node1: %s", node1.dump_str())
        logger.debug("  node2: %s", node2.dump_str())
        return False, 0

    if node1.safe_wid != node2.safe_wid:
        logger.debug("node1.wid(%d) != node2.wid(%d)", node1.wid, node2.wid)
        logger.debug("  node1: %s", node1.dump_str())
        logger.debug("  node2: %s", node2.dump_str())
    else:
        nwmatch += 1

    neis1 = [i for i in node1.neighbors if i is not fa_node1]
    neis2 = [i for i in node2.neighbors if i is not fa_node2]
    neis1 = sorted(neis1, key=sort_nodes)
    neis2 = sorted(neis2, key=sort_nodes)

    for nei1, nei2 in zip(neis1, neis2):
        bm, nwmatch = is_tree_match(nei1, nei2, node1, node2, nwmatch)
        if not bm:
            return False, 0

    return True, nwmatch
