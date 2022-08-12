from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import rdkit.Chem as Chem

from .chemutils import (
    get_clique_mapping,
    get_kekule_mol,
    get_kekule_smiles,
    set_atommap,
)
from .consts import MAX_SITES
from .vocab import Valences

logger = logging.getLogger(__name__)

SiteType = Union[int, Tuple[int, int]]
NodeIDFunc = Callable[["MolTreeNode"], int]
PropOrder = List[List[Tuple["MolTreeNode", "MolTreeNode"]]]
TraceOrder = List[Tuple["MolTreeNode", "MolTreeNode", int]]

T = TypeVar("T")


def not_none(x: Optional[T]) -> T:
    assert x is not None
    return x


class MolTreeNode(object):

    mol: Chem.Mol
    smiles: str
    clique: list[int]
    neighbors: list[MolTreeNode]
    parent: Optional[Any]
    node_slots: dict[int, SiteType]
    node_probs: dict[Any, Any]
    site_used: list[int]
    nid: Optional[int]
    wid: Optional[int]

    label: Optional[str]
    idx: Optional[int]
    depth1: Optional[int]
    depth: Optional[int]
    is_leaf: Optional[bool]
    valence: Optional[Valences]

    def __init__(
        self,
        smiles: Optional[str] = None,
        mol: Optional[Chem.Mol] = None,
        clique: Optional[list[int]] = None,
    ):
        if mol is None and smiles is not None:
            # smiles --> mol
            self.smiles = smiles
            self.mol = get_kekule_mol(smiles)
        elif mol is not None and smiles is None:
            # mol --> smiles
            self.mol = mol
            self.smiles = get_kekule_smiles(mol)
        elif mol is None and smiles is None:
            # null node
            self.smiles = ""
            self.mol = get_kekule_mol("")
        else:
            # both mol and smiles specified
            self.smiles = smiles
            self.mol = mol

        if clique is None:
            self.clique = []
        else:
            # copy
            self.clique = [x for x in clique]
        self.neighbors = []
        self.parent = None

        self.node_slots = {}
        self.node_probs = {}

        self.site_used = [0] * self.size()
        self.nid = None
        self.wid = None

        self.label = None
        self.idx = None
        self.depth1 = None
        self.depth = None
        self.is_leaf = None
        self.valence = None

    def size(self) -> int:
        result: int = self.mol.GetNumAtoms()
        return result

    def is_ring(self) -> bool:
        return self.size() > 2

    def is_bond(self) -> bool:
        return self.size() == 2

    def is_singleton(self) -> bool:
        return self.size() == 1

    def has_node_probs(self) -> bool:
        if len(self.node_probs) == 0:
            return False
        return True

    @property
    def safe_idx(self) -> int:
        return not_none(self.idx)

    @property
    def safe_nid(self) -> int:
        return not_none(self.nid)

    @property
    def safe_wid(self) -> int:
        return not_none(self.wid)

    def add_neighbor(self, nei_node: MolTreeNode) -> None:
        self.neighbors.append(nei_node)

    def setup_attach_sites(
        self, original_mol: Chem.Mol, max_sites: int = MAX_SITES
    ) -> None:
        c1 = self.clique
        rmap = get_clique_mapping(original_mol, c1, self.mol)

        if len(c1) > max_sites:
            orig_sm = Chem.MolToSmiles(original_mol)
            raise RuntimeError(
                f"Clique size {len(c1)} of {self.smiles} in {orig_sm}"
                f" exceeded MAX_SITES:{max_sites}"
            )

        set_atommap(self.mol)

        self.node_slots = {}
        self.node_probs = {}
        for nei_node in self.neighbors:
            assert nei_node.nid is not None
            c2 = nei_node.clique
            c2_rmap = get_clique_mapping(original_mol, c2, nei_node.mol)
            inter = list(set(c1) & set(c2))

            if len(inter) == 1:
                self_inter = rmap[inter[0]]
                if len(c1) > 2 and len(c2) > 2:
                    logger.debug(
                        f"spiro atom in {Chem.MolToSmiles(original_mol)}",
                    )
                    self.node_slots[nei_node.nid] = (self_inter, 0)
                else:
                    self.node_slots[nei_node.nid] = self_inter
            elif len(inter) == 2:
                self_a1, self_a2 = rmap[inter[0]], rmap[inter[1]]
                self_bond = self.mol.GetBondBetweenAtoms(self_a1, self_a2)
                nei_a1, nei_a2 = c2_rmap[inter[0]], c2_rmap[inter[1]]
                nei_bond = nei_node.mol.GetBondBetweenAtoms(nei_a1, nei_a2)

                self_flip = False
                if self_bond.GetBeginAtom().GetIdx() == self_a2:
                    self_flip = True

                nei_flip = False
                if nei_bond.GetBeginAtom().GetIdx() == nei_a2:
                    nei_flip = True

                if self_flip != nei_flip:
                    idir = -1
                else:
                    idir = 1

                self.node_slots[nei_node.nid] = (self_bond.GetIdx(), idir)
            else:
                raise RuntimeError(
                    "too many inter atoms in: " + Chem.MolToSmiles(original_mol)
                )

            site_info = self.node_slots[nei_node.nid]
            if type(site_info) is tuple:
                if site_info[0] >= max_sites:
                    raise RuntimeError(
                        f"site info exceeds max_sites({max_sites})"
                        f"mol:{Chem.MolToSmiles(original_mol)}, "
                    )
            else:
                assert type(site_info) is int
                if site_info >= max_sites:
                    raise RuntimeError(
                        f"site info exceeds max_sites({max_sites})"
                        f"mol:{Chem.MolToSmiles(original_mol)}, "
                    )

        self.site_used = [0] * len(self.clique)

    def dump_str(self) -> str:
        so = []
        so.append(f"SM:{self.smiles}")
        if self.clique:
            so.append(f"CQ:{self.clique}")
        if self.wid is None:
            so.append("wid:!NONE!")
        else:
            so.append(f"wid:{self.wid}")
        if self.label is not None:
            so.append(f"label:{self.label}")
        if self.nid is not None:
            so.append(f"nid:{self.nid}")
        if self.idx is not None:
            so.append(f"idx:{self.idx}")

        nei = []
        for i in self.neighbors:
            if i.nid is not None:
                nei.append(i.nid)
            else:
                nei.append("?")
        so.append(f"NB:{nei}")

        if self.node_slots is not None:
            so.append(f"site:{self.node_slots}")
        return ",".join(so)

    def dump(self) -> None:
        logger.debug(self.dump_str())

    def dump_tree(
        self,
        out: Optional[Callable[[Any], None]] = None,
        fa_node: Optional[MolTreeNode] = None,
        depth: int = 0,
    ) -> None:
        idx = self.nid
        if (
            fa_node is not None
            and len(fa_node.node_slots) > 0
            and len(self.node_slots) > 0
        ):
            pm = None
            if fa_node.nid in self.node_slots:
                pm = self.node_slots[fa_node.nid]
            pp = None
            if self.nid in fa_node.node_slots:
                pp = fa_node.node_slots[self.nid]
            s = f"{pp}/{pm} {idx}:{self.smiles}({self.wid})"
        else:
            s = f"{idx}:{self.smiles}({self.wid})"
        if self.depth1 is not None:
            s += f" {self.depth1=}"
        if out is None:
            print("  " * depth + s)
        else:
            out("  " * depth + s)
        for nei in self.neighbors:
            if nei is fa_node:
                continue
            nei.dump_tree(out=out, fa_node=self, depth=depth + 1)

    def __repr__(self) -> str:
        return self.dump_str()

    def collect_nodes(self, fa_node: Optional[MolTreeNode] = None) -> list[MolTreeNode]:
        res = []
        res.append(self)
        for nei in self.neighbors:
            if nei is fa_node:
                continue
            res.extend(nei.collect_nodes(fa_node=self))
        return res
