from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple

from rdkit import Chem

from .chemutils import (
    atom_equal,
    check_mol,
    copy_atom,
    copy_edit_mol,
    get_debug_smiles,
    set_atommap,
    set_index_atommap,
)
from .consts import MAX_SITES
from .mol_tree import MolTree
from .mol_tree_node import MolTreeNode, SiteType

logger = logging.getLogger(__name__)
_NodeMapItem = Tuple[MolTreeNode, Dict[int, int]]
NodeMap = Dict[int, Dict[int, int]]


def encode_site_info(sval: SiteType) -> int:
    assert isinstance(sval, tuple)
    return sval[0] + (sval[1] + 1) * MAX_SITES


def get_valence(nei: MolTreeNode, j: int) -> int:
    assert nei.valence is not None
    if len(nei.valence) == 1:
        return nei.valence[0][j]

    cands = [nei.valence[i][j] for i in range(len(nei.valence))]
    res = max(cands)
    return res


def attach_mols(
    ctr_mol: Chem.Mol,
    neighbors: list[MolTreeNode],
    prev_nodes: list[MolTreeNode],
    nei_amap: Any,
) -> Chem.Mol:
    logger.debug("----------")
    logger.debug("attach_mols:")
    prev_nids = [node.safe_nid for node in prev_nodes]

    for nei_node in prev_nodes + neighbors:
        logger.debug("nei_node:", nei_node.dump_str())
        nei_id, nei_mol = nei_node.safe_nid, nei_node.mol

        amap = nei_amap[nei_id]

        for atom in nei_mol.GetAtoms():
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        if nei_mol.GetNumBonds() == 0:
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtom().GetIdx()]
                a2 = amap[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
                elif nei_id in prev_nids:  # father node overrides
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())

    return ctr_mol


def get_bonded_val(s_atom: Chem.Atom) -> int:
    bonded_val = 0
    for bond in s_atom.GetBonds():
        bonded_val += bond.GetValenceContrib(s_atom)
    return bonded_val


def build_amap_bond(
    cur_mol: Chem.Mol,
    cur_node: MolTreeNode,
    nei: MolTreeNode,
    cur_amap: dict[int, int],
    s_aid: int,
    n_aid: int,
    smol_size: int,
    nmol_size: int,
    strict_site_match: bool,
) -> dict[int, int]:
    amap = {}

    ##########
    # set scores (cur_node)

    if smol_size == 1:
        s_scores = [1.0]
    elif smol_size == 2:
        sss = sum(cur_node.site_used)
        logger.debug(f"cur_node.site_used: {cur_node.site_used}, sum: {sss}")
        if sss > 0:
            s_scores = [1 - i / sss for i in cur_node.site_used]
        else:
            s_scores = [0.5, 0.5]
        logger.debug(f"s_scores: {s_scores}")
    else:
        if cur_node.has_node_probs():
            s_scores = cur_node.node_probs[nei.safe_nid]
        else:
            s_scores = [0.0] * smol_size
            s_scores[s_aid] = 1.0

    ##########
    # set scores (nei_node)

    if nmol_size == 1:
        n_scores = [1.0]
    elif nmol_size == 2:
        n_scores = [0.5, 0.5]
    else:
        if nei.has_node_probs():
            n_scores = nei.node_probs[cur_node.safe_nid]
        else:
            # no prob --> use best scores
            n_scores = [0.0] * nmol_size
            n_scores[n_aid] = 1.0

    if len(n_scores) == 2:
        logger.debug(f"nei.site_used: {nei.site_used}")

    score_list = []
    for i in range(smol_size):
        if strict_site_match and s_scores[i] == 0.0:
            continue
        s_curmol_aid = cur_amap[i]
        s_atom = cur_mol.GetAtomWithIdx(s_curmol_aid)
        bonded_val = get_bonded_val(s_atom)

        for j in range(nmol_size):
            if strict_site_match and n_scores[j] == 0.0:
                continue
            n_atom = nei.mol.GetAtomWithIdx(j)
            if not atom_equal(n_atom, s_atom):
                continue
            nei_free_val = get_valence(nei, j)
            score = 0.0
            if nei_free_val < bonded_val:
                continue
            if nmol_size == 1:
                if nei_free_val > bonded_val:
                    score = (nei_free_val - bonded_val) * 10.0
                else:
                    continue

            score_list.append((i, j, s_scores[i] + n_scores[j] + score))

    if len(score_list) == 0:
        raise RuntimeError("cannot attach bond cur_node (no possible attachment)")

    score_list = sorted(score_list, key=lambda x: -x[2])
    s_aid = score_list[0][0]
    n_aid = score_list[0][1]

    cur_node.site_used[s_aid] += 1
    nei.site_used[n_aid] += 1

    amap[n_aid] = cur_amap[s_aid]
    return amap


def can_attach_rings(
    cur_mol: Chem.Mol,
    cur_node: MolTreeNode,
    nei: MolTreeNode,
    cur_amap: dict[int, int],
    s_bid: int,
    n_bid: int,
    idir: int,
) -> bool:
    self_bond = cur_node.mol.GetBondWithIdx(s_bid)
    s_a1 = self_bond.GetBeginAtom()
    s_a2 = self_bond.GetEndAtom()

    s_aid1 = s_a1.GetIdx()
    s_aid2 = s_a2.GetIdx()

    s_aid1 = cur_amap[s_aid1]
    s_aid2 = cur_amap[s_aid2]
    s_a1 = cur_mol.GetAtomWithIdx(s_aid1)
    s_a2 = cur_mol.GetAtomWithIdx(s_aid2)
    self_bond = cur_mol.GetBondBetweenAtoms(s_aid1, s_aid2)

    nei_bond = nei.mol.GetBondWithIdx(n_bid)
    n_a1 = nei_bond.GetBeginAtom()
    n_a2 = nei_bond.GetEndAtom()

    if idir < 0:
        n_a1, n_a2 = n_a2, n_a1

    if not atom_equal(n_a1, s_a1) or not atom_equal(n_a2, s_a2):
        return False

    bonded_val = get_bonded_val(s_a1)
    bt = nei_bond.GetValenceContrib(n_a1)
    nei_free_val = get_valence(nei, n_a1.GetIdx()) + bt
    if nei_free_val < bonded_val:
        return False

    bonded_val = get_bonded_val(s_a2)
    bt = nei_bond.GetValenceContrib(n_a2)
    nei_free_val = get_valence(nei, n_a2.GetIdx()) + bt
    if nei_free_val < bonded_val:
        return False

    return True


def build_amap_ring(
    cur_mol: Chem.Mol,
    cur_node: MolTreeNode,
    nei: MolTreeNode,
    cur_amap: dict[int, int],
    self_attach_aids: tuple[int, int],
    nei_attach_aids: tuple[int, int],
    smol_size: int,
    nmol_size: int,
    strict_site_match: bool,
) -> dict[int, int]:
    if self_attach_aids[1] == 0 or nei_attach_aids[1] == 0:
        return build_amap_bond(
            cur_mol,
            cur_node,
            nei,
            cur_amap,
            self_attach_aids[0],
            nei_attach_aids[0],
            smol_size,
            nmol_size,
            strict_site_match,
        )

    try_smol_sites: Sequence[int] = range(smol_size)
    try_nmol_sites: Sequence[int] = range(nmol_size)

    if cur_node.has_node_probs() and nei.has_node_probs():
        s_scores = cur_node.node_probs[nei.safe_nid]
        n_scores = nei.node_probs[cur_node.safe_nid]
    else:
        s_scores = [0.0] * (encode_site_info((smol_size, 1)) + 1)
        n_scores = [0.0] * (encode_site_info((nmol_size, 1)) + 1)
        s_scores[encode_site_info(self_attach_aids)] = 1.0
        n_scores[encode_site_info(nei_attach_aids)] = 1.0
        if strict_site_match:
            try_smol_sites = [self_attach_aids[0]]
            try_nmol_sites = [nei_attach_aids[0]]

    score_list = []
    for i in try_smol_sites:
        for j in try_nmol_sites:
            for idir in (-1, 1):
                if not can_attach_rings(cur_mol, cur_node, nei, cur_amap, i, j, idir):
                    continue
                ix = encode_site_info((i, idir))
                jx = encode_site_info((j, idir))
                score = s_scores[ix] + n_scores[jx]
                score_list.append((i, j, idir, score))

    if len(score_list) == 0:
        raise RuntimeError("cannot attach ring (score_list is empty)")

    def sort_fn(x: tuple[int, int, int, float]) -> float:
        return -x[3]

    ssl = sorted(score_list, key=sort_fn)
    s_bid, n_bid, idir, _ = ssl[0]
    return build_amap_ring_impl(cur_node, nei, cur_amap, (s_bid, idir), (n_bid, idir))


def build_amap_ring_impl(
    cur_node: Chem.Mol,
    nei: MolTreeNode,
    cur_amap: dict[int, int],
    self_attach_aids: tuple[int, int],
    nei_attach_aids: tuple[int, int],
) -> dict[int, int]:
    amap = {}
    s_bid, s_bdir = self_attach_aids
    n_bid, n_bdir = nei_attach_aids

    self_bond = cur_node.mol.GetBondWithIdx(s_bid)
    nei_bond = nei.mol.GetBondWithIdx(n_bid)

    s_a1 = self_bond.GetBeginAtom()
    s_a2 = self_bond.GetEndAtom()
    s_aid1 = s_a1.GetIdx()
    s_aid2 = s_a2.GetIdx()

    n_a1 = nei_bond.GetBeginAtom()
    n_a2 = nei_bond.GetEndAtom()

    if s_bdir == -1:
        assert n_bdir == -1
        n_a1, n_a2 = n_a2, n_a1

    n_aid1 = n_a1.GetIdx()
    n_aid2 = n_a2.GetIdx()

    amap[n_aid1] = cur_amap[s_aid1]
    amap[n_aid2] = cur_amap[s_aid2]

    return amap


def dfs_assm(
    cur_node: MolTreeNode,
    cur_mol: Chem.Mol,
    cur_amap: dict[int, int],
    strict_site_match: bool,
    par_node: Optional[MolTreeNode] = None,
    raise_all: bool = False,
    node_map_data: Optional[list[_NodeMapItem]] = None,
) -> Chem.Mol:
    cur_nid = cur_node.safe_nid
    par_nid = par_node.safe_nid if par_node is not None else -1
    smol_size = cur_node.size()
    if node_map_data is not None:
        node_map_data.append((cur_node, cur_amap))

    for nei in cur_node.neighbors:
        nei_nid = nei.safe_nid
        if nei_nid == par_nid:
            continue

        nmol_size = nei.size()
        try:
            self_attach_aids = cur_node.node_slots[nei_nid]
            nei_attach_aids = nei.node_slots[cur_nid]
        except KeyError:
            if raise_all:
                raise
            else:
                logger.debug("Invalid site info (ignored)")
            continue

        if smol_size <= 2 or nmol_size <= 2:
            if isinstance(self_attach_aids, tuple):
                sat_aid = self_attach_aids[0]
            else:
                sat_aid = self_attach_aids

            if isinstance(nei_attach_aids, tuple):
                nat_aid = nei_attach_aids[0]
            else:
                nat_aid = nei_attach_aids

            amap = build_amap_bond(
                cur_mol,
                cur_node,
                nei,
                cur_amap,
                sat_aid,
                nat_aid,
                smol_size,
                nmol_size,
                strict_site_match,
            )
        else:
            assert smol_size > 2 and nmol_size > 2
            if (
                type(self_attach_aids) is not tuple
                and type(nei_attach_aids) is not tuple
            ):
                # Both attach AIDs are not tuple-->Invalid site info
                if raise_all:
                    raise RuntimeError("Invalid site info")
                else:
                    logger.debug("Invalid site info (ignored)")
                continue

            if not isinstance(self_attach_aids, tuple):
                assert isinstance(nei_attach_aids, tuple)
                sat_aids = (self_attach_aids, nei_attach_aids[1])
            else:
                sat_aids = self_attach_aids

            if not isinstance(nei_attach_aids, tuple):
                assert isinstance(self_attach_aids, tuple)
                nat_aids = (nei_attach_aids, self_attach_aids[1])
            else:
                nat_aids = nei_attach_aids

            amap = build_amap_ring(
                cur_mol,
                cur_node,
                nei,
                cur_amap,
                sat_aids,
                nat_aids,
                smol_size,
                nmol_size,
                strict_site_match,
            )

        nei_amap = {nei_nid: amap}
        cur_mol = attach_mols(cur_mol, [nei], [], nei_amap)

        # check mol validity after attachment
        check_mol(cur_mol)

        # rebuild atom mapping
        next_amap = {atom.GetIdx(): amap[atom.GetIdx()] for atom in nei.mol.GetAtoms()}

        # process nei recursively (depth-first order)
        cur_mol = dfs_assm(
            nei,
            cur_mol,
            next_amap,
            strict_site_match=strict_site_match,
            par_node=cur_node,
            raise_all=raise_all,
            node_map_data=node_map_data,
        )

    return cur_mol


def assemble(
    mol_tree: MolTree,
    strict_site_match: bool,
    raise_all: bool = False,
    require_nodemap: bool = False,
) -> tuple[Chem.Mol, NodeMap] | Chem.Mol:
    # determine start node for assemble
    start_node = None
    # At first, search a ring node
    for node in mol_tree.nodes:
        if node.is_ring():
            start_node = node
            break

    if start_node is None:
        # No ring node in the tree
        for node in mol_tree.nodes:
            if not node.is_bond():
                start_node = node
                break

    if start_node is None:
        start_node = mol_tree.get_root_node()

    # make initial atom mapping
    cur_mol = copy_edit_mol(start_node.mol)
    amap = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

    # perform depth-first assemble recursively
    node_map_data: list[_NodeMapItem] = []
    cur_mol = dfs_assm(
        start_node,
        cur_mol,
        cur_amap=amap,
        strict_site_match=strict_site_match,
        raise_all=raise_all,
        node_map_data=node_map_data,
    )

    if not require_nodemap:
        # remove mapping property
        set_atommap(cur_mol)
        return cur_mol

    # make node-atom index mapping
    set_index_atommap(cur_mol)
    node_map: NodeMap = dict()
    for n, m in node_map_data:
        node_map[n.safe_nid] = m

    return cur_mol, node_map


def decode_mol(
    mol_tree: MolTree,
    strict_site_match: bool = False,
    raise_all: bool = True,
    require_nodemap: bool = False,
) -> tuple[Chem.Mol, NodeMap] | Chem.Mol:
    result = assemble(
        mol_tree,
        strict_site_match,
        raise_all=raise_all,
        require_nodemap=require_nodemap,
    )
    if require_nodemap:
        cur_mol, node_map = result
    else:
        cur_mol = result
        node_map = None
    Chem.SanitizeMol(cur_mol)
    if require_nodemap:
        return cur_mol, node_map
    else:
        return cur_mol
