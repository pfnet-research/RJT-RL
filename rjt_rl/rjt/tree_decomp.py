from __future__ import annotations

from collections import defaultdict

import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

MST_MAX_WEIGHT = 100


def tree_decomp(
    mol: Chem.Mol, nmax_ringsize: int = 8, bbrdg: bool = True
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    n_atoms = mol.GetNumAtoms()

    if n_atoms == 1:
        return [[0]], []

    cliques = build_cliques(mol)

    if nmax_ringsize:
        if any(len(c) > nmax_ringsize for c in cliques):
            sm = Chem.MolToSmiles(mol)
            raise RuntimeError(f"Too many ring members: {sm}")

    nei_list: list[list[int]] = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    cliques = merge_rings(cliques, nei_list, bbrdg_xcpt=not bbrdg)
    cliques = [c for c in cliques if len(c) > 0]

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Build graph (tree) from cliques
    edges = build_graph(cliques, n_atoms, nei_list)

    if len(edges) == 0:
        return cliques, []  # edges

    # Compute Maximum Spanning Tree
    return compute_mst(cliques, edges)


def build_cliques(mol: Chem.Mol) -> list[list[int]]:

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    return cliques


def merge_rings(
    cliques: list[list[int]], nei_list: list[list[int]], bbrdg_xcpt: bool = False
) -> list[list[int]]:

    for i in range(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2:
                    continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    if bbrdg_xcpt:
                        raise RuntimeError("bridge cmpd detected")
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    return cliques


def build_graph(
    cliques: list[list[int]], n_atoms: int, nei_list: list[list[int]]
) -> list[tuple[int, int, int]]:
    edges = defaultdict(int)

    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]

        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]

        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:
            raise RuntimeError(f"complex rings len(rings)={len(rings)}")
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(
                            inter
                        )  # cnei[i] < cnei[j] by construction

    result = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    return result


def compute_mst(
    cliques: list[list[int]], edges: list[tuple[int, int, int]]
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    row, col, data = zip(*edges)
    n_clique = len(cliques)

    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))

    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    # edges: list of (cliqID1, cliqID2) if cliq1 and cliq2 are connected
    result_edges = [(row[i], col[i]) for i in range(len(row))]
    return (cliques, result_edges)
