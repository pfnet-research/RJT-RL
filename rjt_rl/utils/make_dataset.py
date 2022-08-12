from __future__ import annotations

import logging
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import rdkit.Chem as Chem

from rjt_rl.rjt.chemutils import (
    dump_mol,
    get_clique_mapping,
    get_clique_mol,
    get_kekule_mol,
    get_kekule_smiles,
)
from rjt_rl.rjt.tree_decomp import tree_decomp

from .path_like import AnyPathLikeType

logger = logging.getLogger(__name__)

# TODO: config?
allowed_elems: list[str] = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br"]


def process_mol(
    mol: Chem.Mol,
    no_brg_rings: bool,
    cset: Optional[Counter[str]] = None,
    vval_dic: Optional[dict[str, Any]] = None,
    num_ring_atoms: int = 8,
) -> tuple[Any, Any]:
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
        len(x) >= num_ring_atoms for x in ring_info.AtomRings()
    ):
        raise RuntimeError(
            f"num_ring_atoms exceeds {num_ring_atoms} {Chem.MolToSmiles(mol)}"
        )

    if any(atom.GetSymbol() not in allowed_elems for atom in mol.GetAtoms()):
        raise RuntimeError(f"unallowed atom in {Chem.MolToSmiles(mol)}")

    # Perform junction tree decomposition
    cliques, edges = tree_decomp(mol)

    binval_node = False
    logger.debug("MOL: %s", Chem.MolToSmiles(mol))
    for c in cliques:
        cmol = get_clique_mol(mol, c)
        cmol_s = get_kekule_smiles(cmol)
        cmol2 = get_kekule_mol(cmol_s)
        cmol2_s = get_kekule_smiles(cmol2)

        if cmol2_s != cmol_s:
            logger.error(f"inconsistent clique mols: {cmol2_s=} != {cmol_s=}")
            assert cmol2_s == cmol_s

        cmol = cmol2

        ring_info = cmol.GetRingInfo()

        if no_brg_rings and ring_info.NumRings() >= 2:
            print("Skipped: cmpx brdg ring", cmol_s, "in", Chem.MolToSmiles(mol))
            binval_node = True
            continue

        mapping = get_clique_mapping(mol, c, cmol)
        mapping = {v: k for k, v in mapping.items()}
        val_dict = {}
        for atom in cmol.GetAtoms():
            voc_idx = atom.GetIdx()
            orig_idx = mapping[voc_idx]
            orig_atom = mol.GetAtomWithIdx(orig_idx)
            exp_val = orig_atom.GetExplicitValence()
            valence = exp_val + orig_atom.GetImplicitValence()
            bonded_val = 0
            for bond in atom.GetBonds():
                bonded_val += bond.GetValenceContrib(atom)

            if bonded_val - math.floor(bonded_val) > 0.0:
                raise RuntimeError(
                    "Non integer valence in {}".format(Chem.MolToSmiles(mol))
                )

            val_dict[voc_idx] = int(valence - bonded_val)

        val_tup = tuple(val_dict.values())
        if cset is not None:
            cset.update([cmol_s])
        if vval_dic is not None:
            vval_dic[cmol_s].add(val_tup)

    if binval_node:
        raise RuntimeError(
            "mol {} contains invalid node(s)".format(Chem.MolToSmiles(mol))
        )

    edges = [(e[0], e[1]) for e in edges]

    return cliques, edges


def write_data(out_dat: Any, stem: Optional[str], pkl_idx: int) -> None:
    if stem is None:
        return
    fname = "{}_{:03d}.pkl".format(stem, pkl_idx)
    logger.info("Writing pkl: %s", fname)
    with open(fname, "wb") as f:
        pickle.dump(out_dat, f)


def write_vocab(
    output_vocab_csv: AnyPathLikeType, cset: Counter, vval_dic: dict[str, Any]
) -> None:
    output_vocab_csv = Path(output_vocab_csv)
    output_vocab_csv.parent.mkdir(exist_ok=True, parents=True)

    def sort_vocab(s: str) -> tuple[int, int, str]:
        mol = get_kekule_mol(s)
        ring_info = mol.GetRingInfo()
        nring = ring_info.NumRings()
        natoms = mol.GetNumAtoms()
        return (nring, natoms, s)

    if output_vocab_csv:
        voc = list(cset)
        voc = sorted(voc, key=sort_vocab)
        d = []
        for i in voc:
            d.append({"vocab": i, "count": cset[i], "valence": list(vval_dic[i])})
        logger.debug(d)

        out_df = pd.DataFrame(d)
        out_df.to_csv(output_vocab_csv)
