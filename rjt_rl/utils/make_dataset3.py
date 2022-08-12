from __future__ import annotations

import gc
import logging
from collections import Counter, defaultdict
from typing import Any, Optional

import joblib
import numpy as np

from rjt_rl.rjt.chemutils import get_kekule_mol
from rjt_rl.rjt.consts import MAX_SITES
from rjt_rl.rjt.mol_tree import MolTree

from .make_dataset import process_mol, write_data

logger = logging.getLogger(__name__)


def make_split_indices(ntotal: int, nsplit: int) -> list[tuple[int, int]]:
    nn = ntotal // nsplit
    remd = ntotal % nsplit
    sizes = []
    for i in range(nsplit):
        if i < remd:
            sizes.append(nn + 1)
        else:
            sizes.append(nn)
    logger.debug(f"sizes: {sizes}")
    assert ntotal == sum(sizes)
    cs = [0] + np.cumsum(sizes).tolist()
    indices = [(i, j) for i, j in zip(cs[:-1], cs[1:])]
    logger.debug(f"indices: {indices}")
    return indices


def process_smiles(
    smiles: str,
    no_brg_ring: bool = True,
    num_ring_atoms: int = 8,
    max_sites: int = MAX_SITES,
    vval_dic: Optional[dict[str, Any]] = None,
) -> tuple[Optional[MolTree], Optional[Counter]]:
    cur_cset: Counter[str] = Counter()
    mol = get_kekule_mol(smiles)
    try:
        cliques, edges = process_mol(
            mol,
            no_brg_ring,
            cur_cset,
            vval_dic=vval_dic,
            num_ring_atoms=num_ring_atoms,
        )
        mt = MolTree(smiles, cliques, edges)
        mt.setup_attach_sites(max_sites=max_sites)
    except RuntimeError as e:
        logger.warning(f"Cannot create moltree for {smiles}: {e}")
        return None, None

    return mt, cur_cset


def process_smiles_chunk(
    smiles_list: list[str],
    output_pkl_stem: str,
    output_pkl_id: int,
    no_brg_ring: bool = True,
    num_ring_atoms: int = 8,
    max_sites: int = MAX_SITES,
) -> tuple[Counter[str], dict[str, Any]]:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    out_dat: Optional[list[MolTree]] = []
    cset: Counter[str] = Counter()
    vval_dic: dict[str, Any] = defaultdict(set)

    for s in smiles_list:
        mt, cur_cset = process_smiles(
            smiles=s,
            no_brg_ring=no_brg_ring,
            num_ring_atoms=num_ring_atoms,
            max_sites=max_sites,
            vval_dic=vval_dic,
        )
        if mt is None:
            continue
        assert cur_cset is not None
        cset = cset + cur_cset

        assert out_dat is not None
        out_dat.append(mt)

    write_data(out_dat, output_pkl_stem, output_pkl_id)
    out_dat = None
    gc.collect()

    return cset, vval_dic


def process_smiles_list(
    smiles_list: list[str],
    output_pkl_stem: str,
    nsplit: int,
    no_brg_ring: bool = True,
    num_ring_atoms: int = 8,
    n_jobs: int = 1,
    job_verbose: int = 1,
) -> tuple[Counter[str], dict[str, Any]]:
    ntotal = len(smiles_list)
    indices = make_split_indices(ntotal, nsplit)
    logger.info(indices)

    dfns = [
        joblib.delayed(process_smiles_chunk)(
            smiles_list=smiles_list[ist:ien],
            output_pkl_stem=output_pkl_stem,
            output_pkl_id=i,
            no_brg_ring=no_brg_ring,
            num_ring_atoms=num_ring_atoms,
        )
        for i, (ist, ien) in enumerate(indices)
    ]

    results = joblib.Parallel(n_jobs=n_jobs, verbose=job_verbose)(dfns)

    all_cset: Optional[Counter[str]] = None
    all_val_dic: Optional[dict[str, Any]] = None
    for cset, val_dic in results:
        if all_cset is None:
            all_cset = cset
        else:
            all_cset = all_cset + cset

        if all_val_dic is None:
            all_val_dic = val_dic
        else:
            all_val_dic.update(val_dic)

    assert all_cset is not None
    assert all_val_dic is not None
    return all_cset, all_val_dic
