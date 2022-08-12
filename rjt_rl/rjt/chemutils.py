from __future__ import annotations

import logging
from typing import Optional

import rdkit
from packaging import version
from rdkit import Chem

from .rdkit_utils import RDKitUtils2020, RDKitUtilsBase

logger = logging.getLogger(__name__)

_impl: Optional[RDKitUtilsBase] = None


def get_impl() -> RDKitUtilsBase:
    global _impl
    if _impl is None:
        rdkit_ver = version.parse(rdkit.__version__)
        if rdkit_ver <= version.parse("2020.9.3"):
            logger.info("use RDKitUtils2020")
            _impl = RDKitUtils2020()
        else:
            raise RuntimeError(f"Unsupported RDKit version: {rdkit_ver}")
    return _impl


def dump_mol(mol: Chem.Mol, msg: Optional[str] = None) -> None:
    get_impl().dump_mol(mol, msg)


def get_debug_smiles(mol: Chem.Mol) -> str:
    return get_impl().get_debug_smiles(mol)


def get_clique_mol(mol: Chem.Mol, atoms: list[int]) -> Chem.Mol:
    return get_impl().get_clique_mol(mol, atoms)


def copy_edit_mol(mol: Chem.Mol) -> Chem.Mol:
    return get_impl().copy_edit_mol(mol)


def get_kekule_mol(smiles: str) -> Chem.Mol:
    return get_impl().get_kekule_mol(smiles)


def get_kekule_smiles(mol: Chem.Mol) -> str:
    return get_impl().get_kekule_smiles(mol)


def get_clique_mapping(
    amol: Chem.Mol, atom_ids: list[int], voc_mol: Chem.Mol
) -> dict[int, int]:
    return get_impl().get_clique_mapping(amol, atom_ids, voc_mol)


def check_mol(mol: Chem.Mol) -> None:
    get_impl().check_mol(mol)


def set_atommap(mol: Chem.Mol, num: int = 0) -> None:
    get_impl().set_atommap(mol, num)


def set_index_atommap(mol: Chem.Mol) -> None:
    get_impl().set_index_atommap(mol)


def atom_equal(a1: Chem.Atom, a2: Chem.Atom) -> bool:
    return get_impl().atom_equal(a1, a2)


def copy_atom(a1: Chem.Atom) -> Chem.Atom:
    return get_impl().copy_atom(a1)
