from rdkit import Chem
from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.MolStandardize.standardize import Standardizer


def uncharge(mol: Chem.Mol) -> Chem.Mol:
    mol = Uncharger().uncharge(mol)
    return mol


def standardize(mol: Chem.Mol) -> Chem.Mol:
    return Standardizer().standardize(mol)
