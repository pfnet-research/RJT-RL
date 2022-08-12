from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import numpy as np
import torch
from rdkit import Chem
from torch.distributions import Categorical

from .path_like import AnyPathLikeType

logger = logging.getLogger(__name__)


def read_sdf(fname: AnyPathLikeType) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(str(fname), sanitize=True, removeHs=False)
    for mol in suppl:
        if mol is not None:
            return mol


def write_sdf(
    mol: Chem.Mol, fname: AnyPathLikeType, conf_id: Optional[int] = None
) -> None:
    w = Chem.SDWriter(str(fname))
    if conf_id is None:
        w.write(mol)
    else:
        w.write(mol, confId=conf_id)
    w.close()


def get_crd_array(mol: Chem.Mol, ignore_h: bool = True) -> np.ndarray:
    if ignore_h:
        mol = copy.deepcopy(mol)
        mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer(0)
    if conf is None:
        raise RuntimeError("cannot get conformer from mol")
    crd = np.array(conf.GetPositions())
    return crd


def sample_categorical_logits(
    logits: torch.Tensor, use_cpu: bool = False
) -> torch.Tensor:
    device = None
    if use_cpu:
        device = logits.device
        logits = logits.cpu()

    result: torch.Tensor
    try:
        result = Categorical(logits=logits).sample()  # type: ignore
    except RuntimeError:
        logger.error(f"logits={logits}")
        raise

    if use_cpu:
        result = result.to(device)

    return result


def sample_categorical(distrib: Categorical, use_cpu: bool = False) -> torch.Tensor:
    return sample_categorical_logits(distrib.logits, use_cpu)


def sample_categorical_logits_multi(
    logits: torch.Tensor, num_samples: int, use_cpu: bool = False
) -> torch.Tensor:
    device = None
    if use_cpu:
        device = logits.device
        logits = logits.cpu()

    logits = logits.clone()
    logits_dim = logits.dim()
    logit_mask_value = torch.min(logits) - 1e10

    result: torch.Tensor
    result_list = []
    for _ in range(num_samples):
        result = Categorical(logits=logits).sample()  # type: ignore
        result_list.append(result)
        result2 = torch.unsqueeze(result, -1)
        logits.scatter_(logits_dim - 1, result2, logit_mask_value)

    result_tensor = torch.stack(result_list, dim=logits_dim - 1)
    if use_cpu:
        result_tensor = result_tensor.to(device)

    return result_tensor


def sample_multinomial(logits: Any, nsamp: int) -> torch.Tensor:
    device = logits.device
    sampled = []
    logit_mask: float = min(logits) - 1e10
    for _ in range(nsamp):
        mod_logits_f: list[float] = []
        for j in range(len(logits)):
            if j in sampled:
                mod_logits_f.append(logit_mask)
            else:
                mod_logits_f.append(logits[j])
        mod_logits = torch.tensor(mod_logits_f)

        probs = torch.nn.Softmax()(mod_logits)
        result = torch.multinomial(probs, 1).item()

        sampled.append(result)
    return torch.tensor(sampled, device=device)
