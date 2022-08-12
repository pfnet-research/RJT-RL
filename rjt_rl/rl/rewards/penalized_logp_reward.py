from __future__ import annotations

import networkx as nx
from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP

from rjt_rl.rl.envs.states import State

from .base_mol_reward import BaseMolReward
from .sa_score import calculateScore


def calc_penalized_logp(mol: Chem.Mol) -> tuple[float, float, float, float]:
    # This impl is based on:
    #   https://github.com/bowenliu16/rl_graph_generation
    #   gym_molecule.envs.molecule.reward_penalized_log_p
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle, SA, log_p, cycle_score


class PenalizedLogPReward(BaseMolReward):
    def calc_score(self, state: State) -> float:
        mol = state.mol
        result, sa, log_p, cycle_score = calc_penalized_logp(mol)
        state.score_dict = {
            "total_score": result,
            "penalized_logp": result,
            "sa_score": sa,
            "log_p": log_p,
            "cycle_score": cycle_score,
        }
        return result
