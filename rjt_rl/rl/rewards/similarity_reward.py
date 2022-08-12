from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from rjt_rl.rl.envs.states import State

from .base_mol_reward import BaseMolReward, BaseMolRewardConfig


@dataclass
class SimilarityRewardConfig(BaseMolRewardConfig):
    target: str = MISSING
    scale: float = 10.0
    fp_radius: int = 2
    fp_bits: int = 2048


class SimilarityReward(BaseMolReward):
    @staticmethod
    def get_config_class() -> type[SimilarityRewardConfig]:
        return SimilarityRewardConfig

    def __init__(self, config: SimilarityRewardConfig):
        super().__init__(config)

        self.target_mol = Chem.MolFromSmiles(config.target)
        self.sim_scale = config.scale
        self.fp_radius = config.fp_radius
        self.fp_bits = config.fp_bits
        self.targ_fp = self.calc_fp(self.target_mol)

    def calc_fp(self, mol: Chem.Mol) -> Any:
        return AllChem.GetMorganFingerprint(
            mol, self.fp_radius, useCounts=True, useFeatures=True
        )

    def calc_similarity(self, mol: Chem.Mol) -> float:
        result: float = DataStructs.TanimotoSimilarity(self.calc_fp(mol), self.targ_fp)
        return result

    def calc_score(self, state: State) -> float:
        mol = state.mol
        sim = self.calc_similarity(mol)
        result: float = sim * self.sim_scale
        state.score_dict = {"total_score": result, "similarity": sim}
        return result
