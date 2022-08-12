from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .base_history import BaseHistory
from .states import State

logger = logging.getLogger(__name__)


@dataclass
class SimpleHistoryConfig:
    backup_freq: Optional[int] = None
    history_dir: str = "history"
    csv_file_name: str = "results.csv"


class SimpleHistory(BaseHistory):
    @staticmethod
    def get_config_class() -> type[SimpleHistoryConfig]:
        return SimpleHistoryConfig

    @classmethod
    def from_config(cls, config: SimpleHistoryConfig) -> "SimpleHistory":
        return cls(config)

    records: list[dict[str, Any]]

    def __init__(self, config: SimpleHistoryConfig):
        super().__init__()
        self.config = config
        self.backup_freq = config.backup_freq
        self.history_dir = Path(config.history_dir)
        self.records = []

        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.history_dir / self.config.csv_file_name

    def get_result_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        return df

    def register_entry(self, state: State, info: dict[str, Any]) -> None:
        entry = info.copy()
        entry["smiles"] = state.smiles
        entry["num_decode_fail"] = state.num_decode_fail

        for key, value in state.score_dict.items():
            entry[key] = value

        self.records.append(entry)

    def flush_history(self, update: bool = False) -> None:
        logger.info(f"===== write history to: {self.csv_path} =====")
        df = self.get_result_df()
        df.to_csv(self.csv_path)
