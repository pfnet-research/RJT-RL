from __future__ import annotations

import logging
from typing import Any, Optional

from .states import State

logger = logging.getLogger(__name__)


class BaseHistory:
    backup_freq: Optional[int]

    def __init__(self) -> None:
        self.backup_freq = None

    def check_and_backup_history(self, episode: int) -> None:
        if self.backup_freq is not None and self.backup_freq > 0:
            if episode % self.backup_freq == 0:
                self.flush_history(update=True)

    def register_entry(self, state: State, info: dict[str, Any]) -> None:
        pass

    def flush_history(self, update: bool = False) -> None:
        pass
