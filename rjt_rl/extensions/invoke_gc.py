from __future__ import annotations

import gc
import logging
import tracemalloc
from typing import Any, Optional

from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol
from pytorch_pfn_extras.training.extension import Extension

logger = logging.getLogger(__name__)


def gc_callback(phase: str, info: dict[str, Any]) -> None:
    logger.debug(f"{phase=}: {info=}")


class InvokeGC(Extension):
    prev_ss_: Optional[tracemalloc.Snapshot]

    def __init__(self, mem_check: bool = False) -> None:
        gc.disable()
        gc.collect()
        if gc_callback not in gc.callbacks:
            gc.callbacks.append(gc_callback)
        self.mem_check_ = mem_check
        self.prev_ss_ = None

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        if self.mem_check_:
            tracemalloc.start()

    def __call__(self, manager: ExtensionsManagerProtocol) -> Any:
        logger.debug("gc enabled: %s", gc.isenabled())
        logger.debug("StartGC")
        gc.collect()
        logger.debug("EndGC")

        if self.mem_check_:
            ss = tracemalloc.take_snapshot()
            if self.prev_ss_ is not None:
                top_stats = self.prev_ss_.compare_to(ss, "lineno")
                print("[ Top 10 differences ]")
                for stat in top_stats[:10]:
                    print(stat)
            self.prev_ss_ = ss
            tracemalloc.start()
        return
