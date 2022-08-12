import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rdkit import RDLogger


def str_to_logger_enum(s: str) -> int:
    if s == "notset":
        return logging.NOTSET
    if s == "debug":
        return logging.DEBUG
    if s == "info":
        return logging.INFO
    if s == "warning":
        return logging.WARNING
    if s == "error":
        return logging.ERROR
    if s == "critical":
        return logging.CRITICAL
    raise RuntimeError(f"unknown log level: {s}")


def str_to_rdlogger_enum(s: str) -> Any:
    if s == "debug":
        return RDLogger.DEBUG
    if s == "info":
        return RDLogger.INFO
    if s == "warning":
        return RDLogger.WARNING
    if s == "error":
        return RDLogger.ERROR
    if s == "critical":
        return RDLogger.CRITICAL
    raise RuntimeError(f"unknown log level: {s}")


@dataclass
class LoggerConfig:
    level: str = "info"
    format: str = "[%(levelname)1.1s %(module)s:%(funcName)s] %(message)s"
    modules: Dict[str, str] = field(default_factory=dict)

    rdkit_level: Optional[str] = None


def apply_logger_config(config: LoggerConfig) -> None:
    # force logger settings
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        h.close()
    logging.basicConfig(level=str_to_logger_enum(config.level), format=config.format)
    for k, v in config.modules.items():
        logging.getLogger(k).setLevel(str_to_logger_enum(v))

    lg = RDLogger.logger()
    if config.rdkit_level is not None:
        lg.setLevel(str_to_rdlogger_enum(config.rdkit_level))
    # lg.setLevel(RDLogger.CRITICAL)
