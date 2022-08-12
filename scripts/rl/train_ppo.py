import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rdkit
from omegaconf import OmegaConf

from rjt_rl.rl.training.rjt_ppo_trainer import RJTPPOTrainer
from rjt_rl.utils.config_wrapper import validate_config_by_class
from rjt_rl.utils.logger import LoggerConfig, apply_logger_config

logger = logging.getLogger(__name__)


@dataclass
class TopConfig:
    trainer: Any
    logger: LoggerConfig = field(default_factory=LoggerConfig)


def main():
    # user_config = OmegaConf.from_cli()
    # if "yaml" in user_config:
    #     user_config = OmegaConf.load(user_config.yaml)
    # OmegaConf.resolve(user_config)

    cli_config = OmegaConf.from_cli()
    if "yaml" in cli_config:
        yaml_config = OmegaConf.load(cli_config.yaml)
        cli_config.pop("yaml")
        user_config = OmegaConf.merge(yaml_config, cli_config)
    else:
        user_config = cli_config
    OmegaConf.resolve(user_config)

    base_config = OmegaConf.structured(TopConfig)

    config = OmegaConf.merge(base_config, user_config)
    apply_logger_config(config.logger)

    logger.info(f"Using rdkit ver: {rdkit.__version__}")

    trainer_config = validate_config_by_class(RJTPPOTrainer, config["trainer"])

    outdir = Path(trainer_config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, outdir / "run_config.yaml")

    trainer = RJTPPOTrainer.from_config(trainer_config)
    trainer.train()


if __name__ == "__main__":
    main()
    logger.info("DONE")
