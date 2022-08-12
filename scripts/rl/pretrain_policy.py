import logging

import rdkit
from rdkit import RDLogger

from rjt_rl.rl.envs.rjt_mol_env import RJTMolEnv
from rjt_rl.rl.models.policy_pretrain_wrapper import PolicyPretrainWrapper
from rjt_rl.rl.training.expert_pretrain_utils import expert_pretrain, parse_cmdline_args

logger = logging.getLogger(__name__)


def create_env():
    return RJTMolEnv()


def create_model(vocab, env, args):
    return PolicyPretrainWrapper(vocab, env, hidden_size=args.hidden_size)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    logger.info(f"Using rdkit ver: {rdkit.__version__}")

    args = parse_cmdline_args()
    extns = []
    expert_pretrain(
        args,
        create_env_fn=create_env,
        create_model_fn=create_model,
        extension_list=extns,
    )


if __name__ == "__main__":
    main()
