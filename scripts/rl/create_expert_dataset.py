import logging

from rdkit import RDLogger

from rjt_rl.rl.datasets.expert_dataset2 import create_worker_dataset, parse_cmdline_args

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lg = RDLogger.logger()
    # lg.setLevel(RDLogger.ERROR)
    lg.setLevel(RDLogger.CRITICAL)

    args = parse_cmdline_args()
    create_worker_dataset(args)


if __name__ == "__main__":
    main()
