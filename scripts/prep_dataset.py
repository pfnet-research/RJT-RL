import argparse
import logging
import random
from distutils.util import strtobool

import numpy as np
import pandas as pd
import rdkit

from rjt_rl.utils import make_dataset3
from rjt_rl.utils.make_dataset import write_vocab

logger = logging.getLogger(__name__)


def load_smiles(args):
    sms = []
    if args.input_smi is not None:
        for in_smi in args.input_smi:
            logger.info(f"loading file: {in_smi}")
            with open(in_smi, "r") as f:
                lines = f.read().splitlines()
            sms.extend(lines)

    if args.input_csv is not None:
        for in_csv in args.input_csv:
            logger.info(f"loading file: {in_csv}")
            # df = pd.read_csv(in_csv, index_col=0)
            df = pd.read_csv(in_csv)
            sms.extend(df[args.column].values.tolist())
        # logger.info(df)
        # sms = df[args.column].values
    return sms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv", nargs="+", type=str, default=None)
    parser.add_argument("--input_smi", nargs="+", type=str, default=None)
    parser.add_argument("-c", "--column", type=str, default="SMILES")
    parser.add_argument("-v", "--output_vocab_csv", type=str, default=None)
    parser.add_argument("-o", "--output_pkl_stem", type=str, default=None)
    parser.add_argument("--compute_mst", action="store_true", default=None)
    parser.add_argument("-n", "--limit_num", type=int, default=None)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--no_brg_ring", type=strtobool, default=False)
    parser.add_argument("--max_ring_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="random num seed")
    # parser.add_argument(
    #     "--reroot_tree", type=strtobool, default=False, help="reroot tree"
    # )

    parser.add_argument("--ncpu", type=int, default=1)

    args = parser.parse_args()

    ##########

    # lg = rdkit.RDLogger.logger()
    # lg.setLevel(rdkit.RDLogger.CRITICAL)

    # logging.basicConfig(level=logging.DEBUG,
    #                     format="%(levelname)s %(name)s(%(funcName)s %(lineno)d): %(message)s")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    logger.info(f"Using rdkit ver: {rdkit.__version__}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    sms = load_smiles(args)

    if args.limit_num is not None:
        random.shuffle(sms)
        sms = sms[: args.limit_num]

    cset, vval_dic = make_dataset3.process_smiles_list(
        smiles_list=sms,
        output_pkl_stem=args.output_pkl_stem,
        nsplit=args.split,
        no_brg_ring=args.no_brg_ring,
        # args.reroot_tree,
        num_ring_atoms=args.max_ring_size,
        n_jobs=args.ncpu,
    )

    if args.output_vocab_csv:
        write_vocab(args.output_vocab_csv, cset, vval_dic)


if __name__ == "__main__":
    main()
    logger.info("DONE")
