from __future__ import annotations

import argparse
import gc
import logging
import pickle
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Type

import pandas as pd
import torch

from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.vocab import Vocab, load_vocab
from rjt_rl.utils.make_dataset3 import process_smiles
from rjt_rl.utils.path_like import AnyPathLikeType

from .expert_dataset_builder2 import FragDataType, make_step_frags3

logger = logging.getLogger(__name__)


@dataclass
class ExpertDataset2Config:
    input_pkl: Optional[str] = None

    n_total: int = -1
    shuffle: bool = True
    depth_first: bool = False
    gc_interval: int = 400


class ExpertDataset2(torch.utils.data.IterableDataset):
    @staticmethod
    def get_config_class() -> Type[ExpertDataset2Config]:
        return ExpertDataset2Config

    @classmethod
    def from_config(cls, args: ExpertDataset2Config) -> "ExpertDataset2":
        return cls(
            input_pkl=args.input_pkl,
            n_total=args.n_total,
            shuffle=args.shuffle,
            depth_first=args.depth_first,
            gc_interval=args.gc_interval,
        )

    @classmethod
    def from_csv_file(cls, vocab: Vocab, csv_file: AnyPathLikeType) -> "ExpertDataset2":
        df = pd.read_csv(csv_file)
        smiles_list = df.SMILES.values
        mts = []
        for s in smiles_list:
            mt, _ = process_smiles(s)
            if mt is None:
                continue
            mt.set_vocab_wid(vocab)
            mt.set_prop_order(up_only=False)
            mts.append(mt)
        ds = ExpertDataset2(input_pkl=None, n_total=0)
        ds.dataset = mts
        ds.n_total = len(mts)
        return ds

    dataset: Optional[list[MolTree]]

    def __init__(
        self,
        input_pkl: Optional[str],
        n_total: int,
        shuffle: bool = True,
        depth_first: bool = False,
        gc_interval: int = 400,
    ):
        self.shuffle = shuffle
        self.depth_first = depth_first
        self.input_pkl = input_pkl
        self.n_total = n_total
        self.dataset = None
        self.gc_interval = gc_interval

    def __len__(self) -> int:
        return self.n_total

    def load_dataset_pkl(self, ind: int) -> Any:
        pkl_path = Path(f"{self.input_pkl}_{ind}.pkl")
        if not pkl_path.exists():
            logger.warning(f"input pkl file not found: {pkl_path}")
            return None
        logger.info(f"loading tmp pkl: {pkl_path}")
        with pkl_path.open("rb") as f:
            dat = pickle.load(f)
        return dat

    def __iter__(self) -> Iterator[Any]:
        info = torch.utils.data.get_worker_info()  # type: ignore
        if info is None:
            if self.dataset is None:
                self.dataset = []
                for i in range(9999):
                    dat = self.load_dataset_pkl(i)
                    if dat is None:
                        break
                    self.dataset.extend(dat)
            part_dataset = self.dataset
        else:
            worker_id = info.id
            part_dataset = self.load_dataset_pkl(worker_id)
            if part_dataset is None:
                raise RuntimeError(f"file not found for worker: {worker_id}")
            logger.info(f"worker {worker_id} loaded {len(part_dataset)} dataset")

        part_dataset = random.sample(part_dataset, len(part_dataset))

        for i, mt in enumerate(part_dataset):
            if info is not None and i % self.gc_interval == 0:
                gc.collect()
                # logger.info(f"{i} id={info.id} GC done")
            ds = self.process_mol_tree(mt)
            yield ds

    def process_mol_tree(self, moltree: MolTree) -> FragDataType:
        ds = make_step_frags3(
            moltree, shuffle=self.shuffle, depth_first=self.depth_first
        )
        dsc = random.choice(ds)
        return dsc

    def build_batch_dataset(self) -> None:
        raise NotImplementedError()


######
# create expert dataset for pretraining workers


def parse_cmdline_args(argv: Optional[Any] = None) -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab", type=str, default=None, required=True)
    parser.add_argument("-i", "--input_pkl", type=str, default=None, required=False)
    parser.add_argument(
        "--input_pkl_maxidx", type=int, default=sys.maxsize, required=False
    )
    parser.add_argument("--input_pkl_minidx", type=int, default=0, required=False)
    parser.add_argument("--input_limit", type=int, default=None, required=False)

    parser.add_argument(
        "--out_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument("--prefix", type=str, default="tmp", help="Output file prefix")

    parser.add_argument("--num_workers", type=int, default=None, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=128, required=False)

    args = parser.parse_args(argv)

    return args


def create_worker_dataset(args: Any) -> None:
    n_workers = args.num_workers
    input_pkl = args.input_pkl
    vocab = load_vocab(args.vocab)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_idx = args.input_pkl_minidx
    max_idx = args.input_pkl_maxidx
    nlimit = args.input_limit
    prefix = args.prefix

    pkl_idx = min_idx
    dataset = []
    while pkl_idx < max_idx:
        stem = input_pkl
        fname = "{}_{:03d}.pkl".format(stem, int(pkl_idx))
        if not Path(fname).is_file():
            break
        logger.info("Loading pkl: %s", fname)
        with open(fname, "rb") as f:
            dat = pickle.load(f)
            dataset.extend(dat)
        pkl_idx += 1

    # Shuffle/truncate dataset
    dataset = random.sample(dataset, len(dataset))
    if nlimit is not None:
        dataset = dataset[:nlimit]

    # Check dataset size
    nlen = len(dataset)
    batchsize = args.batch_size
    batch_worker = batchsize * n_workers
    if nlen % batch_worker != 0:
        logger.info(
            f"dataset size ({nlen}) is not divisible by "
            f"batchsize * num_workers (={batch_worker})"
        )
        new_len = (nlen // batch_worker) * batch_worker
        dataset = dataset[:new_len]
        nlen = len(dataset)
        assert nlen % batch_worker == 0
        logger.info(f"--> truncated to {nlen}")

    if nlen == 0:
        raise RuntimeError("dataset size is zero")

    for mt in dataset:
        try:
            mt.set_vocab_wid(vocab)
            mt.set_prop_order(up_only=False)
        except Exception:
            traceback.print_exc()
            pass

    n_total = len(dataset)
    logger.info(f"Loaded {n_total} dataset")
    per_worker = n_total // n_workers
    for worker_id in range(n_workers):
        st = worker_id * per_worker
        en = min(st + per_worker, n_total)
        pkl_path = out_dir / f"{prefix}_{worker_id}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(dataset[st:en], f)
        logger.info(f"wrote tmp pkl: {pkl_path}")
