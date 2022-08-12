from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as E
import torch
import torch.optim as optim
from pytorch_pfn_extras.training.triggers.minmax_value_trigger import MinValueTrigger

from rjt_rl.extensions.invoke_gc import InvokeGC
from rjt_rl.rjt.vocab import Vocab, load_vocab
from rjt_rl.rl.datasets.expert_dataset2 import ExpertDataset2
from rjt_rl.rl.datasets.expert_dataset_collator import PretrainDatasetCollator
from rjt_rl.rl.envs.mol_env_base import MolEnvBase
from rjt_rl.rl.models.policy_pretrain_wrapper import PolicyPretrainWrapper
from rjt_rl.rl.training.utils import setup_random_seed

logger = logging.getLogger(__name__)
manager: ppe.training.ExtensionsManager


def parse_cmdline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "-g", "--gpu", type=int, default=-1, help="GPU ID (negative value indicates CPU"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=128, help="minibatch size"
    )
    parser.add_argument("--epoch", type=int, default=10, required=False)

    parser.add_argument("-v", "--vocab", type=str, default=None, required=True)

    # training data
    parser.add_argument("-i", "--input_pkl", type=str, default=None, required=False)
    parser.add_argument(
        "--input_pkl_maxidx", type=float, default=float("inf"), required=False
    )
    parser.add_argument("--input_pkl_minidx", type=int, default=0, required=False)
    parser.add_argument("--input_limit", type=int, default=None, required=False)

    parser.add_argument("-o", "--out", default="results", help="Output directory")
    parser.add_argument("--snap_freq", type=int, default=1, help="snapshot freq")
    parser.add_argument(
        "--snap_name",
        type=str,
        default="snapshot_iter_{.iteration}.pt",
        help="snapshot name",
    )
    parser.add_argument("--save_best_snapshot", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--log_interval_unit", type=str, default="epoch")

    parser.add_argument("--hidden_size", type=int, default=512, required=False)

    parser.add_argument("--use_depth_first", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_total", type=int, default=0)

    parser.add_argument("--gc_interval", type=int, default=None)
    parser.add_argument("--worker_gc_interval", type=int, default=400)

    args = parser.parse_args()

    return args


def expert_pretrain(
    args: argparse.Namespace,
    create_env_fn: Callable[[], MolEnvBase],
    create_model_fn: Callable[[Vocab, MolEnvBase, Any], PolicyPretrainWrapper],
    extension_list: Optional[list[Any]] = None,
) -> None:
    setup_random_seed(args.seed)

    vocab = load_vocab(args.vocab)

    train_ds = ExpertDataset2(
        input_pkl=args.input_pkl,
        n_total=args.n_total,
        depth_first=args.use_depth_first,
        gc_interval=args.worker_gc_interval,
    )
    shuffle = False

    collator = PretrainDatasetCollator()

    train_iter = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    env = create_env_fn()
    env.init(vocab)
    env.seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu) if args.gpu >= 0 else "cpu")
    model = create_model_fn(vocab, env, args)

    model.to(device)

    opt = optim.Adam(model.parameters())

    models: dict[str, torch.nn.Module] = {"main": model}
    optimizers: dict[str, torch.optim.Optimizer] = {"main": opt}
    global manager
    iters_per_epoch = len(train_iter)
    if extension_list is None:
        extension_list = []
    manager = ppe.training.ExtensionsManager(
        models,
        optimizers,
        args.epoch,
        extensions=extension_list,
        iters_per_epoch=iters_per_epoch,
        out_dir=args.out,
    )

    log_interval = (args.log_interval, args.log_interval_unit)

    if args.snap_freq is not None and args.snap_freq > 0:
        logger.info(f"snap freq: {args.snap_freq}")
        writer = E.snapshot_writers.SimpleWriter()
        snp = E.snapshot(writer=writer, filename=args.snap_name)
        manager.extend(snp, trigger=(args.snap_freq, "iteration"))

    if args.save_best_snapshot:
        best_key = "main/loss"
        writer = E.snapshot_writers.SimpleWriter()
        snp = E.snapshot(writer=writer, filename="best_model.pt")
        manager.extend(snp, trigger=MinValueTrigger(best_key))
        manager.extend(
            lambda _: logger.info("*** min value updated"),
            trigger=MinValueTrigger(best_key),
        )

    logrep = E.LogReport(trigger=log_interval)
    manager.extend(logrep)

    manager.extend(
        E.PrintReport(
            [
                "epoch",
                "iteration",
                "main/loss",
            ]
        ),
        trigger=log_interval,
    )

    manager.extend(E.ProgressBar(update_interval=5))

    if args.gc_interval is not None:
        manager.extend(
            InvokeGC(mem_check=False), trigger=(args.gc_interval, "iteration")
        )

    if args.resume is not None:
        if Path(args.resume).exists():
            logger.info(f"resume from {args.resume}")
            state = torch.load(args.resume)  # type: ignore
            manager.load_state_dict(state)
        else:
            logger.info(f"Snapshot file {args.resume} not found (ignored).")

    while not manager.stop_trigger:
        epoch = manager.epoch
        train(args, model, device, train_iter, opt, epoch)


def train(
    args: argparse.Namespace,
    model: PolicyPretrainWrapper,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    model.train()
    for _batch_idx, batch in enumerate(train_loader):

        with manager.run_iteration():
            if batch is None:
                logger.error("batch is None")
                continue
            try:
                batch.to(device)
                optimizer.zero_grad()
                output = model(batch, batch.actions)
                loss = output
                loss.backward()
                optimizer.step()
            except Exception as e:
                logger.error(f"Failed: {e}")
                mol_batch = batch.get_mol_batch()
                for i, mt in enumerate(mol_batch):
                    logger.error(f"{i}: len(nodes): {mt.dump_str()}")
                    mt.dump_tree()
                raise
