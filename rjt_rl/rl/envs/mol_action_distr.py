from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional

import torch
from torch.distributions import Categorical, Distribution
from torch.nn import functional as F

from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.utils import filter_logits, index_tensor
from rjt_rl.rjt.vocab import Vocab
from rjt_rl.utils import (
    sample_categorical,
    sample_categorical_logits,
    sample_multinomial,
)

from .actions import Action2

logger = logging.getLogger(__name__)

SAMPLING_USE_CPU = False


def sample_distrib_multi(logits: torch.Tensor) -> torch.Tensor:

    if SAMPLING_USE_CPU:
        device = logits.device
        logits = logits.cpu()

    probs = torch.nn.Softmax()(logits)
    try:
        result = sample_multinomial(logits, len(logits))
    except RuntimeError:
        logger.error(f"logits={logits}, probs={probs}")
        raise
    logger.debug(result)

    if SAMPLING_USE_CPU:
        result = result.to(device)

    return result


class MolActionDistr3(Distribution):
    def __init__(
        self,
        mol_batch: Sequence[MolTree],
        vocab: Vocab,
        stop_distrib: Categorical,
        targ_distrib: Categorical,
        word_distrib: Categorical,
        sitedir_distrib: Categorical,
        site1_scores: torch.Tensor,
        site2_scores: torch.Tensor,
        stop_ent_coef: float = 1e-20,
        word_ent_coef: float = 1.0,
        targ_ent_coef: float = 1.0,
        site1_ent_coef: float = 0.0,
        site2_ent_coef: float = 0.0,
        sitedir_ent_coef: float = 0.0,
        freeze_actions: Optional[list[int]] = None,
    ):
        self.mol_batch = mol_batch
        self.vocab = vocab
        self.stop_distrib = stop_distrib
        self.targ_distrib = targ_distrib
        self.word_distrib = word_distrib
        self.sitedir_distrib = sitedir_distrib
        self.site1_scores = site1_scores
        self.site2_scores = site2_scores

        self.node_sizes = None

        self.select_valid_words = False

        self.stop_ent_coef = stop_ent_coef
        self.word_ent_coef = word_ent_coef
        self.targ_ent_coef = targ_ent_coef

        self.site1_ent_coef = site1_ent_coef
        self.site2_ent_coef = site2_ent_coef
        self.sitedir_ent_coef = sitedir_ent_coef

        self.use_cpu = SAMPLING_USE_CPU

        self.freeze_actions = freeze_actions

    def entropy(self) -> torch.Tensor:
        batch_size, targ_size = self.targ_distrib.param_shape
        del batch_size, targ_size

        stop_ent: torch.Tensor = self.stop_distrib.entropy()  # type: ignore
        targ_ent: torch.Tensor = self.targ_distrib.entropy()  # type: ignore

        word_ent_unc: torch.Tensor = self.word_distrib.entropy()  # type: ignore
        word_ent = (word_ent_unc * self.targ_distrib.probs).sum(1)

        sd_ent_unc = self.sitedir_distrib.entropy()  # type: ignore
        sitedir_ent = (sd_ent_unc * self.targ_distrib.probs).sum(1)

        site1_distrib = Categorical(logits=self.site1_scores)  # type: ignore
        s1d_ent_unc = site1_distrib.entropy()  # type: ignore
        site1_ent = (s1d_ent_unc * self.targ_distrib.probs).sum(1)

        site2_distrib = Categorical(logits=self.site2_scores)  # type: ignore
        s2d_ent_unc = site2_distrib.entropy()  # type: ignore
        site2_ent = (s2d_ent_unc * self.targ_distrib.probs).sum(1)

        entropy: torch.Tensor = (
            stop_ent * self.stop_ent_coef
            + word_ent * self.word_ent_coef
            + targ_ent * self.targ_ent_coef
            + sitedir_ent * self.sitedir_ent_coef
            + site1_ent * self.site1_ent_coef
            + site2_ent * self.site2_ent_coef
        )

        return entropy

    def _sample_stop_distrib(self) -> torch.Tensor:
        ac_stop = sample_categorical(self.stop_distrib, use_cpu=self.use_cpu)
        logger.debug(f"ac_stop.shape: {ac_stop.shape}")
        logger.debug(f"ac_stop: {ac_stop}")
        return ac_stop

    def _sample_all_word_distrib(self) -> torch.Tensor:
        return sample_categorical(self.word_distrib, use_cpu=self.use_cpu)

    def _sample_all_sitedir_distrib(self) -> torch.Tensor:
        return sample_categorical(self.sitedir_distrib, use_cpu=self.use_cpu)

    def sample(self):  # type: ignore
        ac_stop = self._sample_stop_distrib()
        ac_targ = sample_categorical(self.targ_distrib, use_cpu=self.use_cpu)
        targ_idx = ac_targ[:, None]

        if self.select_valid_words:
            ac_word_list = []
            for i, act in enumerate(ac_targ):
                node = self.mol_batch[i].nodes[act]
                word_scores = self.word_distrib.logits[i, act, :]
                if node.is_singleton():
                    word_scores[self.vocab.not_bond_mask] = -1.0e10

                elif node.is_ring():
                    word_scores[self.vocab.singleton_mask] = -1.0e10
                ac = sample_categorical_logits(word_scores, use_cpu=self.use_cpu)
                logger.debug(f"{i}: sampled word action: {ac}")
                ac_word_list.append(ac)
            ac_word = index_tensor(ac_word_list, device=ac_targ.device)
        else:
            ac_word_all = self._sample_all_word_distrib()
            ac_word = torch.gather(ac_word_all, dim=1, index=targ_idx)
            ac_word = ac_word[:, 0]

        #####

        ac_sitedir_all = self._sample_all_sitedir_distrib()
        ac_sitedir = torch.gather(ac_sitedir_all, dim=1, index=targ_idx)
        ac_sitedir = ac_sitedir[:, 0]

        #####

        site1_scores = self.calc_site1_scores(ac_targ)
        ac_site1 = sample_categorical_logits(site1_scores, use_cpu=self.use_cpu)

        #####

        site2_scores = self.calc_site2_scores(ac_targ, ac_word)
        ac_site2 = sample_categorical_logits(site2_scores, use_cpu=self.use_cpu)
        actions = torch.stack(
            [ac_targ, ac_word, ac_sitedir, ac_site1, ac_site2, ac_stop],
            dim=1,
        )
        return actions

    def calc_site1_scores(self, ac_targ: torch.Tensor) -> torch.Tensor:
        site1_scores = select_index_dim1(self.site1_scores, ac_targ)
        device = site1_scores.device
        targ_node_sizes = index_tensor(
            [mt.nodes[targ].size() for mt, targ in zip(self.mol_batch, ac_targ)],
            device=device,
        )
        site1_scores = filter_logits(site1_scores, targ_node_sizes)
        return site1_scores

    def calc_site2_scores(
        self, ac_targ: torch.Tensor, ac_word: torch.Tensor
    ) -> torch.Tensor:
        site2_scores = select_index_dim1(self.site2_scores, ac_targ)
        device = site2_scores.device
        new_node_sizes = index_tensor(
            [self.vocab.get_word_size(int(i)) for i in ac_word], device=device
        )
        site2_scores = filter_logits(site2_scores, new_node_sizes)
        return site2_scores

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        masked_actions = actions.clone().detach()
        masked_actions[actions == -1] = 0

        ac_targ = masked_actions[:, Action2.TARGET_PRED_ID]
        ac_word = masked_actions[:, Action2.WORD_PRED_ID]

        logprobs_word = select_index_dim1(self.word_distrib.logits, ac_targ)

        logprobs_sitedir = select_index_dim1(self.sitedir_distrib.logits, ac_targ)

        site1_scores = self.calc_site1_scores(ac_targ)
        logprobs_site1 = F.log_softmax(site1_scores, dim=1)

        site2_scores = self.calc_site2_scores(ac_targ, ac_word)
        logprobs_site2 = F.log_softmax(site2_scores, dim=1)

        log_probs = [
            self.targ_distrib.logits,
            logprobs_word,
            logprobs_sitedir,
            logprobs_site1,
            logprobs_site2,
            self.stop_distrib.logits,
        ]

        logp = select_log_prob(log_probs, actions, self.freeze_actions)
        return logp


def select_index_dim1(in_tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if not index.is_cuda:
        index = index.to(in_tensor.device)
    targ_idx = index[:, None, None]
    targ_idx = targ_idx.repeat_interleave(in_tensor.shape[2], dim=2)
    out_tensor = torch.gather(in_tensor, dim=1, index=targ_idx)
    out_tensor = out_tensor[:, 0, :]
    return out_tensor


def select_log_prob(
    log_probs: list[torch.Tensor],
    actions: torch.Tensor,
    freeze_actions: Optional[list[int]] = None,
) -> torch.Tensor:
    device = log_probs[0].device
    nbatch = log_probs[0].shape[0]
    num_actions = len(log_probs)
    zero_lnp = torch.zeros((nbatch, num_actions), device=device)
    targ_list = []
    mask = actions == -1
    zf_actions = actions.clone().detach()
    zf_actions[mask] = 0

    if freeze_actions is None:
        frz_ac = []
    else:
        frz_ac = freeze_actions

    for iac, log_p in enumerate(log_probs):
        zf_ac = zf_actions[:, iac : iac + 1]

        target_log_p = torch.gather(log_p, dim=1, index=zf_ac)
        if iac in frz_ac:
            target_log_p = target_log_p.detach()
        targ_list.append(target_log_p)

    target = torch.cat(targ_list, dim=1)
    target = torch.where(mask, zero_lnp, target)
    target = target.sum(dim=1)
    return target
