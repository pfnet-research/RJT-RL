from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

import torch
import torch.nn.functional as F
from pfrl.agents import PPO
from pfrl.agents.ppo import _elementwise_clip

logger = logging.getLogger(__name__)


@dataclass
class RJTPPOConfig:
    expert_coef: float = -1.0

    epochs: int = 1

    value_func_coef: float = 1

    entropy_coef: float = 0.01
    entropy_clip: float = -1.0

    # PPO Update interval
    update_interval: int = 256

    # PPO minibatch size
    batch_size: int = 128


class RJTPPO(PPO):  # type: ignore
    @staticmethod
    def get_config_class() -> Type[RJTPPOConfig]:
        return RJTPPOConfig

    @classmethod
    def from_config(
        cls, model: Any, optimizer: Any, config: RJTPPOConfig, **kwargs: Any
    ) -> "RJTPPO":
        return cls(
            model,
            optimizer,
            update_interval=config.update_interval,
            minibatch_size=config.batch_size,
            epochs=config.epochs,
            expert_coef=config.expert_coef,
            value_func_coef=config.value_func_coef,
            entropy_coef=config.entropy_coef,
            entropy_clip=config.entropy_clip,
            **kwargs,
        )

    dataset: Optional[torch.utils.data.DataLoader]

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        dataset_loader: Optional[Callable[[], torch.utils.data.DataLoader]] = None,
        expert_coef: float = 0.25,
        entropy_clip: float = -1.0,
        **kwargs: Any,
    ):
        self.dataset_loader = dataset_loader

        if expert_coef <= 1e-4:
            self.exp_weight = None
        else:
            self.exp_weight = expert_coef

        gpu = kwargs["gpu"]
        self.device = torch.device("cuda:{}".format(gpu) if gpu is not None else "cpu")

        self.entropy_clip = entropy_clip
        logger.info(f"entropy_clip: {self.entropy_clip}")

        super().__init__(model, optimizer, **kwargs)
        self.count = 0
        self.dataset = None

        self.current_step = -1
        self.current_episode = -1

    def _load_dataset(self) -> None:
        assert self.dataset_loader is not None
        self.dataset = self.dataset_loader()
        self.iterator = iter(self.dataset)

    def _clamp_lossfn(
        self,
        entropy: torch.Tensor,
        vs_pred: torch.Tensor,
        log_probs: torch.Tensor,
        vs_pred_old: torch.Tensor,
        log_probs_old: torch.Tensor,
        advs: torch.Tensor,
        vs_teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        diff_lnp = log_probs - log_probs_old
        diff_lnp = torch.clamp(diff_lnp, -10.0, 10.0)
        prob_ratio = torch.exp(diff_lnp)

        loss_policy = -torch.mean(
            torch.min(
                prob_ratio * advs,
                torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs,
            ),
        )

        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred, vs_teacher)
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred, vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred, vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))

        return loss_policy, loss_value_func, loss_entropy

    def _lossfun(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.exp_weight is None:
            return self._lossfun_noexp(*args, **kwargs)
        else:
            return self._lossfun_expert(*args, **kwargs)

    def _lossfun_noexp(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.count += 1

        loss_policy, loss_value_func, loss_entropy = self._clamp_lossfn(*args, **kwargs)
        ent_term = torch.max(
            loss_entropy,
            torch.full_like(loss_entropy, fill_value=-self.entropy_clip),
        )
        ppo_loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * ent_term
        )

        total_loss: torch.Tensor = ppo_loss
        logger.info(
            f"Update {self.count} step= {self.current_step} episode= {self.current_episode} : "
            f"policy {loss_policy} "
            f"value {loss_value_func} "
            f"entropy {loss_entropy} "
            f"entropy_clip {ent_term} "
            f"ppo_loss {total_loss}"
        )
        return total_loss

    def _lossfun_expert(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.dataset is None:
            self._load_dataset()
            assert self.dataset is not None

        self.count += 1
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataset)
            batch = next(self.iterator)

        batch = batch.to(self.device)

        distrib, _ = self.model(batch)
        log_prob = distrib.log_prob(batch.actions)
        expert_loss = -log_prob.sum()

        loss_policy, loss_value_func, loss_entropy = self._clamp_lossfn(*args, **kwargs)
        ent_term = torch.max(
            loss_entropy,
            torch.full_like(loss_entropy, fill_value=-self.entropy_clip),
        )
        ppo_loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * ent_term
        )

        total_loss: torch.Tensor = ppo_loss + self.exp_weight * expert_loss
        logger.info(
            f"Update {self.count} {self.current_step} step {self.current_episode} episode:"
            f" policy {loss_policy}"
            f" value {loss_value_func}"
            f" entropy {loss_entropy}"
            f" entropy_clip {ent_term}"
            f" ppo_loss {ppo_loss}"
            f" expert_loss {expert_loss}"
            f" total {total_loss}"
        )
        return total_loss
