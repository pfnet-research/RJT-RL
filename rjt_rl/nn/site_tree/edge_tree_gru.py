from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EdgeTreeGRU(nn.Module):
    def __init__(self, hidden_size: int, node_size: int, edge_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.node_size = node_size
        self.edge_size = edge_size
        eh_size = hidden_size + edge_size

        self.W_z = nn.Linear(node_size, hidden_size)
        self.U_z = nn.Linear(eh_size, hidden_size, bias=False)

        #####

        self.W_r = nn.Linear(node_size, eh_size, bias=False)
        self.U_r = nn.Linear(eh_size, eh_size)

        #####

        self.W_h = nn.Linear(node_size, hidden_size)
        self.U_h = nn.Linear(eh_size, hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, e_nei: torch.Tensor, h_nei: torch.Tensor
    ) -> torch.Tensor:
        eh_size = self.hidden_size + self.edge_size

        sum_h = h_nei.sum(dim=1)

        eh_nei = torch.cat([e_nei, h_nei], dim=2)

        sum_eh = eh_nei.sum(dim=1)

        z_input = self.W_z(x) + self.U_z(sum_eh)
        z = nn.Sigmoid()(z_input)

        ##########

        r_1 = self.W_r(x)
        r_1 = r_1.view(-1, 1, eh_size)

        r_2 = self.U_r(eh_nei)

        r = nn.Sigmoid()(r_1 + r_2)

        ##########

        gated_eh = r * eh_nei

        sum_gated_eh = gated_eh.sum(dim=1)

        h_input = self.W_h(x) + self.U_h(sum_gated_eh)
        pre_h = nn.Tanh()(h_input)

        new_h: torch.Tensor = (1.0 - z) * sum_h + z * pre_h

        return new_h
