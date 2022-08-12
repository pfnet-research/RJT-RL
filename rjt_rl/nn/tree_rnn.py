import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TreeGRU(nn.Module):
    def __init__(self, hidden_size: int, node_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_size = node_size

        self.W_z = nn.Linear(node_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(node_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(node_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        hidden_size = self.hidden_size

        sum_h = h_nei.sum(dim=1)

        z_input = torch.cat([x, sum_h], dim=1)
        z = nn.Sigmoid()(self.W_z(z_input))

        r_1 = self.W_r(x)

        r_1 = r_1.view(-1, 1, hidden_size)
        r_2 = self.U_r(h_nei)
        r = nn.Sigmoid()(r_1 + r_2)

        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)
        h_input = torch.cat([x, sum_gated_h], dim=1)
        pre_h = nn.Tanh()(self.W_h(h_input))

        new_h: torch.Tensor = (1.0 - z) * sum_h + z * pre_h

        return new_h
