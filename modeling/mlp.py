import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.ln_1 = nn.Linear(config.E, config.E * config.mlp_mult)
        self.ln_2 = nn.Linear(config.E * config.mlp_mult, config.E)

    def forward(self, x: torch.Tensor):
        x = self.ln_1(x)
        x = F.gelu(x)
        x = self.ln_2(x)
        return x
