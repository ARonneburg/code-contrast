from typing import Optional, Tuple

import torch
from torch import nn

from .attention import MultiheadSelfAttention
from .mlp import MLP


class Block(nn.Module):
    def __init__(self, config,
                 residual_scale: float = 1.0):
        super().__init__()
        self.ln_a = nn.LayerNorm(config.E)
        self.ln_m = nn.LayerNorm(config.E)

        self.sa = MultiheadSelfAttention(config)
        self.mlp = MLP(config)
        # self.residual_scale = residual_scale
        # self.use_residual_scale = config.use_res_scale

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        residual = hidden_states
        hidden_states = self.ln_a(hidden_states)
        attn_output, present = self.sa(hidden_states,
                                       attention_mask=attention_mask,
                                       layer_past=layer_past,
                                       use_cache=use_cache)
        hidden_states = attn_output + residual
        hidden_states = self.ln_m(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # if self.use_residual_scale:
        #     hidden_states = residual + self.residual_scale * (attn_output + feed_forward_hidden_states)
        # else:
        hidden_states = residual + attn_output + feed_forward_hidden_states

        return hidden_states, present
