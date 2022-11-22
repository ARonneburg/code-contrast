from typing import Optional, Tuple

import torch
from torch import nn

from modeling.block import Block
from modeling.checkpoint_loader import load_config, load_checkpoint
from modeling.generation import generate


class CodifyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.E
        n_vocab_align64 = (config.n_vocab + 63) // 64 * 64
        self.wte = nn.Embedding(n_vocab_align64, self.embed_dim)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.L)])
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(config.E, n_vocab_align64, bias=False)

    @classmethod
    def from_pretrained(cls, path: str):
        config = load_config(path)
        model = cls(config)
        model = load_checkpoint(model, path)
        return model

    def generate(self, *args, **kwargs):
        return generate(self, *args, **kwargs)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor],
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                use_cache: Optional[bool] = False):
        hidden_states = self.wte(x)

        presents = () if use_cache else None
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = block(hidden_states=hidden_states,
                                          attention_mask=attention_mask,
                                          layer_past=layer_past,
                                          use_cache=use_cache)
            if use_cache:
                presents = presents + (present,)

        hidden_states = self.ln_f(hidden_states)
        output = self.lm_head(hidden_states)
        return output, presents
