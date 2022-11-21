from typing import Optional, Tuple

import torch
from torch import nn

from modeling.block import Block
from modeling.checkpoint_loader import load_config, load_checkpoint


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

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor],
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                use_cache: bool = False):
        hidden_state = self.wte(x)

        presents = () if use_cache else None
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        for (block, layer_past) in zip(self.layers, past_key_values):
            hidden_state, present = block(hidden_state=hidden_state,
                                          attention_mask=attention_mask,
                                          layer_past=layer_past,
                                          use_cache=use_cache)
            if use_cache:
                presents = presents + (hidden_state,)

        hidden_states = self.ln_f(hidden_state)
        output = self.lm_head(hidden_states)
        return output, presents
