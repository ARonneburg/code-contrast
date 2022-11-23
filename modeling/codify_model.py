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
        # [1, 66, 1024]
        # print(hidden_states.shape)
        # print(presents)
        # dict(presents=presents, x_bte=hidden_states)
        return output, presents

    def highlight_forward(self, x_bte, first_bt, diffhlpoint):
        B, T, E = x_bte.shape
        assert E == self.embed_dim
        # assert T == self.hps.T, (T, self.hps.T)   # for testing, can be smaller 1024
        assert T == first_bt.shape[1], str(first_bt.shape)
        assert T == diffhlpoint.shape[1], str(diffhlpoint.shape)
        mask_BTT = torch.zeros((B, T, T), device=x_bte.device, dtype=torch.bool)
        for t in range(T):
            mask_BTT[:, t, t] = True
        for b in range(B):
            # first_bt[b]       # [0,0,0,1,0,0,0,0,0]
            # diffhlpoint[b]    # [0,0,0,0,0,0,1,0,0]
            # first nonzero in first_bt
            t1s = (first_bt[b] == 1).nonzero(as_tuple=False).squeeze(1)
            t2s = (diffhlpoint[b] == 1).nonzero(as_tuple=False).squeeze(1)
            assert len(t1s) >= len(t2s)
            for t1, t2 in zip(t1s, t2s):
                t1 = t1.item()
                t2 = t2.item()
                assert t1 < t2
                if t2 < T:
                    t2 += 1
                # fill rectangle on main diagonal
                mask_BTT[b, t1:t2, t1:t2] = 1
        inside, _state = self.bidir_sa.forward(self.bidir_sa_ln(x_bte), mask_BTT)
        return inside
