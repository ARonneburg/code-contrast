from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from code_contrast import SMCEncoding


class HFModel(nn.Module):
    def __init__(
            self,
            checkpoint: str,
            device: str,
            cache_dir: Optional[str] = None,
            use_auth_token: Optional[str] = None
    ):
        super().__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            cache_dir=cache_dir,
            trust_remote_code=True,
            use_auth_token=use_auth_token,
        ).to(device)
        self.encoding = SMCEncoding(checkpoint.replace('/', '_'))

    @classmethod
    def from_pretrained(self,
                        path: str,
                        device: str = "cuda",
                        cache_dir: Optional[str] = None,
                        **unused):
        return HFModel(path, device, cache_dir=cache_dir)

    def forward(self, x, past_key_values: Optional = None, **unused):
        output = self.model(x, past_key_values=past_key_values)
        return output.logits, output.past_key_values

    def lm_forward(self, x, **unused):
        return x  # inference is done in the `forward` method

    def to_device(self, module: nn.Module):
        module = module.to(self.device)
        if self.device.startswith("cuda"):
            module = module.to(torch.half)
        return module

    def generate(self, inputs):
        return self.model.generate(inputs)
