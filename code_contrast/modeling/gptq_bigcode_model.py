from typing import Optional

import os
import torch
from torch import nn

from transformers import AutoConfig
from transformers import modeling_utils
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM

from code_contrast.encoding import SMCEncoding
from code_contrast.modeling.quant import QuantLinear

from typing import Tuple, Any


def disable_torch_init():
    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    modeling_utils._init_weights = False


class GPTQBigCodeModel(nn.Module):

    def __init__(self, model: str, checkpoint: str,
                 bits: int, groupsize: int, device: str,
                 use_auth_token: Optional[str] = None):
        super().__init__()

        self.encoding = SMCEncoding(model.replace('/', '_').replace('-', ''))
        self.device = device
        disable_torch_init()

        config = AutoConfig.from_pretrained(model, use_auth_token=use_auth_token)
        model = GPTBigCodeForCausalLM(config)
        model.eval()

        self._quantize(model, bits, groupsize, self.device)
        # TODO: we should split checkpoint into smaller parts to avoid large RAM usage
        model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        # for layer_name in os.listdir(checkpoint):
        #     model.load_state_dict({
        #         layer_name: torch.load(os.path.join(checkpoint, layer_name),
        #                                map_location=self.device)
        #     }, strict=False)
        self._model = model.to(self.device)

    @staticmethod
    def _quantize(module: nn.Module, bits: int, groupsize: int, device: str,
                  layer_types: Tuple[Any] = (nn.Conv2d, nn.Linear), prefix: str = ""):
        if isinstance(module, QuantLinear):
            return
        for name in dir(module):
            layer = getattr(module, name)
            layer_name = prefix + "." + name if prefix != "" else name
            if isinstance(layer, layer_types) and layer_name not in ["lm_head"]:
                delattr(module, name)
                quant_layer = QuantLinear(
                    bits, groupsize,
                    layer.in_features, layer.out_features,
                    layer.bias is not None)
                setattr(module, name, quant_layer.to(device))
        for name, child in module.named_children():
            GPTQBigCodeModel._quantize(child, bits, groupsize, device, layer_types,
                     prefix + "." + name if prefix != "" else name)

    def forward(self, x, past_key_values: Optional = None, **unused):
        if past_key_values:
            past_key_values = [t[0] for t in past_key_values]
        output = self._model(x, past_key_values=past_key_values)
        return output.logits, [(t, ) for t in output.past_key_values]

    def lm_forward(self, x, **unused):
        return x  # inference is done in the `forward` method
