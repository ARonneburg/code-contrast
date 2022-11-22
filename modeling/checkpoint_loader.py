import os
from pathlib import Path

from cloudpickle import load
# from huggingface_hub import hf_hub_download
from modeling.config import Config

_model_hps = 'model-hps.json'


def _load_gs_file(root_path: str, filename: str):
    import blobfile as bf
    rest = root_path[len("gs://"):]
    slash = '/'
    if root_path[-1] == '/':
        slash = ''
    local = os.path.join("/tmp/small-cache-container", rest, filename)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    path = f'{root_path}{slash}{filename}'
    if os.path.exists(local):
        print("using cached %s" % local)
    else:
        print("download %s" % (path))
        bf.copy(path, local + ".tmp")
        os.rename(local + ".tmp", local)
    return str(local)


def _load_config_from_filesystem(filepath: str):
    filepath = Path(filepath)
    if not filepath.exists():
        raise RuntimeError(f"Not found: {filepath}")

    import json
    with open(str(filepath), 'r') as f:
        config = json.load(f)
    config = Config.from_dict(config)
    return config


def _load_config_from_gs(root_path: str):
    localfile = _load_gs_file(root_path, _model_hps)
    return _load_config_from_filesystem(localfile)


def load_config(path: str):
    if Path(path).exists():
        return _load_config_from_filesystem(f'{path}/{_model_hps}')
    elif path.startswith('gs://'):
        return _load_config_from_gs(path)
    elif 'url':
        ...


def _load_f(root_path: str, filename: str):
    l_path = Path(root_path, filename)
    if not l_path.exists():
        if root_path.startswith('gs:/'):
            l_path = _load_gs_file(root_path, filename)
            l_path = Path(l_path)
    if not l_path.exists():
        raise RuntimeError(f"Not found: {l_path}")
    print(f'loading {l_path}')
    with open(str(l_path), 'rb') as f:
        return load(f)


def _load_checkpoint(model, root_path: str):
    model.wte.weight.data[:] = _load_f(root_path, 'emb')
    model.lm_head.weight.data[:] = _load_f(root_path, 'unemb')
    model.ln_f.weight.data[:] = _load_f(root_path, 'bounce.ln_final.weight')
    model.ln_f.bias.data[:] = _load_f(root_path, 'bounce.ln_final.bias')

    for i in range(1, len(model.layers) + 1):
        f_prefix = f'layers.{i:03d}'
        model.layers[i - 1].ln_a.weight.data[:] = _load_f(root_path, f'{f_prefix}.ln_a.weight')
        model.layers[i - 1].ln_a.bias.data[:] = _load_f(root_path, f'{f_prefix}.ln_a.bias')
        model.layers[i - 1].ln_m.weight.data[:] = _load_f(root_path, f'{f_prefix}.ln_m.weight')
        model.layers[i - 1].ln_m.bias.data[:] = _load_f(root_path, f'{f_prefix}.ln_m.bias')

        model.layers[i - 1].mlp.ln_1.weight.data[:] = _load_f(root_path, f'{f_prefix}.pw.W1')
        model.layers[i - 1].mlp.ln_1.bias.data[:] = _load_f(root_path, f'{f_prefix}.pw.b1')
        model.layers[i - 1].mlp.ln_2.weight.data[:] = _load_f(root_path, f'{f_prefix}.pw.W2')
        model.layers[i - 1].mlp.ln_2.bias.data[:] = _load_f(root_path, f'{f_prefix}.pw.b2')

        model.layers[i - 1].sa.qkv.weight.data[:] = _load_f(root_path, f'{f_prefix}.sa.qkv')
        model.layers[i - 1].sa.qkv.bias.data[:] = _load_f(root_path, f'{f_prefix}.sa.qkv_bias')
        model.layers[i - 1].sa.out.weight.data[:] = _load_f(root_path, f'{f_prefix}.sa.backproj')
        model.layers[i - 1].sa.out.bias.data[:] = _load_f(root_path, f'{f_prefix}.sa.backproj_bias')

    return model


def load_checkpoint(model, path: str):
    if Path(path).exists() or path.startswith('gs://'):
        model = _load_checkpoint(model, path)
    for param in model.parameters():
        param.requires_grad = False
    return model
