import os

from setuptools import setup
from setuptools import find_packages


class CudaIsNotAvailableError(Exception):
    pass


additional_setup_kwargs = dict()
if os.environ.get("BUILD_QUANT_CUDA", "0") == "1":
    try:
        import torch
        from torch.utils import cpp_extension
        if not torch.cuda.is_available():
            raise CudaIsNotAvailableError
        additional_setup_kwargs = {
            "ext_modules": [
                cpp_extension.CUDAExtension("quant_cuda", [
                    "quant_cuda/quant_cuda.cpp",
                    "quant_cuda/quant_cuda_kernel.cu"
                ])
            ],
            "cmdclass": {"build_ext": cpp_extension.BuildExtension},
        }
    except ImportError:
        print("To build quant_cuda extension install torch")
    except CudaIsNotAvailableError:
        print("To build quant_cuda extension install cuda")


setup(
    name="code-contrast",
    py_modules=["code_contrast"],
    packages=find_packages(),
    package_data={"code_contrast": ["encoding/*.json", "model_caps/htmls/*.html"]},
    version="0.0.3",
    install_requires=["numpy", "tokenizers", "fastapi", "hypercorn", "termcolor",
                      "huggingface_hub", "tiktoken", "cdifflib", "cloudpickle",
                      "sentencepiece", "dataclasses_json", "torch", "transformers"],
    **additional_setup_kwargs,
)
