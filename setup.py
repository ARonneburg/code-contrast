from setuptools import setup

setup(
    name="code-contrast",
    py_modules=["api_server", "code_contrast"],
    version="0.0.1",
    install_requires=["numpy", "tokenizers", "fastapi", "uvicorn", "termcolor",
                      "cdifflib", "cloudpickle", "dataclasses_json", "torch"]
)
