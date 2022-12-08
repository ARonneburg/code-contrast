from setuptools import setup
from setuptools import find_packages


setup(
    name="code-contrast",
    py_modules=["self_hosting", "code_contrast"],
    packages=find_packages(),
    package_data={"code_contrast": ["encoding/*.json"]},
    version="0.0.1",
    install_requires=["numpy", "tokenizers", "fastapi", "uvicorn", "termcolor",
                      "cdifflib", "cloudpickle", "dataclasses_json", "torch"],
)
