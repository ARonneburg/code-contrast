from setuptools import setup
from setuptools import find_packages


setup(
    name="code-contrast",
    py_modules=["api_server", "code_contrast"],
    packages=find_packages(),
    package_data={"code_contrast": ["encodings/*.json"]},
    version="0.0.1",
    install_requires=["numpy", "tokenizers", "fastapi", "uvicorn", "termcolor",
                      "cdifflib", "cloudpickle", "dataclasses_json", "torch"],
)
