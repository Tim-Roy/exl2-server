
from setuptools import setup

setup(
    name="exl2",
    packages=["exl2"],
    install_requires=[
        "exllamav2~=0.2",
        "fastapi~=0.115",
        "flash-attn~=2.6",
        "packaging~=24.1",
        "pydantic~=2.9",
        "pyyaml~=6.0",
        "requests~=2.32",
        "safetensors~=0.4",
        "tokenizers~=0.20",
        "torch~=2.5",
        "uvicorn~=0.32",
        "wheel~=0.44",
    ],
)
