import json
from os import PathLike, getenv
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from .exceptions import MissingEnvironmentVariable


class LLMConfig(BaseModel):
    llm_path: PathLike | str
    config_fname: str = Field(default="config.json")
    max_position_embeddings_kw: str = Field(default="max_position_embeddings")
    max_position_embeddings: int = Field(init=False)
    weight_format: str
    prompt_format: str

    @model_validator(mode="before")
    def get_max_position_embeddings(cls, values):
        if "max_position_embeddings" in values:
            return values
        llm_path = values.get("llm_path")
        config_fname = values.get("config_fname", "config.json")
        max_position_embeddings_kw = values.get("max_position_embeddings_kw", "max_position_embeddings")
        path = Path(llm_path, config_fname)
        if not path.is_file() or path.suffix != ".json":
            raise ValueError(f"Invalid llm_path: {llm_path} must be a valid JSON file.")
        try:
            with path.open("r") as f:
                config_data = json.load(f)
            values["max_position_embeddings"] = config_data[max_position_embeddings_kw]
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error reading config from {llm_path}: {e}")
        return values


def get_all_configs() -> dict[str, LLMConfig]:
    """Return instance of `LLMConfig` from `model_name_short`.

    Assumes that the config file is in yaml format saved as $MODEL_HOME/models.yaml

    Returns
    -------
        Dict of str: LLMConfig.

    """
    model_home = getenv("MODEL_HOME")
    if model_home is None:
        raise MissingEnvironmentVariable("MODEL_HOME is not set")
    with open(Path(model_home, "models.yaml"), "r") as f:
        all_model_configs = yaml.safe_load(f)["Models"]
    for k, v in all_model_configs.items():
        v["llm_path"] = Path(model_home, v["llm_path"])
        all_model_configs[k] = LLMConfig(**v)
    return all_model_configs


def get_llm_config(model_name_short: str, config: dict | str | PathLike = None) -> LLMConfig:
    """Return instance of `LLMConfig` from `model_name_short`.

    config can either be a dict or a path to a config yaml file.
    If no config is provided, the config file is assumed to be saved as $MODEL_HOME/models.yaml

    Args:
    ----
        model_name_short (str): Must match one of the keys in model config.
        config (dict|str| PathLike): Optional. Model configuration dict or path to config location. Default = None.

    Returns:
    -------
        Instance of LLMConfig.

    """
    if not config:
        model_home = getenv("MODEL_HOME")
        if model_home is None:
            raise MissingEnvironmentVariable("MODEL_HOME is not set. Either set MODEL_HOME or pass a config.")
        config = Path(model_home, "models.yaml")
    else:
        model_home = ""
    if not isinstance(config, dict):
        config_path = Path(config)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    if model_name_short not in config["Models"]:
        model_list = ", ".join(list(config["Models"].keys()))
        raise KeyError(f"{model_name_short} not found. Available models are: {model_list}.")
    model_config = config["Models"][model_name_short]
    model_config["llm_path"] = Path(model_home, model_config["llm_path"])
    return LLMConfig(**model_config)
