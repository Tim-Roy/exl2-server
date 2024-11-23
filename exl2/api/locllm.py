from os import PathLike
from pathlib import Path
from typing import Optional

from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler

from ..chat.chat_prompts import PROMPT_FORMATS, PromptFormat
from ..chat.chat_utils import encode_prompt, format_prompt_for_single_reponse
from ..models import get_llm_config


class LocLLMManager:
    """Load Local LLM.

    Args:
    ----
        model_name (str): Name of model.
        model_path (str | PathLike): Path containing model weights, tokenizer, and configs.
        weight_format (str): Format of model weights.
        prompt_format (PromptFormat): Instance of PromptFormat used for chat templates.
        model_name_short (str): Optional. Will use model_name if not provided.

    """

    SUPPORTED_FORMATS = ["exl2"]

    def __init__(
        self,
        model_name: str,
        model_path: str | PathLike,
        weight_format: str,
        prompt_format: PromptFormat,
        model_name_short: str | None = None,
    ):
        if weight_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"{weight_format} not supported. Must be one of: {','.join(self.SUPPORTED_FORMATS)}")
        if not model_name_short:
            model_name_short = model_name
        self.model_name_short = model_name_short
        self.model_name = model_name
        self.model_path = model_path
        self.weight_format = weight_format
        self.prompt_format = prompt_format
        self.default_system_prompt = self.prompt_format.default_system_prompt()

        self.config = None
        self.model = None
        self.cache = None
        self.tokenizer = None
        self.stop_conditions = None
        self.load_model()

    @classmethod
    def from_config(cls, model_name_short: str, config: dict | str | PathLike = None):
        """Initiate class instance from a config.

        config can either be a dict or a path to a config yaml file.
        If no config is provided, the config file is assumed to be saved as $MODEL_HOME/models.yaml

        Args:
        ----
            model_name_short (str): Must match one of the keys in model config.
            config (dict|str| PathLike): Optional. Model configuration dict or path to config location. Default = None.

        """
        model_name_short = model_name_short
        model_mgr_config = get_llm_config(model_name_short, config)
        model_path = model_mgr_config.llm_path
        model_name = Path(model_path).name
        weight_format = model_mgr_config.weight_format
        prompt_format = PROMPT_FORMATS[model_mgr_config.prompt_format]()

        return cls(model_name, model_path, weight_format, prompt_format, model_name_short)

    def load_model(self):
        """Only exl2 format supported.

        Extend the class for additional format support.
        """
        if self.weight_format == "exl2":
            self._load_model_exl2()

    def _load_model_exl2(self):
        if not self.model:
            self.config = ExLlamaV2Config(self.model_path)
            self.model = ExLlamaV2(self.config)
            self.cache = ExLlamaV2Cache(self.model, lazy=True)
            self.model.load_autosplit(self.cache, progress=False)
            self.tokenizer = ExLlamaV2Tokenizer(self.config)
            self.stop_conditions = [x for x in self.prompt_format.stop_conditions(self.tokenizer) if x]

    def is_model_loaded(self):
        return self.model is not None

    def get_tokenizer(self):
        return self.tokenizer

    def get_stop_conditions(self):
        return self.stop_conditions

    def reload_model(self):
        self.model = None
        self.load_model()

    def format_prompt_for_single_reponse(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not system_prompt:
            system_prompt = self.default_system_prompt
        return format_prompt_for_single_reponse(self.prompt_format, prompt, system_prompt)

    def encode_prompt(self, text: str, to_list: bool = False):
        encoding = encode_prompt(self.tokenizer, self.prompt_format, text)
        if to_list:
            encoding = encoding.squeeze(0).tolist()
        return encoding

    def generate(self, prompt: str, settings: dict, max_new_tokens: int = 1000):
        if self.weight_format == "exl2":
            generated_text = self._generate_exl2(prompt, settings, max_new_tokens)
        return generated_text

    def _generate_exl2(self, prompt: str, settings: dict, max_new_tokens: int = 1000):
        gen_settings = ExLlamaV2Sampler.Settings(**settings)
        generator = ExLlamaV2DynamicGenerator(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer, gen_settings=gen_settings
        )
        generated_text = generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            completion_only=True,
            stop_conditions=self.stop_conditions,
        )
        return generated_text
