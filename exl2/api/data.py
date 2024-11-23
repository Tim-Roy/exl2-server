from typing import Optional

from pydantic import BaseModel


class LLMSettings(BaseModel):
    max_new_tokens: Optional[int] = 500
    token_repetition_penalty: Optional[float] = 1.025
    token_repetition_range: Optional[int] = -1
    token_repetition_decay: Optional[float] = 0
    token_frequency_penalty: Optional[float] = 0.0
    token_presence_penalty: Optional[float] = 0.0
    temperature: Optional[float] = 0.8
    smoothing_factor: Optional[float] = 0.0
    min_temp: Optional[float] = 0
    max_temp: Optional[float] = 0.0
    temp_exponent: Optional[float] = 1.0
    top_k: Optional[float] = 50
    top_p: Optional[float] = 0.8
    top_a: Optional[float] = 0.0
    min_p: Optional[int] = 0
    tfs: Optional[int] = 0
    typical: Optional[int] = 0
    skew: Optional[int] = 0
    temperature_last: Optional[bool] = False
    mirostat: Optional[bool] = False
    mirostat_tau: Optional[float] = 1.5
    mirostat_eta: Optional[float] = 0.1
    mirostat_mu: float | None = None
    cfg_scale: float | None = None
    dry_allowed_length: Optional[int] = 2
    dry_base: Optional[float] = 1.75
    dry_multiplier: Optional[float] = 0.0
    dry_range: Optional[int] = 0
    dry_max_ngram: Optional[int] = 20
    ngram_index: Optional[int] = 0
    xtc_probability: Optional[float] = 0.0
    xtc_threshold: Optional[float] = 0.1
    xtc_ignore_tokens: frozenset[int] | None = None


class GenerateRequest(LLMSettings):
    prompt: str
    system_prompt: Optional[str] = None
