from .chat_prompts import PromptFormat


def format_prompt_for_single_reponse(prompt_format: PromptFormat, user_prompt: str, system_prompt: str | None = None):
    """Format a user_prompt and optional custom system promp.

    Args:
    ----
        prompt_format (PromptFormat): An instance of PrompFormat.
        user_prompt (str): Raw user prompt.
        system_prompt (str): Optional. Raw system prompt. Will use defaul system prompt if none is provided.
            Default = None.

    Returns:
    -------
        Formatted str of user_prompt and system_prompt, including special tokens.

    """
    if not system_prompt:
        system_prompt = prompt_format.default_system_prompt()
    return (
        prompt_format.first_prompt().replace("<|system_prompt|>", system_prompt).replace("<|user_prompt|>", user_prompt)
    )


def encode_prompt(tokenizer, prompt_format: PromptFormat, text: str):
    """Encode a string with tokenizer.

    Args:
    ----
        tokenizer
        prompt_format (PromptFormat): An instance of PrompFormat.
        text (str): Raw text.

    Returns:
    -------
        A pytorch Tensor.

    """
    add_bos, add_eos, encode_special_tokens = prompt_format.encoding_options()
    return tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos, encode_special_tokens=encode_special_tokens)
