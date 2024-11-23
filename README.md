# EXL2 Server

This contains a minimal setup to standup an API server for an EXL2 formatted LLM using FastAPI and uvicorn. This example was adapted from my personal set-up, and hence has a few nuances that can easily be changed or adapted to. This set-up allows served models to be switched with ease once set-up.

The project is intended for users who have comfort writing and reading python code and working in a Linux environment. However, no python code is required for usage, and the basic shell commands required are explicitly described below.

## Installation and setup steps

#### Requirements

- NVIDIA GPU with sufficient RAM.
- Linux environment

#### Installation

1. Download a model quantized in exl2 format ([search huggingface for options](https://huggingface.co/models?search=exl2))
1. (Optional, but recommended) Create and activate a python virtual environment with python 3.10+.
1. Install [ExLlamaV2](https://github.com/turboderp/exllamav2).
1. Install [flash attention](https://github.com/Dao-AILab/flash-attention) (ExLlamaV2 dependency).
1. Clone this repo (`git clone https://github.com/Tim-Roy/exl2-server.git`)
1. Install exl2 and dependencies.
```
cd exl2-server
pip install . # Or pip3, conda etc.
```

#### Configuration

The following 3 components are needed for final configuration. This assumes that:
- All EXL2 models are saved in the same parent directory: `MODEL_HOME`
- A light weight file named `models.yaml` containing configurations are also stored in the same parent directory.
- When start the API server, and environment variable `EXL2_MODEL` will be determine which model is served. The server executable, uvicorn, does not allow custom arguments to be passed. Hence why this approach is taken.

1. **MODEL_HOME environment variable**. This is the parent path where EXL2 models can be found. Either add this to a shell config or explicitly set by:

```
export MODEL_HOME="/path/to/models" # update with the correct location
```

Or, if directly using python, add the following:

```
import os
os.environ["MODEL_HOME"] = "/path/to/models" # update with the correct location
```

2. **EXL2_MODEL environment variable** This must exactly match one of the entries in the model configuration (below). This can be set similarly to MODEL_HOME or passed as an env variable when starting the server.

```
export EXL2_MODEL="Qwen2.5-14B"
```

or

```
import os
os.environ["EXL2_MODEL"] = "Qwen2.5-14B"
```

3. **Models configuration.** An example entry is provided under *models.yaml*. To add a new entry:
- Copy an existing entry to the bottom.
- Update the key (Qwen2.5-14B in the example) to a unique descriptive name of your choosing.
- Update the path to the name of the folder.
- Update the prompt_format for the given model. Available chat formats can be found by running:
```
python -m exl2.print_prompt_formats
```
*If the you are still unsure which model format name to use for a given model, check ./exl2/chat/chat_prompts.py for each specified format.*

The remaining key-value pairs typically will not need to be changed for exl2-formatted models. However, if needed, the following entries may need to be updated:
- `config_fname`. The default value = *config.json*.
- `max_position_embeddings_kw`. The default value = *max_position_embeddings*. This is the keyword used to derive the value for `max_position_embeddings` (the context window maximum). To override this value (for example, if using YaRN YaRN), set `max_position_embeddings`. If `max_position_embeddings` is set, the value for `max_position_embeddings_kw` will be ignored.

Update the existing file and create a soft link by:

`ln -s ./models.yaml $MODEL_HOME/models.yaml`

or copy the file over and edit the copy:

`cp ./models.yaml $MODEL_HOME/models.yaml`

## Initialize server

Initialize the server directly through the terminal. For example:

```
export MODEL_HOME="/path/to/models"
export EXL2_MODEL="Qwen2.5-14B"
uvicorn exl2.server:app --host 0.0.0.0 --port 8000
```

## How to use

1. To generate a single response, use the **generate** endpoint:
```
curl -X POST "localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Write me a haiku about bears.",
        "max_new_tokens": 1000,
        "temperature": 0.9
      }'

```