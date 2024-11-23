"""Basic LLM API server endpoints."""

import os
import time

from fastapi import Depends, FastAPI, HTTPException, status

from exl2.api.data import GenerateRequest
from exl2.api.locllm import LocLLMManager

EXL2_MODEL = os.environ["EXL2_MODEL"]

# Initialize FastAPI
app = FastAPI()


def initialize_model_manager(MODEL_NAME_SHORT: str) -> LocLLMManager:
    manager = LocLLMManager.from_config(MODEL_NAME_SHORT)
    return manager


def get_model_manager():
    return model_manager


@app.on_event("startup")
async def startup_event():
    global model_manager
    model_manager = initialize_model_manager(EXL2_MODEL)


@app.on_event("shutdown")
async def shutdown_event():
    global model_manager
    model_manager = None


@app.post("/reload-model")
async def reload_model(manager: LocLLMManager = Depends(get_model_manager)):
    manager.reload_model()
    return {"status": "Model reloaded"}


@app.post("/api/generate")
async def generate_text(request: GenerateRequest, model_manager: LocLLMManager = Depends(get_model_manager)):
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    settings = request.dict(exclude={"prompt", "max_new_tokens", "system_prompt"})
    try:
        formatted_prompt = model_manager.format_prompt_for_single_reponse(request.prompt, request.system_prompt)
        time_start = time.time()
        output = model_manager.generate(formatted_prompt, settings, request.max_new_tokens)
        elapsed_time = time.time() - time_start
        output_ids = model_manager.encode_prompt(output)
        tok_sec = round((output_ids.numel() + 1) / elapsed_time, 1)
        return {
            "output": output,
            "formatted_prompt": formatted_prompt,
            "tok_per_sec": tok_sec,
            "model": model_manager.model_name_short,
            "generation_time": round(elapsed_time, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tokenize")
async def encode_text(text: str, model_manager: LocLLMManager = Depends(get_model_manager)):
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    return {"encoded_text": model_manager.encode_prompt(text, to_list=True), "model": model_manager.model_name_short}


@app.get("/api/model_info")
async def model_info(model_manager: LocLLMManager = Depends(get_model_manager)):
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    return {"model": model_manager.model_name, "model_name_short": model_manager.model_name_short}
