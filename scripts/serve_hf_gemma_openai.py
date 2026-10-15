"""OpenAI-compatible Gemma 3 server for the project's external LLM port.

Runs the unquantized BF16 Hugging Face checkpoint with Accelerate's automatic
GPU/CPU layer placement (useful when a 12B BF16 model is larger than 24 GiB).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--gpu-memory", default="14GiB")
parser.add_argument("--cpu-memory", default="90GiB")
parser.add_argument("--max-new-tokens", type=int, default=5000)
args = parser.parse_args()

api_key = os.environ.get("LLM_API_KEYS", "local").split(",")[0].strip()
if not api_key:
    raise RuntimeError("LLM_API_KEYS must contain a non-empty API key")

processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
model = Gemma3ForConditionalGeneration.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
    low_cpu_mem_usage=True,
    local_files_only=True,
).eval()

app = FastAPI(title="Local Gemma 3 OpenAI-compatible API")


def authorize(authorization: str | None) -> None:
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": content}],
                }
            )
            continue
        parts = []
        for part in content or []:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                parts.append({"type": "image", "url": url})
        normalized.append({"role": message["role"], "content": parts})
    return normalized


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    authorize(authorization)
    return {
        "object": "list",
        "data": [{"id": "gemma-3-12b-it", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    payload: dict[str, Any], authorization: str | None = Header(default=None)
) -> JSONResponse:
    authorize(authorization)
    started = time.perf_counter()
    try:
        inputs = processor.apply_chat_template(
            normalize_messages(payload.get("messages", [])),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")
        max_tokens = min(
            int(payload.get("max_tokens") or args.max_new_tokens), args.max_new_tokens
        )
        temperature = float(payload.get("temperature") or 0.0)
        generation: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation["temperature"] = temperature
            if payload.get("top_p") is not None:
                generation["top_p"] = float(payload["top_p"])
        with torch.inference_mode():
            output = model.generate(**inputs, **generation)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        output_tokens = int(output.shape[-1] - prompt_tokens)
        text = processor.decode(
            output[0, prompt_tokens:], skip_special_tokens=True
        ).strip()
    except Exception as exc:
        logging.exception("Gemma 3 generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gemma-3-12b-it",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length" if output_tokens >= max_tokens else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
            },
            "latency_seconds": round(time.perf_counter() - started, 4),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18989)
