"""RunPod serverless handler for NorskMistral-119B GGUF inference."""

import runpod
from llama_cpp import Llama

llm = None
MODEL_PATH = "/runpod-volume/models/norsk-mistral-119b-gguf/m51Lab-NorskMistral-119B-Q4_K_M.gguf"


def load_model():
    global llm
    if llm is None:
        print(f"Loading model from {MODEL_PATH}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        print("Model loaded.")
    return llm


def handler(job):
    """RunPod serverless handler - yields streaming tokens."""
    model = load_model()
    inp = job["input"]

    messages = inp.get("messages", [])
    max_tokens = min(inp.get("max_tokens", 1024), 2048)
    temperature = inp.get("temperature", 0.7)

    stream = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield {"token": content}


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
