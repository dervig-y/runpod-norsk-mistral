"""RunPod serverless handler for NorskMistral via llama-server (llama.cpp)."""

import os
import subprocess
import time
import requests
import runpod

LLAMA_HOST = "http://127.0.0.1:8080"
GGUF_PATH = "/runpod-volume/models/norsk-mistral-119b-gguf/m51Lab-NorskMistral-119B-Q4_K_M.gguf"


def start_llama_server():
    """Start llama-server with the GGUF model."""
    if not os.path.exists(GGUF_PATH):
        print(f"ERROR: GGUF not found at {GGUF_PATH}")
        for root, dirs, files in os.walk("/runpod-volume", topdown=True):
            for f in files:
                print(f"  {os.path.join(root, f)}")
            if root.count(os.sep) > 4:
                dirs.clear()
        raise RuntimeError(f"Model not found: {GGUF_PATH}")

    print(f"Starting llama-server with {GGUF_PATH}...")
    proc = subprocess.Popen(
        [
            "llama-server",
            "-m", GGUF_PATH,
            "--host", "0.0.0.0",
            "--port", "8080",
            "-ngl", "99",
            "-c", "4096",
            "-fa",
        ],
    )
    return proc


def wait_for_server():
    """Wait for llama-server to be ready."""
    for i in range(300):  # Up to 5 min for model loading
        try:
            r = requests.get(f"{LLAMA_HOST}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        if i % 30 == 0 and i > 0:
            print(f"Still waiting for model to load... ({i}s)")
        time.sleep(1)
    return False


# Start server on module load
server_proc = start_llama_server()

print("Waiting for model to load into GPU memory...")
if not wait_for_server():
    raise RuntimeError("llama-server failed to start or load model")

print("llama-server ready!")


def handler(job):
    """Handle inference request via OpenAI-compatible API."""
    inp = job["input"]
    prompt = inp.get("prompt", "")
    temperature = inp.get("temperature", 0.7)
    num_predict = inp.get("num_predict", 1024)

    try:
        resp = requests.post(
            f"{LLAMA_HOST}/v1/chat/completions",
            json={
                "model": "norsk-mistral",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": num_predict,
                "temperature": temperature,
                "stream": False,
            },
            timeout=300,
        )

        if resp.status_code != 200:
            error_text = resp.text[:500]
            print(f"llama-server error {resp.status_code}: {error_text}")
            return {"error": f"Server error {resp.status_code}: {error_text}"}

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return {
            "response": message.get("content", ""),
            "eval_count": usage.get("completion_tokens", 0),
            "total_duration": 0,
        }
    except Exception as e:
        print(f"Handler error: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
