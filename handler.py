"""RunPod serverless handler for NorskMistral via Ollama."""

import os
import subprocess
import time
import requests
import runpod

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = os.environ.get("MODEL_NAME", "norsk-mistral")


def wait_for_ollama():
    """Wait for Ollama server to be ready."""
    for _ in range(60):
        try:
            r = requests.get(f"{OLLAMA_HOST}/", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# Start Ollama server on module load
subprocess.Popen(
    ["ollama", "serve"],
    env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"},
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

if not wait_for_ollama():
    raise RuntimeError("Ollama server failed to start")

print(f"Ollama ready. Model: {MODEL_NAME}")


def handler(job):
    """Handle inference request."""
    inp = job["input"]
    prompt = inp.get("prompt", "")
    temperature = inp.get("temperature", 0.7)
    num_predict = inp.get("num_predict", 1024)

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "response": data.get("response", ""),
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0),
            "total_duration": data.get("total_duration", 0),
        }
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
