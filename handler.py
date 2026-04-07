"""RunPod serverless handler for NorskMistral via Ollama."""

import os
import subprocess
import time
import requests
import runpod

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = os.environ.get("MODEL_NAME", "norsk-mistral")
GGUF_PATH = "/runpod-volume/models/norsk-mistral-119b-gguf/m51Lab-NorskMistral-119B-Q4_K_M.gguf"


def wait_for_ollama():
    for _ in range(120):
        try:
            r = requests.get(f"{OLLAMA_HOST}/", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def ensure_model():
    """Check model exists. Only create from GGUF if missing (first-time setup)."""
    tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
    models = [m["name"] for m in tags.get("models", [])]
    print(f"Models on volume: {models}")

    if any(MODEL_NAME in m for m in models):
        print(f"Model {MODEL_NAME} found. Ready.")
        return True

    # Model not found - create from GGUF (one-time setup, takes ~10 min)
    if not os.path.exists(GGUF_PATH):
        print(f"ERROR: GGUF not found at {GGUF_PATH}")
        return False

    print(f"First-time setup: creating model from {GGUF_PATH}...")
    with open("/tmp/Modelfile", "w") as f:
        f.write(f"FROM {GGUF_PATH}\n")

    result = subprocess.run(
        ["ollama", "create", MODEL_NAME, "-f", "/tmp/Modelfile"],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, "OLLAMA_HOST": OLLAMA_HOST},
    )
    print(f"Create: {result.stdout}")
    if result.returncode != 0:
        print(f"Create error: {result.stderr}")
        return False

    print(f"Model {MODEL_NAME} created. Future startups will be fast.")
    return True


# Start Ollama
subprocess.Popen(
    ["ollama", "serve"],
    env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"},
)

if not wait_for_ollama():
    raise RuntimeError("Ollama failed to start")

print("Ollama ready.")
if not ensure_model():
    print("WARNING: Model not available.")

print("Worker ready.")


def handler(job):
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

        if resp.status_code != 200:
            return {"error": f"Ollama {resp.status_code}: {resp.text[:500]}"}

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
