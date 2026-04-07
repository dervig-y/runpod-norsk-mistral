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
    """Create model from GGUF if needed."""
    tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
    models = [m["name"] for m in tags.get("models", [])]
    print(f"Models: {models}")

    if any(MODEL_NAME in m for m in models):
        # Test that the model actually loads
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate",
                json={"model": MODEL_NAME, "prompt": "test", "stream": False,
                      "options": {"num_predict": 1}}, timeout=120)
            if r.status_code == 200:
                print(f"Model {MODEL_NAME} verified OK.")
                return True
            print(f"Model exists but failed to load ({r.status_code}), re-creating...")
            requests.delete(f"{OLLAMA_HOST}/api/delete", json={"name": MODEL_NAME}, timeout=30)
        except Exception as e:
            print(f"Model test failed: {e}, re-creating...")
            requests.delete(f"{OLLAMA_HOST}/api/delete", json={"name": MODEL_NAME}, timeout=30)

    if not os.path.exists(GGUF_PATH):
        print(f"ERROR: GGUF not found at {GGUF_PATH}")
        return False

    print(f"Creating model from {GGUF_PATH}...")
    modelfile = f"FROM {GGUF_PATH}\n"
    with open("/tmp/Modelfile", "w") as f:
        f.write(modelfile)

    result = subprocess.run(
        ["ollama", "create", MODEL_NAME, "-f", "/tmp/Modelfile"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "OLLAMA_HOST": OLLAMA_HOST},
    )
    print(f"Create output: {result.stdout}")
    if result.returncode != 0:
        print(f"Create error: {result.stderr}")
        return False

    print(f"Model {MODEL_NAME} created.")
    return True


# Start Ollama
subprocess.Popen(
    ["ollama", "serve"],
    env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"},
)

if not wait_for_ollama():
    raise RuntimeError("Ollama failed to start")

print(f"Ollama ready. Ensuring model...")
if not ensure_model():
    print("WARNING: Model setup failed.")

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
