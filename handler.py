"""RunPod serverless handler for NorskMistral via Ollama."""

import os
import subprocess
import time
import requests
import runpod

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = os.environ.get("MODEL_NAME", "norsk-mistral")
GGUF_PATH = "/runpod-volume/models/norsk-mistral-119b-gguf/m51Lab-NorskMistral-119B-Q4_K_M.gguf"
MODELFILE_PATH = "/tmp/Modelfile"


def wait_for_ollama():
    """Wait for Ollama server to be ready."""
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
    """Create model from GGUF. Always re-create to avoid version mismatch."""
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
        models = [m["name"] for m in tags.get("models", [])]
        print(f"Existing models: {models}")

        # Delete old model to force re-create with current Ollama version
        if any(MODEL_NAME in m for m in models):
            print(f"Deleting old model {MODEL_NAME} to re-create with current version...")
            requests.delete(f"{OLLAMA_HOST}/api/delete", json={"name": MODEL_NAME}, timeout=30)
    except Exception as e:
        print(f"Failed to check/delete models: {e}")

    # Model not found - create from GGUF
    if not os.path.exists(GGUF_PATH):
        print(f"ERROR: GGUF file not found at {GGUF_PATH}")
        # List what's on the volume
        for root, dirs, files in os.walk("/runpod-volume"):
            for f in files:
                print(f"  {os.path.join(root, f)}")
            if root.count(os.sep) > 3:
                break
        return False

    print(f"Creating model from {GGUF_PATH}...")
    with open(MODELFILE_PATH, "w") as f:
        f.write(f"FROM {GGUF_PATH}\n")

    result = subprocess.run(
        ["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "OLLAMA_HOST": OLLAMA_HOST},
    )
    print(f"ollama create stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"ollama create stderr: {result.stderr}")
        return False

    print(f"Model {MODEL_NAME} created successfully.")
    return True


# Start Ollama server
subprocess.Popen(
    ["ollama", "serve"],
    env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"},
)

if not wait_for_ollama():
    raise RuntimeError("Ollama server failed to start")

print("Ollama server ready.")

if not ensure_model():
    print("WARNING: Model not available. Requests will fail.")

print(f"Worker ready. Model: {MODEL_NAME}")


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

        if resp.status_code != 200:
            error_text = resp.text[:500]
            print(f"Ollama error {resp.status_code}: {error_text}")
            return {"error": f"Ollama {resp.status_code}: {error_text}"}

        data = resp.json()
        return {
            "response": data.get("response", ""),
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0),
            "total_duration": data.get("total_duration", 0),
        }
    except Exception as e:
        print(f"Handler error: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
