"""RunPod serverless handler - llama-server reads GGUF directly from volume."""

import os
import subprocess
import time
import requests
import runpod

GGUF_PATH = "/runpod-volume/models/norsk-mistral-119b-gguf/m51Lab-NorskMistral-119B-Q4_K_M.gguf"
LLAMA_HOST = "http://127.0.0.1:8080"

# Find llama-server binary
import shutil
LLAMA_BIN = shutil.which("llama-server") or shutil.which("llama-server", path="/app:/usr/local/bin:/usr/bin:/bin")
if not LLAMA_BIN:
    # Search for it
    result = subprocess.run(["find", "/", "-name", "llama-server", "-type", "f"], capture_output=True, text=True, timeout=10)
    candidates = result.stdout.strip().split("\n")
    LLAMA_BIN = candidates[0] if candidates[0] else None
if not LLAMA_BIN:
    raise RuntimeError("llama-server binary not found")

print(f"Using llama-server at {LLAMA_BIN}")
print(f"Starting with {GGUF_PATH}...")
subprocess.Popen([
    LLAMA_BIN,
    "-m", GGUF_PATH,
    "--host", "0.0.0.0",
    "--port", "8080",
    "-ngl", "99",
    "-c", "4096",
])

# Wait for server to load model into GPU
for i in range(300):
    try:
        r = requests.get(f"{LLAMA_HOST}/health", timeout=2)
        if r.status_code == 200 and r.json().get("status") == "ok":
            print(f"llama-server ready after {i}s")
            break
    except Exception:
        pass
    if i % 30 == 0 and i > 0:
        print(f"Loading model... {i}s")
    time.sleep(1)
else:
    raise RuntimeError("llama-server failed to start")

print("Worker ready.")


def handler(job):
    inp = job["input"]
    try:
        resp = requests.post(
            f"{LLAMA_HOST}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": inp.get("prompt", "")}],
                "max_tokens": inp.get("num_predict", 1024),
                "temperature": inp.get("temperature", 0.7),
                "stream": False,
            },
            timeout=300,
        )
        if resp.status_code != 200:
            return {"error": f"llama-server {resp.status_code}: {resp.text[:500]}"}

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        return {
            "response": choice.get("message", {}).get("content", ""),
            "eval_count": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
