FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates zstd && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama the same way as on a pod (gets latest version with mistral4 support)
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN pip3 install --break-system-packages --no-cache-dir runpod==1.7.0 requests

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
