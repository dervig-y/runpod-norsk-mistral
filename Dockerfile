FROM nvidia/cuda:12.4.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and download tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Install RunPod SDK
RUN pip3 install --break-system-packages --no-cache-dir runpod==1.7.0 requests

# Download latest llama.cpp release (pre-built CUDA 12 binary)
RUN curl -L -o /tmp/llama.zip \
    "https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-server-linux-cuda12-x86_64.zip" && \
    unzip /tmp/llama.zip -d /usr/local/bin/ && \
    chmod +x /usr/local/bin/llama-server && \
    rm /tmp/llama.zip

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
