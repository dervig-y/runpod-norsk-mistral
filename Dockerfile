FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# This image already has llama-server with CUDA
# Just add Python + RunPod SDK on top

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --break-system-packages --no-cache-dir runpod==1.7.0 requests

COPY handler.py /handler.py

# The base image ENTRYPOINT runs llama-server directly.
# We need to override it to run our Python handler instead.
ENTRYPOINT ["/usr/bin/python3", "-u", "/handler.py"]
CMD []
