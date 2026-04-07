# Stage 1: Get llama-server binary from official image
FROM ghcr.io/ggml-org/llama.cpp:server-cuda AS llama

# Stage 2: Runtime with Python + llama-server
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libcurl4 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir runpod==1.7.0 requests

# Copy llama-server binary and libs from official image
COPY --from=llama /usr/local/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama /usr/local/lib/ /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
