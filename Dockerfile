FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --break-system-packages --no-cache-dir runpod==1.7.0 requests

COPY handler.py /handler.py

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
