FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive

# Install build deps for llama-cpp-python CUDA build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

# Install llama-cpp-python with CUDA support + runpod SDK
RUN CMAKE_ARGS="-DGGML_CUDA=on" \
    pip install --no-cache-dir \
    llama-cpp-python==0.3.8 \
    runpod==1.7.0

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
