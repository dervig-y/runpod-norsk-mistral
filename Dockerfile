# Build llama-cpp-python with CUDA support
# nvidia/cuda devel image has CUDA toolkit needed for compilation
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

# Build llama-cpp-python with CUDA
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip3 install --no-cache-dir \
    llama-cpp-python==0.3.8 \
    runpod==1.7.0

COPY handler.py /handler.py

# Verify imports work
RUN python3 -c "import runpod; from llama_cpp import Llama; print('OK')"

CMD ["python3", "-u", "/handler.py"]
