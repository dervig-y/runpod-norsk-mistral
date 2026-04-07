FROM runpod/base:0.6.2-cuda12.2.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir \
    llama-cpp-python==0.3.8 \
    runpod==1.7.0

COPY handler.py /handler.py

# Verify imports work
RUN python -c "import runpod; from llama_cpp import Llama; print('OK')"

CMD ["python", "-u", "/handler.py"]
