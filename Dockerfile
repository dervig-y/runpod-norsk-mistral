FROM runpod/base:0.6.2-cuda12.2.0

# Install llama-cpp-python CPU build (CUDA acceleration via runtime libs on RunPod)
RUN pip install --no-cache-dir \
    llama-cpp-python==0.3.8 \
    runpod==1.7.0

COPY handler.py /handler.py

RUN python -c "import runpod; from llama_cpp import Llama; print('OK')"

CMD ["python", "-u", "/handler.py"]
