FROM runpod/base:0.6.2-cuda12.2.0

# Install prebuilt llama-cpp-python wheel with CUDA 12 support (no compilation needed)
RUN pip install --no-cache-dir \
    llama-cpp-python==0.3.8 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
    && pip install --no-cache-dir runpod==1.7.0

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
