FROM ollama/ollama

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir runpod==1.7.0 requests

COPY handler.py /handler.py

# Override Ollama's ENTRYPOINT so our handler controls startup
ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
