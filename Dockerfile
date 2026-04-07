FROM ghcr.io/abetlen/llama-cpp-python:latest-cuda12.4.1

RUN pip install --no-cache-dir runpod==1.7.0

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
