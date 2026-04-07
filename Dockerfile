FROM runpod/base:0.6.2-cuda12.2.0

RUN pip3 install --no-cache-dir \
    llama-cpp-python==0.3.8 \
    runpod==1.7.0

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
