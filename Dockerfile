FROM python:3.11-slim

# RUNTIME build arg selects the ONNX execution provider:
#   cpu      (default) — works everywhere, no extra setup
#   cuda     — NVIDIA GPU, requires nvidia-container-toolkit on the host
#   directml — AMD/Intel GPU via DirectML, requires WSL2 + /dev/dxg
ARG RUNTIME=cpu

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN if [ "$RUNTIME" = "cuda" ]; then \
      pip install --no-cache-dir onnxruntime-gpu; \
    elif [ "$RUNTIME" = "directml" ]; then \
      pip install --no-cache-dir onnxruntime-directml; \
    else \
      pip install --no-cache-dir onnxruntime; \
    fi

RUN pip install --no-cache-dir onnx-asr huggingface_hub fastapi uvicorn python-multipart numpy soundfile

WORKDIR /app
COPY server.py .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
