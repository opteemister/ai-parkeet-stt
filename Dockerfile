FROM python:3.11-slim

# RUNTIME build arg selects the ONNX execution provider:
#   cpu      (default) — works everywhere, no extra setup
#   cuda     — NVIDIA GPU, requires nvidia-container-toolkit on the host
#   directml — AMD/Intel GPU via DirectML, requires WSL2 + /dev/dxg
ARG RUNTIME=cpu

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install onnxruntime backend.
# For CUDA: also install NVIDIA CUDA 12 runtime libraries via pip (official NVIDIA packages).
# These are user-space libs and do not conflict with host GPU drivers.
RUN if [ "$RUNTIME" = "cuda" ]; then \
      pip install --no-cache-dir \
        onnxruntime-gpu \
        nvidia-cublas-cu12 \
        nvidia-curand-cu12 \
        nvidia-cufft-cu12 \
        nvidia-cudnn-cu12; \
    elif [ "$RUNTIME" = "directml" ]; then \
      pip install --no-cache-dir onnxruntime-directml; \
    else \
      pip install --no-cache-dir onnxruntime; \
    fi

# Make CUDA libraries discoverable at runtime (only matters when RUNTIME=cuda)
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

RUN pip install --no-cache-dir onnx-asr huggingface_hub fastapi uvicorn python-multipart numpy soundfile

WORKDIR /app
COPY server.py .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
