# parakeet-stt — project context

STT server for the `tele-mind` project (n8n workflow: Telegram → transcribe → Obsidian).
Also works as a drop-in OpenAI Whisper API replacement for apps like Superwhisper.

---

## Why Parakeet (not Whisper)

After testing all Whisper variants, Whisper could not be made fast enough on this hardware. Switched to:

**nvidia/parakeet-tdt-0.6b-v3** — multilingual (25 languages including Russian and Ukrainian),
ONNX export by istupakov, CC-BY-4.0 license.

**Why ONNX / onnx-asr:**
- Thin wrapper (MIT, istupakov) — only numpy + onnxruntime, no PyTorch
- TDT decoder is non-autoregressive → parallelizes well on CPU and GPU
- Model downloaded from HuggingFace automatically on first run

---

## Hardware & deployments

### ser5 (AMD Ryzen 7 6800H)
| | |
|---|---|
| CPU | AMD Ryzen 7 6800H (single NUMA node) |
| RAM | 32GB |
| OS | Windows 11 + WSL2 (Docker) |

Running: `PROVIDER=cpu`, `PARAKEET_QUANTIZATION=int8`, `INTRA_THREADS=16`, **RTFx ~20x**

### LXC on Proxmox (EPYC 7K62 + RTX 3090)
| | |
|---|---|
| CPU | AMD EPYC 7K62 (48 cores, 8 NUMA nodes × 6 cores) |
| GPU | NVIDIA RTX 3090 (24GB VRAM) |
| OS | Proxmox LXC + Docker (no nvidia-container-toolkit) |

Running: `PROVIDER=tensorrt`, `PARAKEET_QUANTIZATION=fp32`, `TRT_FP16=true`, `INTRA_THREADS=16`, **RTFx ~200x**

GPU via direct device passthrough (`/dev/nvidia*`) + host `libcuda.so.1` volume mount.

---

## Performance summary

| Provider | Quantization | RAM | VRAM | RTFx (Ryzen) | RTFx (EPYC) | RTFx (RTX 3090) | 20s clip |
|---|---|---|---|---|---|---|---|
| CPU | int8 | ~0.7 GB | — | ~20x | ~20x | — | ~1.0s |
| CPU | fp32 | ~2.5 GB | — | ~20x | ~20x | — | ~1.0s |
| CUDA | fp32 | ~1 GB | ~4 GB | — | — | ~17x | ~1.2s |
| TensorRT | fp16 | ~4 GB* | ~2.5 GB | — | — | ~200x | **~0.10s** |

\* Peaks at ~6 GB during first-run TRT engine compilation, drops to ~2-4 GB at runtime.

---

## Critical learnings

### int8 doesn't work on CUDA
ONNX Runtime has no CUDA kernels for int8 ops. Using int8 model with CUDAExecutionProvider
causes all ops to fall back to CPU → **742 Memcpy nodes** (constant CPU↔GPU transfers).
Worse than pure CPU. Fix: `PARAKEET_QUANTIZATION=fp32` → 2 memcpy nodes, proper GPU inference.

Code: `QUANTIZATION = None if _quant == "fp32" else _quant`
(onnx-asr uses `quantization=None` for unquantized fp32, not the string "fp32")

### NUMA tuning on EPYC
Default INTRA_THREADS = all cores (30 on EPYC 7K62) → cross-NUMA memory thrashing → RTFx ~2x.
Ryzen 7 6800H has single NUMA node → works at full speed without tuning.
EPYC optimal: `INTRA_THREADS=16` → RTFx ~20x. Start with 6 (1 NUMA node), increase until RTFx stops improving.

### TRT profile_max_shapes (critical)
Default `profile_max_shapes["batch"] = 16` → TRT builds engine for batch 1-16, suboptimal.
Must set before load_model:
```python
from onnx_asr.onnx import TensorRtOptions
TensorRtOptions.profile_max_shapes["batch"] = 1
TensorRtOptions.profile_max_shapes["waveform_len_ms"] = 70_000
```

### TRT engine caching
Set in provider options: `trt_engine_cache_enable: True`, `trt_engine_cache_path: "/trt_engines"`.
Without this, TRT recompiles on every restart even with volume mount.
Cache file: `TensorrtExecutionProvider_cache_sm86.timing` (sm86 = RTX 3090 Ampere arch).

### TRT warmup at startup
First inference after startup takes ~18s (JIT init). server.py automatically runs a 1s silent WAV
through asr.recognize() before "Model ready" to absorb this cost at startup time.

### libcuda.so on Proxmox LXC
`onnxruntime-gpu[cuda,cudnn]` bundles CUDA 12 runtime libs via pip but NOT `libcuda.so` (driver API).
Mount from host: `/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro`
Without it: `CUDA failure 35: CUDA driver version insufficient`.

### TensorRT pip package libs
`pip install tensorrt` → `libnvinfer.so.10` lands in `/usr/local/lib/python3.11/site-packages/tensorrt_libs/`.
Must be in LD_LIBRARY_PATH. Without it: `libnvinfer.so.10: cannot open shared object file`.

### fp16 quantization doesn't exist in model repo
Only `int8` and fp32 (no suffix) exist. fp16 TRT is achieved via `trt_fp16_enable: True` in
provider options — TRT compiles the fp32 ONNX model to fp16 internally.

---

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI server, onnx-asr + ffmpeg, CPU/CUDA/TRT/DirectML |
| `Dockerfile` | `RUNTIME=cpu/cuda/tensorrt/directml` build arg |
| `docker-compose.yml` | CPU default |
| `docker-compose.cuda.yml` | CUDA with nvidia-container-toolkit |
| `docker-compose.cuda.lxc.yml` | CUDA on Proxmox LXC (direct device passthrough) |
| `docker-compose.tensorrt.lxc.yml` | TensorRT on Proxmox LXC |

---

## Running

### CPU
```bash
docker compose up -d --build
```

### TensorRT (Proxmox LXC, recommended for RTX 3090)
```bash
mkdir -p trt_engines
docker compose -f docker-compose.tensorrt.lxc.yml up -d --build
```
First build: ~10 min (TRT pip ~1 GB). First start: ~2-5 min (engine compilation). Subsequent starts: fast (cached).

### CUDA only (Proxmox LXC)
```bash
docker compose -f docker-compose.cuda.lxc.yml up -d --build
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `cpu` | `cpu` / `cuda` / `tensorrt` / `directml` |
| `PARAKEET_QUANTIZATION` | `int8` | `int8` or `fp32`. Use `fp32` for CUDA/TRT |
| `HF_HOME` | `/models` | Model cache directory |
| `INTRA_THREADS` | cpu count | Threads per op — tune for NUMA topology |
| `INTER_THREADS` | cpu count | Threads between ops |
| `TRT_WORKSPACE_GB` | `1` | GPU memory for TRT compilation workspace |
| `TRT_FP16` | `true` | Compile TRT engine in fp16 (less VRAM, faster) |
| `TRT_MAX_AUDIO_SEC` | `70` | Max audio length for TRT engine; longer falls back to CUDA |
| `TRT_CACHE_PATH` | `/trt_engines` | Path for compiled TRT engine cache |

---

## API

Compatible with the OpenAI Whisper API — works as a drop-in replacement.

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

  file            <audio file>        required
  model           any string          optional, ignored (OpenAI compat)
  language        ru                  optional, default "ru"
  response_format json | text         optional, default "json"

GET /health → {"status": "ok"}
```

---

## n8n integration

Used from the `tele-mind` workflow in n8n:
- URL: `http://parakeet-stt:8000/v1/audio/transcriptions` (inside `domis_network`)
- Node: `n8n-nodes-base.httpRequest`, multipart/form-data, field `file`
