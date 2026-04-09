# parakeet-stt — project context

STT server for the `tele-mind` project (n8n workflow: Telegram → transcribe → Obsidian).
Also works as a drop-in OpenAI Whisper API replacement for apps like Superwhisper.

---

## Why Parakeet (not Whisper)

After testing all Whisper variants (see `/Volumes/Data/System/whisper-stt/CLAUDE.md`),
Whisper could not be made fast enough on this hardware. Switched to:

**nvidia/parakeet-tdt-0.6b-v3** — multilingual (25 languages including Russian and Ukrainian),
ONNX export by istupakov, CC-BY-4.0 license.

**Why ONNX:**
- `onnx-asr` (MIT, istupakov) — thin wrapper, only numpy + onnxruntime, no PyTorch
- TDT decoder is non-autoregressive → parallelizes well on CPU and GPU
- ~20x faster than faster-whisper large-v3-turbo on the same hardware
- Model is downloaded from HuggingFace automatically on first run (~300MB int8)

---

## Hardware

### Current deployment (ser5, AMD)
| | |
|---|---|
| CPU | AMD Ryzen 7 6800H |
| GPU | AMD Radeon 680M — not used (CPU is faster in practice) |
| RAM | 32GB |
| OS | Windows 11 + WSL2 (Docker) |

Running with `PROVIDER=cpu`, `PARAKEET_QUANTIZATION=int8`, RTF ~0.05x.

### Alternative deployment (NVIDIA RTX 3090)
Fully supported — use `docker-compose.cuda.yml`.
Expected RTF < 0.01x (essentially instant for voice notes).

---

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI server, onnx-asr + ffmpeg |
| `Dockerfile` | Python 3.11-slim + ffmpeg + onnxruntime (selectable via `RUNTIME` build arg) |
| `docker-compose.yml` | CPU deployment (default) |
| `docker-compose.cuda.yml` | NVIDIA GPU deployment |

---

## Running

### CPU (default)
```bash
docker compose up -d --build
```

### NVIDIA GPU (RTX 3090 etc.)
Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.
```bash
docker compose -f docker-compose.cuda.yml up -d --build
```

---

## Configuration

### Build arg
| Arg | Values | Description |
|---|---|---|
| `RUNTIME` | `cpu` (default) / `cuda` / `directml` | Selects which onnxruntime package to install |

### Environment variables
| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `cpu` | `cpu` / `cuda` / `directml` — must match `RUNTIME` build arg |
| `PARAKEET_QUANTIZATION` | `int8` | `int8` / `fp16` / `fp32` — use `fp16` for CUDA |
| `HF_HOME` | `/models` | Model cache directory |

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

Response (`json` format):
```json
{"text": "transcribed text"}
```

Response (`text` format): plain string — used by some clients like Superwhisper.

---

## n8n integration

Used from the `tele-mind` workflow in n8n:
- URL: `http://parakeet-stt:8000/v1/audio/transcriptions` (inside `domis_network`)
- Node: `n8n-nodes-base.httpRequest`, multipart/form-data, field `file`
