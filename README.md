# parakeet-stt

A fast, lightweight Speech-to-Text (STT) server based on [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — a multilingual model supporting 25 languages including Russian and Ukrainian.

Exposes an **OpenAI Whisper-compatible API**, so it works as a drop-in replacement for any app that uses the Whisper API (Superwhisper, n8n, custom scripts, etc.).

Built on [onnx-asr](https://github.com/istupakov/onnx-asr) — a lightweight ONNX inference wrapper with no PyTorch dependency.

---

## Features

- OpenAI Whisper API compatible (`POST /v1/audio/transcriptions`)
- Multilingual: English, Russian, Ukrainian, and 22 more languages
- Fast: RTF ~0.05x on CPU (11s of audio ≈ 0.5s processing)
- Three execution providers: **CPU**, **CUDA** (NVIDIA), **DirectML** (AMD/Intel via WSL2)
- Accepts any audio format ffmpeg can read (`.oga`, `.mp3`, `.wav`, `.m4a`, etc.)
- Model cached locally — no re-download on restart

---

## Quick start

### Requirements

- Docker + Docker Compose
- For CUDA: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### CPU (default)

```bash
docker compose up -d --build
```

### NVIDIA GPU — nvidia-container-toolkit (standard)

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

```bash
docker compose -f docker-compose.cuda.yml up -d --build
```

### NVIDIA GPU — Proxmox LXC (direct device passthrough)

For Proxmox LXC containers where nvidia-container-toolkit is not available.
GPU devices are passed through directly from the host.

```bash
docker-compose -f docker-compose.cuda.lxc.yml up -d --build
```

### Test

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.ogg \
  -F language=ru
# {"text":"transcribed text here"}
```

---

## Configuration

### Execution providers

The provider is set at **build time** via `RUNTIME` and at **runtime** via `PROVIDER`.
Both must match.

| `RUNTIME` (build arg) | `PROVIDER` (env var) | When to use |
|---|---|---|
| `cpu` (default) | `cpu` | Any machine, no GPU required |
| `cuda` | `cuda` | NVIDIA GPU (RTX, Tesla, etc.) |
| `directml` | `directml` | AMD/Intel GPU via WSL2 |

To switch provider — rebuild the image:
```bash
docker compose -f docker-compose.cuda.yml up -d --build
```

### Environment variables

| Variable | Default | Options | Description |
|---|---|---|---|
| `PROVIDER` | `cpu` | `cpu` / `cuda` / `directml` | Execution provider |
| `PARAKEET_QUANTIZATION` | `int8` | `int8` / `fp16` / `fp32` | Model precision — use `fp16` for CUDA |
| `HF_HOME` | `/models` | any path | Where to cache the model |

### Recommended quantization per provider

The current ONNX export (`istupakov/parakeet-tdt-0.6b-v3-onnx`) only provides `int8` weights.
Use `int8` for all providers.

---

## API reference

Compatible with the [OpenAI Audio Transcriptions API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

### `POST /v1/audio/transcriptions`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | Audio file (any format ffmpeg supports) |
| `model` | string | no | Accepted for API compatibility, ignored |
| `language` | string | no | Language code, e.g. `ru`, `en`, `uk`. Default: `ru` |
| `response_format` | string | no | `json` (default) or `text` |

**Response (`json`):**
```json
{"text": "transcribed text"}
```

**Response (`text`):**
```
transcribed text
```

### `GET /health`

```json
{"status": "ok"}
```

---

## Using with Superwhisper

In Superwhisper settings, configure a custom OpenAI-compatible server:

- **API URL:** `http://your-server-ip:8000`
- **Model:** `whisper-1` (or any string — it's ignored)
- **API Key:** anything (not validated)

---

## Using with n8n

Node type: `HTTP Request`

| Setting | Value |
|---|---|
| Method | POST |
| URL | `http://parakeet-stt:8000/v1/audio/transcriptions` |
| Body | Multipart Form Data |
| Field name | `file` |

---

## NVIDIA GPU setup

Install nvidia-container-toolkit on the host:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is accessible inside Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Then start the server:
```bash
docker compose -f docker-compose.cuda.yml up -d --build
```

---

## Supported languages

English, Spanish, French, German, Russian, Ukrainian, Polish, Italian, Portuguese, Dutch,
Czech, Slovak, Romanian, Hungarian, Bulgarian, Croatian, Slovenian, Danish, Swedish,
Finnish, Estonian, Latvian, Lithuanian, Greek, Maltese.

---

## Credits

- **Model:** [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — CC-BY-4.0
- **ONNX export & onnx-asr library:** [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr) — MIT

---

## License

MIT — see [LICENSE](LICENSE).

> The underlying model (nvidia/parakeet-tdt-0.6b-v3) is licensed under CC-BY-4.0,
> which requires attribution when redistributing the model weights.
> This project does not redistribute the weights — they are downloaded directly from HuggingFace.
