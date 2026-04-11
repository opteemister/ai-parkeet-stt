# parakeet-stt

A fast, lightweight Speech-to-Text server based on [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — a multilingual model supporting 25 languages including Russian and Ukrainian.

Exposes an **OpenAI Whisper-compatible API**, making it a drop-in replacement for any app that uses the Whisper API (Superwhisper, n8n, custom scripts, etc.).

Built on [onnx-asr](https://github.com/istupakov/onnx-asr) — a lightweight ONNX inference wrapper with no PyTorch dependency.

---

## Why Parakeet TDT?

Parakeet TDT uses a **non-autoregressive TDT (Token-and-Duration Transducer) decoder**, which means the entire output is computed in a single forward pass — no beam search, no iterative decoding. This makes it fundamentally faster than autoregressive models like Whisper on the same hardware.

---

## Execution providers

Three execution backends are supported. Choose based on your hardware:

| Provider | Build arg | Env var | Use when |
|---|---|---|---|
| **CPU** | `RUNTIME=cpu` | `PROVIDER=cpu` | Any machine, no GPU required |
| **CUDA** | `RUNTIME=cuda` | `PROVIDER=cuda` | NVIDIA GPU, quick setup |
| **TensorRT** | `RUNTIME=tensorrt` | `PROVIDER=tensorrt` | NVIDIA GPU, maximum performance |

---

## Performance comparison

Tested on real voice recordings (Russian, ~15-20s clips).

| Provider | Quantization | RAM | VRAM | RTFx — Ryzen 7 6800H | RTFx — EPYC 7K62 ¹ | RTFx — RTX 3090 | 20s clip | vs CPU |
|---|---|---|---|---|---|---|---|---|
| CPU | int8 | ~0.7 GB | — | ~20x | ~20x | — | ~1.0s | 1× baseline |
| CPU | fp32 | ~2.5 GB | — | ~20x | ~20x | — | ~1.0s | 1× baseline |
| CUDA | fp32 | ~1 GB | ~4 GB | — | — | ~17x | ~1.2s | ~0.9× ³ |
| TensorRT | fp32 | ~4 GB ² | ~4 GB | — | — | ~200x | **~0.10s** | **~10×** |
| TensorRT | fp16 ⁴ | ~4 GB ² | ~2.5 GB | — | — | ~200x | **~0.10s** | **~10×** |

> **RTFx** = audio duration ÷ processing time. RTFx 20 = 20 seconds of audio processed per second.

> ¹ EPYC 7K62 required NUMA tuning to reach ~20x. Default setting (all 30 cores) gave only ~2x. See [CPU NUMA tuning](#cpu-numa-tuning).

> ² RAM peaks at ~6 GB during first-run TRT engine compilation, drops back to ~2-4 GB at runtime.

> ³ Plain CUDA without TensorRT is barely faster than an optimized CPU on this hardware — not worth the GPU for CUDA alone.

> ⁴ TRT fp16: the ONNX model is fp32, TRT compiles it to fp16 internally. Less VRAM, same accuracy.

**For reference** — [istupakov's benchmarks](https://istupakov.github.io/onnx-asr/benchmarks/) on other hardware:

| Provider | Quantization | RTFx — T4 | RTFx — RTX (4090/5070Ti) |
|---|---|---|---|
| CPU | fp32 | — | 33 |
| CUDA | fp32 | 44 | 75 |
| TensorRT | fp32 | 83 | 253 |
| TensorRT | fp16 | 190 | 279 |

---

## Provider deep dive

### CPU

**Recommended quantization:** `int8` (smaller model, less RAM)

CPU inference uses ONNX Runtime's `CPUExecutionProvider`. The TDT decoder parallelizes well across cores, making CPU performance surprisingly competitive.

**Key insight — int8 vs fp32 on CPU:**
According to benchmarks, `fp32` is marginally faster than `int8` on CPU (33 vs 31 RTFx). This is counterintuitive but happens because modern CPUs have wide fp32 SIMD units (AVX-512) while ONNX Runtime's int8 path requires dequantization overhead. In practice the difference is small — use `int8` to save RAM.

**Key insight — Ryzen vs EPYC:**
On AMD Ryzen 7 6800H (single NUMA node, 8 cores), performance was ~20 RTFx out of the box with no tuning needed. On EPYC 7K62 (48 cores, 8 NUMA nodes × 6 cores each), the default setting of spawning one thread per logical CPU caused massive cross-NUMA memory latency, dropping to ~2 RTFx. After limiting `INTRA_THREADS=16` (spanning ~3 NUMA nodes), EPYC matched Ryzen at ~20 RTFx. See [CPU tuning](#cpu-numa-tuning).

---

### CUDA

**Recommended quantization:** `fp32` (required — int8 will not work)

**Why int8 doesn't work on CUDA:**
ONNX Runtime's `CUDAExecutionProvider` only has GPU kernels for fp32 and fp16 operations. When you load an int8 quantized model and run it with CUDA, ORT silently falls back all int8 ops to CPU. The result is a graph with **742 Memcpy nodes** — constant CPU↔GPU data transfers for every operation. This performs *worse* than running purely on CPU. The fix is to use the unquantized fp32 model (`PARAKEET_QUANTIZATION=fp32`), which maps cleanly to native CUDA kernels and results in only 2 Memcpy nodes.

---

### TensorRT

**Recommended quantization:** `fp32` input model, `TRT_FP16=true` (TRT compiles to fp16)

TensorRT is NVIDIA's inference optimizer. It takes the fp32 ONNX model, analyzes the full computation graph, fuses operations, selects optimal CUDA kernels, and compiles everything into a hardware-specific binary engine. This is why it's 3-10x faster than plain CUDA.

**Key insight — batch size profile:**
TRT builds an engine optimized for a range of input shapes. The default profile allows batch sizes 1–16. Since we always transcribe one audio clip at a time, setting `TRT_PROFILE_BATCH=1` tells TRT to optimize exclusively for batch=1 — a smaller, faster engine. This is configured automatically in `server.py`.

**Key insight — fp16 compilation:**
Setting `TRT_FP16=true` (default) tells TRT to compile the fp32 ONNX weights into fp16 precision. This halves VRAM usage (~4 GB → ~2.5 GB) and is faster on Tensor Core GPUs (all RTX series). Accuracy is not meaningfully affected for ASR tasks.

**Key insight — engine compilation:**
The first time TRT runs, it compiles the engine for your specific GPU. This takes **2–5 minutes** and uses ~6 GB of RAM. The compiled engine is cached to `./trt_engines/` and reused on all subsequent starts. A warmup inference runs automatically at startup to ensure the engine is fully initialized before the first real request.

**Key insight — first request warmup:**
Even with a cached engine, the first inference after startup incurs a one-time JIT initialization cost (~15-20s). `server.py` automatically runs a silent 1-second warmup clip at startup to absorb this cost before the server begins accepting requests.

---

## Quick start

### CPU (default)

```bash
docker compose up -d --build
```

### CUDA — standard (nvidia-container-toolkit)

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

```bash
docker compose -f docker-compose.cuda.yml up -d --build
```

### CUDA — Proxmox LXC (direct device passthrough)

For Proxmox LXC containers where nvidia-container-toolkit is not available.

```bash
docker compose -f docker-compose.cuda.lxc.yml up -d --build
```

### TensorRT — Proxmox LXC

Maximum performance. Image build takes longer (~10 min) due to the TensorRT pip package (~1 GB).
First startup compiles the TRT engine (~2-5 min). Subsequent startups use the cached engine.

```bash
mkdir -p trt_engines
docker compose -f docker-compose.tensorrt.lxc.yml up -d --build
```

### Test

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F language=ru
# {"text":"transcribed text here"}
```

---

## Configuration

### Environment variables

| Variable | Default | Options | Description |
|---|---|---|---|
| `PROVIDER` | `cpu` | `cpu` / `cuda` / `tensorrt` / `directml` | Execution provider — must match `RUNTIME` build arg |
| `PARAKEET_QUANTIZATION` | `int8` | `int8` / `fp32` | Model precision. Use `fp32` for CUDA/TRT |
| `HF_HOME` | `/models` | any path | Model cache directory |
| `INTRA_THREADS` | cpu count | integer | Threads per operation. Tune for NUMA (see below) |
| `INTER_THREADS` | cpu count | integer | Threads between operations |
| `TRT_WORKSPACE_GB` | `1` | integer | GPU memory for TRT engine compilation workspace |
| `TRT_FP16` | `true` | `true` / `false` | Compile TRT engine in fp16 (recommended) |
| `TRT_MAX_AUDIO_SEC` | `70` | integer | Max audio length the TRT engine handles; longer clips fall back to CUDA |
| `TRT_CACHE_PATH` | `/trt_engines` | any path | Where to cache compiled TRT engines |

### Quantization per provider

| Provider | Use | Why |
|---|---|---|
| CPU | `int8` | Smaller model (~0.7 GB), negligible speed difference |
| CUDA | `fp32` | int8 has no CUDA kernels in ORT → falls back to CPU with 742 data transfers |
| TensorRT | `fp32` | TRT input must be fp32; set `TRT_FP16=true` to compile engine in fp16 |

---

## CPU NUMA tuning

Relevant if you run on multi-socket or high-core-count CPUs (EPYC, Threadripper, Xeon).

**What is NUMA?**
NUMA (Non-Uniform Memory Access) means memory is physically split across multiple nodes, each attached to a subset of CPU cores. Accessing memory from a non-local node is 2-5x slower. EPYC 7K62 has 8 NUMA nodes × 6 cores each.

**The problem:**
By default, ONNX Runtime spawns one thread per logical CPU. On a 48-core EPYC this means 48 threads competing across 8 NUMA nodes, causing constant cross-node memory access. Result: ~2 RTFx instead of ~20 RTFx.

**The fix:**
Limit thread count to keep threads within a small number of NUMA nodes:

```bash
# Find your NUMA topology
lscpu | grep -E 'NUMA|Core|Socket'
numactl --hardware   # shows NUMA nodes and their CPUs
```

```yaml
# docker-compose.yml
environment:
  - INTRA_THREADS=16  # tune: start with cores-per-node, increase until RTFx stops improving
  - INTER_THREADS=16
```

**Tuning results on EPYC 7K62** (48 logical CPUs, 8 NUMA nodes × 6 cores):

| INTRA_THREADS | RTFx | Notes |
|---|---|---|
| 30 (default) | ~2 | cross-NUMA memory thrashing |
| 6 | ~12 | one NUMA node |
| 16 | ~20 | optimal — spans ~3 nodes without overloading the memory bus |

**Single-node CPUs** (Ryzen, Core i-series, most laptop CPUs) have a single NUMA node and don't need tuning — they reach maximum performance with default settings.

---

## Proxmox LXC setup

For running inside a Proxmox LXC container without nvidia-container-toolkit.

**Prerequisites in the LXC config** (`/etc/pve/lxc/<ID>.conf`):

```
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 509:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-caps/nvidia-cap1 dev/nvidia-caps/nvidia-cap1 none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-caps/nvidia-cap2 dev/nvidia-caps/nvidia-cap2 none bind,optional,create=file
```

The compose files (`docker-compose.cuda.lxc.yml`, `docker-compose.tensorrt.lxc.yml`) handle the rest:
- Pass `/dev/nvidia*` devices into the container
- Mount `/usr/lib/x86_64-linux-gnu/libcuda.so.1` from the host (the driver API library is not bundled in pip packages and must come from the host driver installation)

**Why libcuda.so must be mounted:**
`onnxruntime-gpu` bundles CUDA 12 runtime libraries (cuBLAS, cuDNN, cuFFT, etc.) via pip. The one library it cannot bundle is `libcuda.so` — the NVIDIA driver API. This library is part of the GPU driver and lives on the host. Without mounting it, CUDA initialization fails with `CUDA driver version insufficient` (error 35).

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

**Response (`text`):** plain string — used by some clients like Superwhisper.

### `GET /health`

```json
{"status": "ok"}
```

---

## Using with Open WebUI

1. Go to **Settings → Audio**
2. Set **STT Engine** to `OpenAI`
3. Set **OpenAI Base URL** to `http://your-server-ip:8000/v1`
4. Set **OpenAI API Key** to any string (not validated)
5. Set **STT Model** to `parakeet` (or any string — ignored)
6. Click **Save**

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
