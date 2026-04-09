import os
import tempfile
import logging
import subprocess
import time
import onnx_asr
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

_MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
QUANTIZATION = os.getenv("PARAKEET_QUANTIZATION", "int8")
PROVIDER = os.getenv("PROVIDER", "cpu").lower()

_PROVIDERS = {
    "cpu":      ["CPUExecutionProvider"],
    "cuda":     ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider",  "CPUExecutionProvider"],
}

_providers = _PROVIDERS.get(PROVIDER, _PROVIDERS["cpu"])

# Set explicit thread count to suppress pthread_setaffinity_np errors in containers
_sess_options = ort.SessionOptions()
_sess_options.intra_op_num_threads = os.cpu_count() or 4
_sess_options.inter_op_num_threads = 1

logger.info(f"Loading {_MODEL_NAME} | quantization={QUANTIZATION} | provider={PROVIDER}")
asr = onnx_asr.load_model(_MODEL_NAME, quantization=QUANTIZATION, providers=_providers, sess_options=_sess_options)
logger.info("Model ready")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="parakeet"),        # accepted for OpenAI API compat, ignored
    language: str = Form(default="ru"),
    response_format: str = Form(default="json"),  # "json" or "text"
):
    suffix = os.path.splitext(file.filename or "audio.oga")[1] or ".oga"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    wav_path = input_path + ".wav"

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", input_path],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
        logger.info(f"Audio: {duration:.1f}s ({suffix})")

        conv = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, text=True,
        )
        if conv.returncode != 0:
            logger.error(f"ffmpeg: {conv.stderr[-300:]}")
            return JSONResponse(status_code=500, content={"error": "audio conversion failed"})

        t0 = time.time()
        text = str(asr.recognize(wav_path, language=language))
        elapsed = time.time() - t0
        rtf = elapsed / duration if duration > 0 else 0.0
        logger.info(f"Done in {elapsed:.2f}s | RTF={rtf:.2f}x | text={repr(text[:80])}")

        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text}

    finally:
        os.unlink(input_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
