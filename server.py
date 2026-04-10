import os
import tempfile
import logging
import subprocess
import time
import onnx_asr
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

QUANTIZATION = os.getenv("PARAKEET_QUANTIZATION", "int8")

logger.info(f"Loading nemo-parakeet-tdt-0.6b-v3 | quantization={QUANTIZATION}")
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization=QUANTIZATION)
logger.info("Model ready")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), language: str = Form(default="ru")):
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
        text = model.recognize(wav_path)
        elapsed = time.time() - t0
        rtf = elapsed / duration if duration > 0 else 0.0
        logger.info(f"Done in {elapsed:.2f}s | RTF={rtf:.2f}x | text={repr(str(text)[:80])}")

        return {"text": str(text)}

    finally:
        os.unlink(input_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
