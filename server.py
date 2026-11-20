import logging
import logging.handlers
import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pipeline import run_pipeline, get_lid

import json as _json


class JSONFormatter(logging.Formatter):
    def format(self, record):
        return _json.dumps({
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        })


console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s %(name)-10s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))

os.makedirs("logs", exist_ok=True)
file_handler = logging.handlers.RotatingFileHandler("logs/pipeline.jsonl", maxBytes=10_000_000, backupCount=5)
file_handler.setFormatter(JSONFormatter())

logging.basicConfig(level=logging.INFO, handlers=[console, file_handler])
log = logging.getLogger("server")

app = FastAPI(
    title="Voice Note Transcription Pipeline",
    description="Audio → Language Detection → STT",
)


@app.on_event("startup")
def startup():
    get_lid()


@app.post("/transcribe")
async def transcribe(
    user_id: str = Form(..., description="User ID (required)"),
    audio: UploadFile = File(..., description="Audio file (.wav, .mp3, .ogg)"),
):
    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await audio.read())
        tmp.flush()
        try:
            result = run_pipeline(
                user_id=user_id,
                audio_path=tmp.name,
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    return {
        "user_id": result.user_id,
        "language": result.lid.language,
        "language_name": result.lid.language_name,
        "confidence": round(result.lid.confidence, 4),
        "is_indian": result.lid.is_indian,
        "lid_source": result.lid.source,
        "top3": [{"language": l, "probability": round(p, 4)} for l, p in result.lid.top3],
        "transcript": result.stt.transcript,
        "duration": result.stt.duration,
        "engine": result.stt.engine,
        "stt_latency_ms": result.stt.latency_ms,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
