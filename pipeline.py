"""
Voice Note Transcription Pipeline
──────────────────────────────────
Two-stage LID:
  1. lid9 (bundled, 9 Indian languages) — fast, in-process
  2. If lid9 confidence < 90% → VoxLingua107 (HTTP, 107 languages)

STT routing:
  - lid9 confidence >= 90%  → CB Vinu STT (that Indian language)
  - lid9 confidence < 90%   → VoxLingua107 → Whisper (audio + language code)
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jwt
import requests
import websockets

from lid9 import LanguageID

import config

log = logging.getLogger("pipeline")

# ── LID9 setup ────────────────────────────────────────────────────

_lid_instance: Optional[LanguageID] = None


def get_lid() -> LanguageID:
    global _lid_instance
    if _lid_instance is None:
        log.info("LID  | loading lid9 ONNX model")
        _lid_instance = LanguageID()
        log.info("LID  | model loaded, languages=%s", _lid_instance.languages)
    return _lid_instance


LID9_TO_ISO = {
    "bengali": "bn", "gujarati": "gu", "hindi": "hi",
    "kannada": "kn", "malayalam": "ml", "marathi": "mr",
    "tamil": "ta", "telugu": "te", "english": "en",
}

CB_VINU_LANGUAGES = {
    "bn": "bengali", "gu": "gujarati", "hi": "hindi",
    "kn": "kannada", "ml": "malayalam", "mr": "marathi",
    "ta": "tamil", "te": "telugu", "en": "english",
}

LID9_CONFIDENCE_THRESHOLD = 0.90


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class LIDResult:
    language: str
    language_name: str
    confidence: float
    is_indian: bool
    top3: list
    source: str  # "lid9" or "voxlingua107"


@dataclass
class STTResult:
    transcript: str
    language: str
    duration: float
    engine: str
    latency_ms: float


@dataclass
class PipelineResult:
    user_id: str
    audio_path: str
    lid: LIDResult
    stt: STTResult


# ── Stage 1: lid9 (bundled) ──────────────────────────────────────

def detect_language_lid9(audio_path: str) -> LIDResult:
    log.info("LID  | stage 1 — lid9 for %s", Path(audio_path).name)
    lid = get_lid()

    start = time.time()
    r = lid.identify(audio_path, min_confidence=0.0)
    latency = (time.time() - start) * 1000

    lang_name = r.language or "unknown"
    iso = LID9_TO_ISO.get(lang_name, lang_name)

    log.info("LID  | lid9 result: %s (%s) confidence=%.4f latency=%.0fms",
             iso, lang_name, r.confidence, latency)
    return LIDResult(
        language=iso,
        language_name=lang_name,
        confidence=r.confidence,
        is_indian=iso in CB_VINU_LANGUAGES,
        top3=[(LID9_TO_ISO.get(l, l), p) for l, p in r.top_k[:3]],
        source="lid9",
    )


# ── Stage 2: VoxLingua107 (HTTP fallback) ────────────────────────

def detect_language_voxlingua(audio_path: str) -> LIDResult:
    log.info("LID  | stage 2 — voxlingua107 for %s", Path(audio_path).name)
    with open(audio_path, "rb") as f:
        start = time.time()
        resp = requests.post(
            f"{config.VOXLINGUA_URL}/identify",
            files={"audio": (Path(audio_path).name, f)},
            params={"top_k": 5},
            timeout=30,
        )
        latency = (time.time() - start) * 1000
    resp.raise_for_status()
    data = resp.json()

    lang = data.get("language") or "unknown"
    conf = data.get("confidence", 0.0)
    top_k = data.get("top_k", [])
    lang_name = top_k[0].get("language_name", "") if top_k else ""

    log.info("LID  | voxlingua107 result: %s (%s) confidence=%.4f latency=%.0fms",
             lang, lang_name, conf, latency)
    return LIDResult(
        language=lang,
        language_name=lang_name,
        confidence=conf,
        is_indian=lang in CB_VINU_LANGUAGES,
        top3=[(e["language"], e["probability"]) for e in top_k[:3]],
        source="voxlingua107",
    )


# ── Combined LID ─────────────────────────────────────────────────

def detect_language(audio_path: str) -> LIDResult:
    lid9_result = detect_language_lid9(audio_path)

    if lid9_result.confidence >= LID9_CONFIDENCE_THRESHOLD:
        log.info("LID  | lid9 confident (%.4f >= %.2f), using lid9 result",
                 lid9_result.confidence, LID9_CONFIDENCE_THRESHOLD)
        return lid9_result

    log.info("LID  | lid9 not confident (%.4f < %.2f), falling back to voxlingua107",
             lid9_result.confidence, LID9_CONFIDENCE_THRESHOLD)
    try:
        return detect_language_voxlingua(audio_path)
    except Exception as e:
        log.error("LID  | voxlingua107 failed: %s, using lid9 result as fallback", e)
        return lid9_result


# ── STT ───────────────────────────────────────────────────────────

def _b2b_token() -> str:
    return jwt.encode(
        {"sub": "b2b-backend", "aud": "stt", "exp": int(time.time()) + 300},
        config.STT_B2B_TOKEN_KEY,
        algorithm="HS256",
    )


def _ensure_pcm_wav(audio_path: str) -> tuple[str, bool]:
    """Convert audio to 16kHz mono PCM WAV if needed. Returns (path, was_converted)."""
    if audio_path.endswith(".wav"):
        return audio_path, False
    wav_path = audio_path + ".16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, check=True,
    )
    return wav_path, True


async def _transcribe_indian_ws(audio_path: str, language_iso: str) -> STTResult:
    stt_name = CB_VINU_LANGUAGES[language_iso]
    token = _b2b_token()
    ws_base = config.STT_INDIAN_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{ws_base}/stt-{stt_name}/v1/stream?sample_rate=16000&token={token}"
    log.info("STT  | engine=cb_vinu_ws language=%s (%s)", language_iso, stt_name)

    wav_path, converted = _ensure_pcm_wav(audio_path)
    try:
        with open(wav_path, "rb") as f:
            f.read(44)
            audio_data = f.read()
    finally:
        if converted:
            Path(wav_path).unlink(missing_ok=True)

    start = time.time()
    transcript = ""
    async with websockets.connect(url, open_timeout=10) as ws:
        ready = await asyncio.wait_for(ws.recv(), timeout=5)
        log.info("STT  | ws ready: %s", ready)

        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            await ws.send(audio_data[i:i + chunk_size])

        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = __import__("json").loads(msg)
                if data.get("type") == "partial" and data.get("text"):
                    transcript = data["text"]
                if data.get("type") in ("final", "result"):
                    transcript = data.get("text", transcript)
                    break
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass

    latency = (time.time() - start) * 1000
    duration = len(audio_data) / (16000 * 2)
    log.info("STT  | done latency=%.0fms duration=%.1fs chars=%d", latency, duration, len(transcript))
    return STTResult(
        transcript=transcript.strip(),
        language=stt_name,
        duration=round(duration, 2),
        engine="cb_vinu_ws",
        latency_ms=round(latency, 1),
    )


def transcribe_indian(audio_path: str, language_iso: str) -> STTResult:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _transcribe_indian_ws(audio_path, language_iso)).result()
    return asyncio.run(_transcribe_indian_ws(audio_path, language_iso))


def transcribe_whisper(audio_path: str, language: Optional[str] = None) -> STTResult:
    log.info("STT  | engine=whisper language_hint=%s", language)
    with open(audio_path, "rb") as f:
        data_fields = {}
        if language:
            data_fields["language"] = language
        start = time.time()
        resp = requests.post(
            config.WHISPER_URL,
            files={"file": (Path(audio_path).name, f)},
            data=data_fields,
            timeout=120,
        )
        latency = (time.time() - start) * 1000

    resp.raise_for_status()
    data = resp.json()

    transcript = data.get("text", "").strip()
    log.info("STT  | done latency=%.0fms duration=%.1fs chars=%d", latency, data.get("duration", 0.0), len(transcript))
    return STTResult(
        transcript=transcript,
        language=data.get("language", language or "unknown"),
        duration=data.get("duration", 0.0),
        engine="whisper",
        latency_ms=round(latency, 1),
    )


def transcribe(audio_path: str, lid_result: LIDResult) -> STTResult:
    # lid9 confident about an Indian language → CB Vinu directly
    if lid_result.source == "lid9" and lid_result.is_indian:
        log.info("STT  | routing to cb_vinu (lid9 confident: %s)", lid_result.language)
        return transcribe_indian(audio_path, lid_result.language)

    # VoxLingua107 detected an Indian language with CB Vinu support → CB Vinu
    if lid_result.source == "voxlingua107" and lid_result.language in CB_VINU_LANGUAGES:
        log.info("STT  | routing to cb_vinu (voxlingua107 detected indian: %s)", lid_result.language)
        return transcribe_indian(audio_path, lid_result.language)

    # Everything else → Whisper with language hint
    lang_hint = lid_result.language if lid_result.language != "unknown" else None
    log.info("STT  | routing to whisper (language=%s, source=%s) hint=%s",
             lid_result.language, lid_result.source, lang_hint)
    return transcribe_whisper(audio_path, language=lang_hint)


# ── Pipeline ──────────────────────────────────────────────────────

def run_pipeline(
    user_id: str,
    audio_path: str,
) -> PipelineResult:
    log.info("PIPELINE | start user_id=%s file=%s", user_id, Path(audio_path).name)
    pipeline_start = time.time()

    lid_result = detect_language(audio_path)
    stt_result = transcribe(audio_path, lid_result)

    total_ms = (time.time() - pipeline_start) * 1000
    log.info("PIPELINE | done user_id=%s lid_source=%s language=%s engine=%s total=%.0fms",
             user_id, lid_result.source, lid_result.language, stt_result.engine, total_ms)

    return PipelineResult(
        user_id=user_id,
        audio_path=audio_path,
        lid=lid_result,
        stt=stt_result,
    )
