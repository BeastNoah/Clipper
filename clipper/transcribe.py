"""Step 2: speech to text with faster-whisper, keeping a start and end time for every word."""

from __future__ import annotations

import os
import time
from pathlib import Path

from .util import ClipperError, cached, log

TRANSCRIBE_VERSION = 1


def _run_whisper(audio: Path, model_name: str, language: str | None, duration: float) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ClipperError("faster-whisper is not installed: pip install faster-whisper") from e

    log(f"transcribe: loading whisper model '{model_name}' (downloaded once on first use)")
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    segments, info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,  # stops one misheard phrase from repeating through the rest
    )
    log(f"transcribe: language {info.language} ({info.language_probability:.0%})")

    words: list[dict] = []
    t0, last_report = time.monotonic(), 0.0
    for seg in segments:
        for w in seg.words or []:
            text = w.word
            if not text.strip():
                continue
            start = max(float(w.start), words[-1]["s"] if words else 0.0)
            end = max(float(w.end), start + 0.01)
            words.append({"w": text, "s": round(start, 3), "e": round(end, 3), "p": round(float(w.probability), 3)})
        if duration and time.monotonic() - last_report > 10:
            last_report = time.monotonic()
            speed = seg.end / max(last_report - t0, 1e-6)
            log(f"transcribe: {seg.end / duration:4.0%} ({speed:.1f}x real time)")

    if not words:
        raise ClipperError("No speech was found in the audio.")
    return {"language": info.language, "model": model_name, "words": words}


def transcribe(work: Path, meta: dict, model_name: str = "small", language: str | None = None,
               force: bool = False) -> dict:
    params = {"version": TRANSCRIBE_VERSION, "model": model_name, "language": language,
              "duration": round(meta["duration"], 2)}
    return cached(
        work / "transcript.json", params,
        lambda: _run_whisper(work / "audio.wav", model_name, language, meta["duration"]),
        force=force, label="transcribe",
    )
