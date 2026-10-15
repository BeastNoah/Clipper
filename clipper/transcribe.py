"""Step 2: speech to text with word timings.

Two backends: faster-whisper (CPU, works everywhere) and mlx-whisper (Apple Silicon GPU, much faster).
The same code also re-transcribes each chosen clip with a bigger model for more accurate captions.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import wave
from pathlib import Path

import numpy as np

from .util import ClipperError, Progress, cached, log

TRANSCRIBE_VERSION = 2
SAMPLE_RATE = 16000

MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

_fw_models: dict[str, object] = {}


def mlx_available() -> bool:
    return (sys.platform == "darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx_whisper") is not None)


def resolve_backend(name: str) -> str:
    if name == "auto":
        return "mlx" if mlx_available() else "faster-whisper"
    if name == "mlx" and not mlx_available():
        raise ClipperError("The mlx backend needs a Mac with Apple Silicon and: pip install mlx-whisper")
    if name not in ("mlx", "faster-whisper"):
        raise ClipperError(f"Unknown whisper backend '{name}' (use auto, faster-whisper or mlx)")
    return name


def load_audio(path: Path, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """16 kHz mono float32 samples of the work folder's audio.wav (optionally just a time range)."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ClipperError(f"{path.name} is not 16 kHz mono 16-bit audio")
        first = max(0, int(start * SAMPLE_RATE))
        last = w.getnframes() if end is None else min(w.getnframes(), int(end * SAMPLE_RATE))
        w.setpos(min(first, w.getnframes()))
        raw = w.readframes(max(0, last - first))
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def _clean(words: list[dict]) -> list[dict]:
    """Drop empty words and make times monotonic with a positive length."""
    out: list[dict] = []
    for w in words:
        text = w["w"]
        if not text.strip():
            continue
        start = max(float(w["s"]), out[-1]["s"] if out else 0.0)
        end = max(float(w["e"]), start + 0.01)
        out.append({"w": text, "s": round(start, 3), "e": round(end, 3), "p": round(float(w.get("p", 1.0)), 3)})
    return out


def _faster_whisper(audio, model_name: str, language: str | None, vad: bool, total: float,
                    label: str) -> tuple[list[dict], str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ClipperError("faster-whisper is not installed: pip install faster-whisper") from e
    if model_name not in _fw_models:
        log(f"{label}: loading whisper model '{model_name}' (downloaded once on first use)")
        try:
            _fw_models[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8",
                                                  cpu_threads=os.cpu_count() or 4)
        except Exception as e:  # noqa: BLE001 - download and load errors come in many types
            raise ClipperError(f"Couldn't load the whisper model '{model_name}' ({type(e).__name__}: {e}). "
                               "It is downloaded from huggingface.co on first use, so check the internet "
                               "connection, or choose another model.") from e
    model = _fw_models[model_name]
    segments, info = model.transcribe(
        audio if isinstance(audio, np.ndarray) else str(audio),
        language=language,
        word_timestamps=True,
        vad_filter=vad,
        condition_on_previous_text=False,  # stops one misheard phrase from repeating through the rest
    )
    words = []
    progress = Progress(label, total)
    for seg in segments:
        words += [{"w": w.word, "s": w.start, "e": w.end, "p": w.probability} for w in seg.words or []]
        progress.update(seg.end)
    Progress.clear_line()
    return words, info.language


def _mlx(audio, model_name: str, language: str | None, label: str) -> tuple[list[dict], str]:
    import mlx_whisper

    repo = MLX_REPOS.get(model_name, model_name)
    log(f"{label}: running {repo} on the GPU (downloaded once on first use)")
    result = mlx_whisper.transcribe(
        audio if isinstance(audio, np.ndarray) else str(audio),
        path_or_hf_repo=repo, language=language, word_timestamps=True,
        condition_on_previous_text=False, verbose=None,
    )
    words = [{"w": w["word"], "s": w["start"], "e": w["end"], "p": w.get("probability", 1.0)}
             for seg in result.get("segments", []) for w in seg.get("words", [])]
    return words, result.get("language") or language or "en"


def transcribe_audio(audio, model_name: str, backend: str, language: str | None, *, vad: bool,
                     total: float, label: str) -> tuple[list[dict], str]:
    if backend == "mlx":
        words, lang = _mlx(audio, model_name, language, label)
    else:
        words, lang = _faster_whisper(audio, model_name, language, vad, total, label)
    return _clean(words), lang


def transcribe(work: Path, meta: dict, model_name: str = "small", language: str | None = None,
               backend: str = "auto", force: bool = False) -> dict:
    backend = resolve_backend(backend)
    params = {"version": TRANSCRIBE_VERSION, "model": model_name, "backend": backend,
              "language": language, "duration": round(meta["duration"], 2)}

    def compute() -> dict:
        words, lang = transcribe_audio(work / "audio.wav", model_name, backend, language, vad=True,
                                       total=meta["duration"], label="transcribe")
        if not words:
            raise ClipperError("No speech was found in the audio.")
        log(f"transcribe: {len(words)} words, language {lang}")
        return {"language": lang, "model": model_name, "words": words}

    return cached(work / "transcript.json", params, compute, force=force, label="transcribe")


def words_in(words: list[dict], start: float, end: float) -> list[dict]:
    return [w for w in words if start <= (w["s"] + w["e"]) / 2 < end]


def clip_words(work: Path, transcript: dict, clip: dict, clip_dir: Path, model_name: str | None,
               backend: str = "auto", force: bool = False) -> list[dict]:
    """Words for a clip's captions, re-transcribed with `model_name` (e.g. turbo) for accuracy.

    Only a few minutes of audio, so even a large model is quick. With model_name None or
    "none", the words come from the main transcript.
    """
    fallback = words_in(transcript["words"], clip["start"], clip["end"])
    if not model_name or model_name == "none":
        return fallback
    backend = resolve_backend(backend)
    params = {"version": TRANSCRIBE_VERSION, "model": model_name, "backend": backend,
              "start": clip["start"], "end": clip["end"], "language": transcript.get("language")}

    def compute() -> list[dict]:
        pad = 1.0
        a = max(0.0, clip["start"] - pad)
        audio = load_audio(work / "audio.wav", a, clip["end"] + pad)
        words, _ = transcribe_audio(audio, model_name, backend, transcript.get("language"), vad=False,
                                    total=len(audio) / SAMPLE_RATE, label=f"captions #{clip['rank']}")
        shifted = [{**w, "s": round(w["s"] + a, 3), "e": round(w["e"] + a, 3)} for w in words]
        return words_in(shifted, clip["start"], clip["end"]) or fallback

    return cached(clip_dir / "words.json", params, compute, force=force,
                  label=f"captions #{clip['rank']}: words ({model_name})")
