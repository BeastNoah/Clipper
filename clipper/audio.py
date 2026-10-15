"""Loudness of the audio track: non-verbal cues for Claude, and a silence check for jump cuts."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from .transcribe import SAMPLE_RATE

FRAME = 0.05          # seconds per loudness measurement
LOUD_DB = 6.0         # a sentence this much louder than typical speech is marked [loud]
ACTIVE_DB = 12.0      # sound this far above the silence floor counts as "something is audible"
QUIET_DB = 8.0        # a stretch this close to the silence floor is safe to cut


class Loudness:
    def __init__(self, db: np.ndarray):
        self.db = db if len(db) else np.zeros(1)
        self.floor = float(np.percentile(self.db, 10))

    @classmethod
    def from_samples(cls, samples: np.ndarray, rate: int = SAMPLE_RATE) -> "Loudness":
        hop = int(rate * FRAME)
        n = len(samples) // hop
        x = samples[:n * hop].astype(np.float32).reshape(n, hop)
        return cls(10 * np.log10(np.mean(x * x, axis=1) + 1e-12))

    @classmethod
    def from_work(cls, work: Path) -> "Loudness | None":
        """Read audio.wav a minute at a time, so even a 3-hour stream needs little memory."""
        path = work / "audio.wav"
        if not path.exists():
            return None
        hop = int(SAMPLE_RATE * FRAME)
        parts = []
        with wave.open(str(path), "rb") as w:
            while True:
                raw = w.readframes(hop * 1200)
                if len(raw) < 2 * hop:
                    break
                x = np.frombuffer(raw, np.int16)[: len(raw) // 2 // hop * hop].astype(np.float32) / 32768.0
                x = x.reshape(-1, hop)
                parts.append(10 * np.log10(np.mean(x * x, axis=1) + 1e-12))
        return cls(np.concatenate(parts) if parts else np.zeros(1))

    def window(self, a: float, b: float) -> np.ndarray:
        i0 = max(0, int(a / FRAME))
        i1 = max(i0 + 1, int(math.ceil(b / FRAME)))
        return self.db[i0:i1]

    def is_quiet(self, a: float, b: float) -> bool:
        w = self.window(a, b)
        return len(w) == 0 or float(np.percentile(w, 80)) < self.floor + QUIET_DB

    def active_share(self, a: float, b: float) -> float:
        w = self.window(a, b)
        return float((w > self.floor + ACTIVE_DB).mean()) if len(w) else 0.0


def sentence_cues(loud: Loudness | None, words: list[dict], sentences: list, duration: float) -> dict[int, str]:
    """Short bracketed notes per sentence: [loud], [+1.8s non-speech sound] (often laughter or
    applause) and [pause 2.1s]."""
    if loud is None or not sentences:
        return {}
    step = max(1, len(words) // 3000)  # a sample of words is plenty to find the typical speech level
    speech = [loud.window(w["s"], w["e"]) for w in words[::step]]
    level = float(np.median(np.concatenate(speech))) if speech else loud.floor + 20
    out = {}
    for i, s in enumerate(sentences):
        notes = []
        w = loud.window(s.start, s.end)
        if len(w) and float(np.percentile(w, 90)) > level + LOUD_DB:
            notes.append("loud")
        gap_end = sentences[i + 1].start if i + 1 < len(sentences) else min(s.end + 3.0, duration)
        gap = gap_end - s.end
        if gap >= 1.0:
            if loud.active_share(s.end + 0.1, gap_end - 0.1) > 0.5:
                notes.append(f"+{gap:.1f}s non-speech sound")
            elif gap >= 1.5:
                notes.append(f"pause {gap:.1f}s")
        if notes:
            out[i] = " [" + "; ".join(notes) + "]"
    return out
