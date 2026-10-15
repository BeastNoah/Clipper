"""Shared helpers: logging, subprocess calls, ffprobe, and the per-step cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    import numpy as np


class ClipperError(RuntimeError):
    """An error we can explain to the user without a traceback."""


class Cancelled(ClipperError):
    pass


# Per-thread hooks so the web UI can watch (and cancel) a pipeline running in a worker thread.
_local = threading.local()


def set_reporter(sink: Callable[[str, Any], None] | None, cancel: threading.Event | None = None) -> None:
    """sink(kind, payload) receives ("log", text), ("progress", {...}) and ("step", name) events."""
    _local.sink, _local.cancel = sink, cancel


def report(kind: str, payload: Any) -> None:
    sink = getattr(_local, "sink", None)
    if sink is not None:
        sink(kind, payload)


def check_cancel() -> None:
    ev = getattr(_local, "cancel", None)
    if ev is not None and ev.is_set():
        raise Cancelled("Cancelled")


def log(msg: str) -> None:
    Progress.clear_line()
    print(f"[clipper] {msg}", file=sys.stderr, flush=True)
    report("log", msg)


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"


class Progress:
    """A one-line progress display with a time estimate (and a cancellation point for the UI)."""

    _on_line = False

    def __init__(self, label: str, total: float):
        self.label, self.total = label, max(float(total), 1e-9)
        self.t0 = self.last = time.monotonic()
        self.tty = sys.stderr.isatty()

    def update(self, done: float) -> None:
        check_cancel()
        now = time.monotonic()
        if now - self.last < (0.3 if self.tty else 15):
            return
        self.last = now
        frac = min(max(done / self.total, 0.0), 1.0)
        eta = (now - self.t0) * (1 - frac) / frac if frac > 0.02 else None
        text = f"{self.label}: {frac:4.0%}" + (f", about {fmt_duration(eta)} left" if eta is not None else "")
        report("progress", {"label": self.label, "fraction": round(frac, 4), "eta": eta})
        if self.tty:
            print(f"\r[clipper] {text}   ", end="", file=sys.stderr, flush=True)
            Progress._on_line = True
        else:
            print(f"[clipper] {text}", file=sys.stderr, flush=True)

    @classmethod
    def clear_line(cls) -> None:
        if cls._on_line:
            print("\r\033[K", end="", file=sys.stderr, flush=True)
            cls._on_line = False


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ClipperError(f"'{name}' was not found on PATH. Install it first (see README).")
    return path


def run(cmd: list[str], *, cwd: Path | None = None, input: bytes | None = None) -> subprocess.CompletedProcess:
    """Run a command, raising ClipperError with the tail of stderr on failure."""
    proc = subprocess.run(cmd, cwd=cwd, input=input, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-15:]
        raise ClipperError(f"{Path(cmd[0]).name} failed (exit {proc.returncode}):\n" + "\n".join(tail))
    return proc


def ffprobe(path: Path) -> dict:
    out = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]).stdout
    return json.loads(out)


def video_stream(info: dict) -> dict | None:
    return next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)


def audio_stream(info: dict) -> dict | None:
    return next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)


def parse_rate(rate: str | None) -> Fraction | None:
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        r = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return None
    return r if r > 0 else None


def read_frames(video: Path, start: float, duration: float, width: int, height: int,
                scale: bool = False) -> Iterator["np.ndarray"]:
    """Yield BGR frames of [start, start + duration) as numpy arrays, decoded by ffmpeg.

    Every caller uses the same seek arguments, so the analysis and the render see the same frames.
    """
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", str(video),
           "-t", f"{duration:.3f}", "-an", "-sn"]
    if scale:
        cmd += ["-vf", f"scale={width}:{height}:flags=area"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = width * height * 3
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            code = proc.wait()
        if code != 0:
            err.seek(0)
            raise ClipperError("ffmpeg could not decode the video:\n" + err.read().decode(errors="replace")[-1500:])


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically so an interrupted run never leaves a half-written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def fingerprint(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def cached(path: Path, params: dict, compute: Callable[[], Any], *, force: bool = False, label: str = "") -> Any:
    """Return the JSON result stored at `path` if it was made with the same params, else compute it.

    The params are stored next to the result, so changing an option that affects a step
    reruns that step (and only that step).
    """
    if not force and path.exists():
        try:
            stored = read_json(path)
            if stored.get("params") == params:
                if label:
                    log(f"{label}: reusing {path.name}")
                return stored["result"]
        except (json.JSONDecodeError, KeyError):
            pass
    t0 = time.monotonic()
    result = compute()
    write_json(path, {"params": params, "result": result})
    if label:
        log(f"{label}: done in {time.monotonic() - t0:.1f}s")
    return result


def fmt_time(seconds: float) -> str:
    m, s = divmod(max(seconds, 0.0), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def slugify(text: str, max_len: int = 40) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")[:max_len].strip("-") or "clip"
