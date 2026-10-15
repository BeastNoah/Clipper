"""`clipper doctor`: check that everything clipper needs is installed and working."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from .captions import PRESETS, build_cover_ass, make_style


class Report:
    def __init__(self):
        self.failed = 0

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  ok    {label}" + (f": {detail}" if detail else ""))

    def warn(self, label: str, detail: str) -> None:
        print(f"  note  {label}: {detail}")

    def fail(self, label: str, detail: str) -> None:
        self.failed += 1
        print(f"  FIX   {label}: {detail}")


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def font_in_use(font: str) -> tuple[str, str] | None:
    """Ask libass which font file it really uses for `font`: (file, font name), or None if unknown."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "t.ass").write_text(build_cover_ass("Test", make_style("bold", font=font)), encoding="utf-8")
        proc = subprocess.run(["ffmpeg", "-v", "verbose", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.1",
                               "-vf", "ass=t.ass", "-frames:v", "1", "-f", "null", "-"],
                              cwd=d, capture_output=True, text=True)
    m = re.search(r"fontselect: \(.*?\) -> (.+?), \d+, (\S+)", proc.stderr)
    return (m[1], m[2]) if m else None


def _same_font(wanted: str, found: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())  # noqa: E731
    return norm(wanted) in norm(found)


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=8):
            return True
    except Exception:  # noqa: BLE001
        return False


def doctor(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="clipper doctor")
    p.add_argument("--no-claude", action="store_true", help="skip the test request to Claude")
    p.add_argument("--model", default="haiku", help="model for the test request (default haiku)")
    args = p.parse_args(argv)
    r = Report()
    mac_arm = sys.platform == "darwin" and platform.machine() == "arm64"

    print("System")
    if sys.version_info >= (3, 10):
        r.ok("Python", platform.python_version())
    else:
        r.fail("Python", f"{platform.python_version()} is too old; clipper needs 3.10 or newer")

    print("ffmpeg")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        r.fail("ffmpeg", "not found. Install it (macOS: brew install ffmpeg)")
    else:
        first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
        r.ok("ffmpeg", first.replace("ffmpeg version ", "").split(" Copyright")[0])
        from .render import has_libass, has_videotoolbox
        if has_libass():
            r.ok("libass (captions)")
            for preset in sorted({s.font for s in PRESETS.values()}):
                found = font_in_use(preset)
                if found is None:
                    r.warn(f"font {preset}", "couldn't tell which font libass picks")
                elif _same_font(preset, found[1]) or _same_font(preset, Path(found[0]).stem):
                    r.ok(f"font {preset}", found[0])
                else:
                    r.warn(f"font {preset}", f"not installed; captions would use {found[1]} instead. "
                           "Install the font or choose another with --font")
        else:
            r.fail("libass (captions)", "this ffmpeg can't burn in subtitles. "
                   "macOS: brew install ffmpeg-full (or a build with --enable-libass)")
        if sys.platform == "darwin":
            if has_videotoolbox():
                r.ok("hardware encoder", "h264_videotoolbox (use --encoder videotoolbox)")
            else:
                r.warn("hardware encoder", "h264_videotoolbox not available in this ffmpeg")

    print("Claude")
    claude = shutil.which("claude")
    if not claude:
        r.fail("claude CLI", "not found. Install Claude Code (https://claude.com/claude-code) and log in")
    elif args.no_claude:
        r.ok("claude CLI", claude)
    else:
        try:
            env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_SESSION_ID"}
            proc = subprocess.run([claude, "-p", "--output-format", "json", "--model", args.model,
                                   "--tools", "", "--no-session-persistence"],
                                  input=b"Reply with just: OK", capture_output=True, timeout=120, env=env)
            data = json.loads(proc.stdout or b"{}")
            if data.get("is_error") or proc.returncode != 0:
                r.fail("claude CLI", f"test request failed: {str(data.get('result') or proc.stderr[-300:])}. "
                       "Run `claude` once to log in")
            else:
                r.ok("claude CLI", f"logged in, test request answered ({args.model})")
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            r.fail("claude CLI", f"test request failed ({e}). Run `claude` once to log in")

    print("Speech recognition")
    if _version("faster-whisper"):
        r.ok("faster-whisper", _version("faster-whisper"))
        from faster_whisper.utils import download_model
        missing = []
        for name in ("small", "turbo"):
            try:
                download_model(name, local_files_only=True)
            except Exception:  # noqa: BLE001
                missing.append(name)
        if missing:
            where = "huggingface.co is reachable" if _reachable("https://huggingface.co") else \
                "but huggingface.co is NOT reachable from here"
            (r.warn if "NOT" not in where else r.fail)(
                "whisper models", f"{', '.join(missing)} will download on first use ({where})")
        else:
            r.ok("whisper models", "small and turbo are downloaded")
    else:
        r.fail("faster-whisper", "not installed: pip install faster-whisper")
    if mac_arm:
        if importlib.util.find_spec("mlx_whisper"):
            r.ok("mlx-whisper", f"{_version('mlx-whisper')} (Apple Silicon GPU, used automatically)")
        else:
            r.warn("mlx-whisper", "not installed; transcription would be several times faster with "
                   "pip install mlx-whisper")

    print("Faces")
    cv = _version("opencv-python-headless") or _version("opencv-python")
    if cv:
        r.ok("OpenCV", cv)
        from .reframe import FaceDetector
        model = FaceDetector._yunet_model()
        if model:
            r.ok("YuNet face model", str(model))
        else:
            import cv2
            if hasattr(cv2, "CascadeClassifier"):
                r.warn("YuNet face model", "couldn't download it; the less accurate Haar detector will be used")
            else:
                r.fail("face detection", "no YuNet model and no Haar fallback (OpenCV 5); "
                       "check your internet connection or install opencv-python-headless<5")
    else:
        r.fail("OpenCV", "not installed: pip install 'opencv-python-headless<5'")

    print("Downloads")
    if _version("yt-dlp"):
        r.ok("yt-dlp", _version("yt-dlp"))
        runtime = next((t for t in ("deno", "node", "bun") if shutil.which(t)), None)
        if runtime:
            r.ok("JavaScript runtime for YouTube", runtime)
        else:
            r.warn("JavaScript runtime for YouTube", "none found; YouTube downloads need one "
                   "(macOS: brew install deno)")
    else:
        r.fail("yt-dlp", "not installed: pip install 'yt-dlp[default]'")

    print()
    print("Everything looks good." if not r.failed else f"{r.failed} thing(s) to fix (marked FIX above).")
    return 1 if r.failed else 0
