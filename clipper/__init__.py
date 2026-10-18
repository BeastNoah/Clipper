"""clipper: turn one long video into a few short vertical clips."""

import os as _os
from pathlib import Path as _Path

__version__ = "0.1.0"


def _prefer_full_ffmpeg() -> None:
    """Put a libass-enabled ffmpeg first on PATH for every `ffmpeg`/`ffprobe` call.

    Homebrew's default `ffmpeg` has no libass (needed to burn in captions); `ffmpeg-full` does,
    but it is keg-only and therefore not on PATH. FFMPEG_DIR overrides the search.
    """
    candidates = [_os.environ.get("FFMPEG_DIR", ""),
                  "/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"]
    for d in candidates:
        if d and (_Path(d) / "ffmpeg").exists():
            parts = _os.environ.get("PATH", "").split(_os.pathsep)
            if not parts or parts[0] != d:
                _os.environ["PATH"] = _os.pathsep.join([d, *[p for p in parts if p != d]])
            return


_prefer_full_ffmpeg()
