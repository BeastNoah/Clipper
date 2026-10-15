"""Step 6: build each vertical frame in Python and encode it with captions and the original audio."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from .util import ClipperError, fingerprint, log, read_frames, read_json, write_json

RENDER_VERSION = 1
OUT_W, OUT_H = 1080, 1920


def check_ffmpeg_has_libass() -> None:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if not any(line.split()[1:2] == ["ass"] for line in out.splitlines()):
        raise ClipperError("Your ffmpeg was built without libass, which is needed to burn in captions. "
                           "Install a full build (on macOS: brew install ffmpeg-full, or brew reinstall ffmpeg).")


def crop_frame(frame: np.ndarray, x: int, cw: int) -> np.ndarray:
    import cv2

    return cv2.resize(frame[:, x:x + cw], (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)


def fit_frame(frame: np.ndarray) -> np.ndarray:
    """The whole frame, scaled to the output width, over a blurred, darkened copy of itself."""
    import cv2

    h, w = frame.shape[:2]
    if w / h > OUT_W / OUT_H:
        bw = int(h * OUT_W / OUT_H)
        src = frame[:, (w - bw) // 2:(w - bw) // 2 + bw]
    else:
        bh = int(w * OUT_H / OUT_W)
        src = frame[(h - bh) // 2:(h - bh) // 2 + bh]
    small = cv2.resize(src, (OUT_W // 4, OUT_H // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 10)
    out = cv2.convertScaleAbs(cv2.resize(small, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR), alpha=0.55)

    s = min(OUT_W / w, OUT_H / h)
    fw, fh = min(OUT_W, int(round(w * s))), min(OUT_H, int(round(h * s)))
    fg = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0, x0 = (OUT_H - fh) // 2, (OUT_W - fw) // 2
    out[y0:y0 + fh, x0:x0 + fw] = fg
    return out


def _encode(video: Path, meta: dict, clip: dict, plan: dict, ass: Path, out: Path, crf: int, preset: str) -> None:
    fps = Fraction(meta["fps"])
    start, dur = clip["start"], clip["end"] - clip["start"]
    W, H, cw = plan["width"], plan["height"], plan["crop_w"]
    xs = plan["x"]
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-framerate", str(fps), "-i", "pipe:0",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video.resolve()),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", f"ass={ass.name}",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}",
        "-movflags", "+faststart", str(out.resolve()),
    ]
    with tempfile.TemporaryFile() as err:
        # Run from the clip folder so the subtitle path needs no filter-graph escaping.
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err, cwd=ass.parent)
        frames = read_frames(video, start, dur, W, H)
        n = 0
        try:
            for n, frame in enumerate(frames, 1):
                x = xs[min(n - 1, len(xs) - 1)] if xs else None
                img = fit_frame(frame) if x is None else crop_frame(frame, x, cw)
                enc.stdin.write(np.ascontiguousarray(img).data)
            enc.stdin.close()
        except BrokenPipeError:
            pass  # the encoder stopped early; its exit code and stderr say why
        except BaseException:
            enc.kill()
            enc.wait()
            raise
        finally:
            frames.close()
        if enc.wait() != 0:
            err.seek(0)
            raise ClipperError("ffmpeg failed while encoding:\n" + err.read().decode(errors="replace")[-1500:])
    if n == 0:
        raise ClipperError("No frames were decoded for this clip")


def render(work: Path, meta: dict, clip: dict, plan: dict, ass: Path, clip_dir: Path, *,
           crf: int = 20, preset: str = "medium", force: bool = False) -> Path:
    out = clip_dir / "clip.mp4"
    stamp = clip_dir / "render.json"
    params = {
        "version": RENDER_VERSION, "start": clip["start"], "end": clip["end"], "plan": fingerprint(plan),
        "captions": hashlib.sha1(ass.read_bytes()).hexdigest()[:12], "crf": crf, "preset": preset,
    }
    if not force and out.exists() and stamp.exists() and read_json(stamp) == params:
        log(f"render #{clip['rank']}: reusing {out.parent.name}/{out.name}")
        return out
    log(f"render #{clip['rank']}: encoding {clip['end'] - clip['start']:.0f}s at {OUT_W}x{OUT_H}")
    tmp = clip_dir / "clip.tmp.mp4"
    _encode(work / "video.mp4", meta, clip, plan, ass, tmp, crf, preset)
    os.replace(tmp, out)
    write_json(stamp, params)
    return out
