"""Step 6: build each vertical frame in Python and encode it with captions and the original audio."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from .edit import EditMap
from .util import ClipperError, Progress, fingerprint, log, read_frames, read_json, run, write_json

RENDER_VERSION = 2
OUT_W, OUT_H = 1080, 1920
COVER_AT = 1.0  # seconds into the finished clip
ENCODERS = ("x264", "videotoolbox", "auto")


def _ffmpeg_list(flag: str) -> str:
    return subprocess.run(["ffmpeg", "-hide_banner", flag], capture_output=True, text=True).stdout


def has_libass() -> bool:
    return any(line.split()[1:2] == ["ass"] for line in _ffmpeg_list("-filters").splitlines())


def has_videotoolbox() -> bool:
    return "h264_videotoolbox" in _ffmpeg_list("-encoders")


def check_ffmpeg_has_libass() -> None:
    if not has_libass():
        raise ClipperError("Your ffmpeg was built without libass, which is needed to burn in captions. "
                           "Install a full build (on macOS: brew install ffmpeg-full, or brew reinstall ffmpeg).")


def resolve_encoder(name: str) -> str:
    if name == "auto":
        return "videotoolbox" if sys.platform == "darwin" and has_videotoolbox() else "x264"
    if name == "videotoolbox" and not has_videotoolbox():
        raise ClipperError("This ffmpeg has no h264_videotoolbox encoder (it only exists on macOS)")
    return name


def encoder_args(encoder: str, crf: int, preset: str) -> list[str]:
    if encoder == "videotoolbox":  # the Mac's hardware encoder: much faster, slightly bigger files
        return ["-c:v", "h264_videotoolbox", "-b:v", "12M", "-maxrate", "16M", "-bufsize", "24M"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]


# ---------------------------------------------------------------------------------------------
# Frames

def crop_frame(frame: np.ndarray, x: int, cw: int) -> np.ndarray:
    import cv2

    return cv2.resize(frame[:, x:x + cw], (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)


def split_frame(frame: np.ndarray, v: list[int]) -> np.ndarray:
    """Two people stacked: left person on top, right person below, with a thin dark seam."""
    import cv2

    xl, yl, xr, yr, hc = v
    wc = min(frame.shape[1], int(round(hc * 9 / 8 / 2)) * 2)
    half = OUT_H // 2
    top = cv2.resize(frame[yl:yl + hc, xl:xl + wc], (OUT_W, half), interpolation=cv2.INTER_CUBIC)
    bottom = cv2.resize(frame[yr:yr + hc, xr:xr + wc], (OUT_W, half), interpolation=cv2.INTER_CUBIC)
    out = np.vstack([top, bottom])
    out[half - 3:half + 3] = 16
    return out


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


def vertical_frame(frame: np.ndarray, v, cw: int) -> np.ndarray:
    if v is None:
        return fit_frame(frame)
    if isinstance(v, list):
        return split_frame(frame, v)
    return crop_frame(frame, v, cw)


# ---------------------------------------------------------------------------------------------
# Encoding

def _audio_filter(edit: EditMap) -> list[str]:
    """Join the kept parts of the audio, with 10 ms fades so the cuts don't click."""
    parts, labels = [], []
    for i, (a, b) in enumerate(edit.segments):
        fade_out = max(0.0, b - a - 0.01)
        parts.append(f"[1:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.01,afade=t=out:st={fade_out:.4f}:d=0.01[a{i}]")
        labels.append(f"[a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]")
    return ["-filter_complex", ";".join(parts), "-map", "[aout]"]


def _encode(video: Path, meta: dict, clip: dict, plan: dict, edit: EditMap, ass: Path, out: Path,
            enc_args: list[str], cover_frame: Path) -> None:
    fps = Fraction(meta["fps"])
    start, dur = clip["start"], clip["end"] - clip["start"]
    W, H, cw = plan["width"], plan["height"], plan["crop_w"]
    xs = plan["x"]
    audio = ["-map", "1:a:0"] if edit.is_identity else _audio_filter(edit)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-framerate", str(fps), "-i", "pipe:0",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video.resolve()),
        "-map", "0:v:0", *audio,
        "-vf", f"ass={ass.name}",
        *enc_args, "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{edit.duration:.3f}",
        "-movflags", "+faststart", str(out.resolve()),
    ]
    keep = edit.frame_ranges(float(fps))
    total = sum(b - a for a, b in keep)
    cover_index = min(total - 1, round(COVER_AT * float(fps)))
    progress = Progress(f"render #{clip['rank']}", total)
    with tempfile.TemporaryFile() as err:
        # Run from the clip folder so the subtitle path needs no filter-graph escaping.
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err, cwd=ass.parent)
        frames = read_frames(video, start, dur, W, H)
        written, r = 0, 0
        try:
            for i, frame in enumerate(frames):
                while r < len(keep) and i >= keep[r][1]:
                    r += 1
                if r >= len(keep):
                    break
                if i < keep[r][0]:
                    continue  # cut out by a jump cut
                img = vertical_frame(frame, xs[min(i, len(xs) - 1)] if xs else None, cw)
                if written == cover_index:
                    import cv2
                    cv2.imwrite(str(cover_frame), img)
                enc.stdin.write(np.ascontiguousarray(img).data)
                written += 1
                progress.update(written)
            enc.stdin.close()
        except BrokenPipeError:
            pass  # the encoder stopped early; its exit code and stderr say why
        except BaseException:
            enc.kill()
            enc.wait()
            raise
        finally:
            frames.close()
            Progress.clear_line()
        if enc.wait() != 0:
            err.seek(0)
            raise ClipperError("ffmpeg failed while encoding:\n" + err.read().decode(errors="replace")[-1500:])
    if written == 0:
        raise ClipperError("No frames were decoded for this clip")


def make_cover(clip_dir: Path) -> Path:
    """cover.jpg: a frame from the start of the clip with the title on it."""
    out = clip_dir / "cover.jpg"
    run(["ffmpeg", "-y", "-v", "error", "-i", "cover_frame.png", "-vf", "ass=cover.ass",
         "-q:v", "3", out.name], cwd=clip_dir)
    return out


def render(work: Path, meta: dict, clip: dict, plan: dict, edit: EditMap, ass: Path, clip_dir: Path, *,
           encoder: str = "x264", crf: int = 20, preset: str = "medium", force: bool = False) -> Path:
    out = clip_dir / "clip.mp4"
    stamp = clip_dir / "render.json"
    encoder = resolve_encoder(encoder)
    enc_args = encoder_args(encoder, crf, preset)
    params = {
        "version": RENDER_VERSION, "start": clip["start"], "end": clip["end"], "plan": fingerprint(plan),
        "edit": edit.segments, "captions": hashlib.sha1(ass.read_bytes()).hexdigest()[:12],
        "cover": hashlib.sha1((clip_dir / "cover.ass").read_bytes()).hexdigest()[:12], "encoder": enc_args,
    }
    params = json.loads(json.dumps(params))  # tuples -> lists, to compare with the stored JSON
    if not force and out.exists() and stamp.exists() and read_json(stamp) == params:
        log(f"render #{clip['rank']}: reusing {out.parent.name}/{out.name}")
        return out
    cut = clip["end"] - clip["start"] - edit.duration
    log(f"render #{clip['rank']}: encoding {edit.duration:.0f}s at {OUT_W}x{OUT_H} ({encoder})"
        + (f", {cut:.1f}s of pauses cut" if cut > 0.05 else ""))
    tmp = clip_dir / "clip.tmp.mp4"
    _encode(work / "video.mp4", meta, clip, plan, edit, ass, tmp, enc_args, clip_dir / "cover_frame.png")
    os.replace(tmp, out)
    make_cover(clip_dir)
    write_json(stamp, params)
    return out
