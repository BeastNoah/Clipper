"""Step 6: build each output frame in Python and encode it with captions and the original audio."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .edit import EditMap
from .formats import VERTICAL, Format
from .util import ClipperError, Progress, fingerprint, log, read_frames, read_json, run, write_json

if TYPE_CHECKING:
    from .ingest import ClipVideo

RENDER_VERSION = 3
COVER_AT = 1.0  # seconds into the finished clip
ENCODERS = ("x264", "videotoolbox", "auto")
ZOOM = 1.12                    # how far an auto-zoom punches in
ZOOM_IN, ZOOM_OUT = 0.18, 0.35  # seconds to ease in and out
LOGO_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


def _ffmpeg_list(flag: str) -> str:
    return subprocess.run(["ffmpeg", "-hide_banner", flag], capture_output=True, text=True).stdout


def has_libass() -> bool:
    return any(line.split()[1:2] == ["ass"] for line in _ffmpeg_list("-filters").splitlines())


def has_videotoolbox() -> bool:
    return "h264_videotoolbox" in _ffmpeg_list("-encoders")


def check_ffmpeg_has_libass() -> None:
    if not has_libass():
        raise ClipperError("Your ffmpeg was built without libass, which is needed to burn in captions. "
                           "Install a full build (on macOS: brew install ffmpeg-full, or set FFMPEG_DIR to a folder with a libass-enabled ffmpeg).")


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


@dataclass
class Piece:
    """One stretch of the video in a clip: start-end on the transcript's timeline, cut from `src`, with its
    framing plan and jump cuts. Most clips are one piece; a clip with earlier parts plays several in a row."""
    src: "ClipVideo"
    start: float
    end: float
    plan: dict
    edit: EditMap

    def key(self) -> list:
        return [self.start, self.end, fingerprint(self.plan), self.edit.segments, self.src.key()]


# ---------------------------------------------------------------------------------------------
# Extras drawn on the frames

def _file_key(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


@dataclass
class Look:
    zoom: list[tuple[float, float]] = field(default_factory=list)  # output-time windows to punch in on
    logo: Path | None = None
    logo_position: str = "top-right"
    opacity: int = 90            # percent, for the logo
    progress: str | None = None  # colour of a progress bar along the top, e.g. "#FFE600"
    broll: list[tuple[float, float, Path]] = field(default_factory=list)  # output-time windows showing a picture or video
    fade_between: float = 0.0    # seconds of a dip to black at each join between the clip's parts
    fade_in: float = 0.0
    fade_out: float = 0.0
    music: tuple[Path, int] | None = None  # a music file and its volume in percent, mixed under the sound

    def key(self) -> dict:
        logo = _file_key(self.logo) if self.logo else None
        key = {"zoom": [[round(a, 2), round(b, 2)] for a, b in self.zoom], "logo": logo,
               "logo_position": self.logo_position, "progress": self.progress}
        if self.opacity != 90:  # only when changed, so renders from before the setting existed still count
            key["opacity"] = self.opacity
        if self.broll:
            key["broll"] = [[round(a, 2), round(b, 2), _file_key(p)] for a, b, p in self.broll]
        if self.fade_between or self.fade_in or self.fade_out:
            key["fades"] = [self.fade_between, self.fade_in, self.fade_out]
        if self.music:
            key["music"] = [_file_key(self.music[0]), self.music[1]]
        return key


IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def asset_frames(path: Path, duration: float, fmt: Format, fps: Fraction):
    """Frames of a picture (held) or a video (looped) for `duration` seconds, letterboxed to the output size."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if path.suffix.lower() in IMAGE_TYPES:
        cmd += ["-loop", "1", "-framerate", str(fps), "-i", str(path)]
    else:
        cmd += ["-stream_loop", "-1", "-i", str(path)]
    cmd += ["-t", f"{duration:.3f}", "-an", "-sn",
            "-vf", f"fps={fps},scale={fmt.width}:{fmt.height}:force_original_aspect_ratio=decrease,"
                   f"pad={fmt.width}:{fmt.height}:(ow-iw)/2:(oh-ih)/2:color=black,format=bgr24",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = fmt.width * fmt.height * 3
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(fmt.height, fmt.width, 3)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            code = proc.wait()
        if code != 0:
            err.seek(0)
            raise ClipperError(f"ffmpeg could not read {path.name}:\n" + err.read().decode(errors="replace")[-1500:])


class BRoll:
    """Pictures or videos shown over stretches of the finished clip (the captions still go on top)."""

    def __init__(self, windows: list[tuple[float, float, Path]], fmt: Format, fps: Fraction):
        self.windows = sorted(windows)
        self.fmt, self.fps = fmt, fps
        self.current: tuple | None = None  # (index, frames, last frame)

    def frame(self, t: float) -> "np.ndarray | None":
        for k, (a, b, path) in enumerate(self.windows):
            if a <= t < b:
                if self.current is None or self.current[0] != k:
                    self.close()
                    self.current = [k, asset_frames(path, b - a, self.fmt, self.fps), None]
                try:
                    self.current[2] = next(self.current[1])
                except StopIteration:
                    pass  # shorter than asked: hold the last frame
                return self.current[2]
        self.close()
        return None

    def close(self) -> None:
        if self.current is not None:
            self.current[1].close()
            self.current = None


def fade_factor(t: float, duration: float, joins: list[float], look: Look) -> float:
    """How bright the frame at output time t is: dips to black at the joins and at the ends, if asked."""
    f = 1.0
    if look.fade_in > 0:
        f = min(f, t / look.fade_in)
    if look.fade_out > 0:
        f = min(f, (duration - t) / look.fade_out)
    if look.fade_between > 0:
        for j in joins:
            f = min(f, abs(t - j) / look.fade_between)
    return min(max(f, 0.0), 1.0)


def read_logo(path: Path) -> np.ndarray:
    """The logo as 8-bit BGRA."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ClipperError(f"Couldn't read the logo {path}")
    if img.dtype == np.uint16:  # 16-bit PNGs: scale down to 8 bits before blending
        img = (img >> 8).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def logo_size(shape: tuple[int, ...], fmt: Format) -> tuple[int, int]:
    """Width and height of a logo of this (h, w) shape on the frame."""
    w = int(fmt.width * (0.14 if fmt.ratio < 1 else 0.09))
    h = max(1, int(shape[0] * w / shape[1]))
    max_h = int(fmt.height * 0.2)  # tall logos are sized by height so they always fit the frame
    if h > max_h:
        h, w = max_h, max(1, int(shape[1] * max_h / shape[0]))
    return w, h


def _ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def zoom_at(t: float, windows: list[tuple[float, float]]) -> float:
    z = 1.0
    for a, b in windows:
        if t < a or t > b + ZOOM_OUT:
            continue
        f = _ease((t - a) / ZOOM_IN) if t < a + ZOOM_IN else 1.0 if t <= b else 1 - _ease((t - b) / ZOOM_OUT)
        z = max(z, 1 + (ZOOM - 1) * f)
    return z


class Overlay:
    """Logo and progress bar, drawn onto finished frames."""

    def __init__(self, look: Look, fmt: Format):
        import cv2

        self.look, self.fmt, self.logo = look, fmt, None
        if look.logo:
            img = read_logo(look.logo)
            w, h = logo_size(img.shape, fmt)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            margin, top, bottom = fmt.corner_margins()
            x = margin if "left" in look.logo_position else fmt.width - w - margin
            y = top if "top" in look.logo_position else fmt.height - h - bottom
            x, y = min(max(0, x), fmt.width - w), min(max(0, y), fmt.height - h)
            self.logo, self.pos = img, (x, y)
            self.alpha = img[:, :, 3:4].astype(np.float32) / 255.0 * (look.opacity / 100)
        if look.progress:
            hx = look.progress.lstrip("#")
            self.bar_color = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
            self.bar_h = max(6, int(fmt.height * 0.006))

    def apply(self, img: np.ndarray, fraction: float) -> np.ndarray:
        if self.logo is not None:
            x, y = self.pos
            h, w = self.logo.shape[:2]
            region = img[y:y + h, x:x + w].astype(np.float32)
            img[y:y + h, x:x + w] = (region * (1 - self.alpha) + self.logo[:, :, :3] * self.alpha).astype(np.uint8)
        if self.look.progress:
            img[: self.bar_h, : int(self.fmt.width * min(1.0, fraction))] = self.bar_color
        return img


# ---------------------------------------------------------------------------------------------
# Frames

def crop_frame(frame: np.ndarray, x: int, cw: int, fmt: Format = VERTICAL, zoom: float = 1.0) -> np.ndarray:
    import cv2

    if zoom > 1.001:  # a smaller window around the same centre, a little above the middle (faces)
        H, W = frame.shape[:2]
        w2, h2 = int(round(cw / zoom)), int(round(H / zoom))
        x0 = int(min(max(x + cw / 2 - w2 / 2, 0), W - w2))
        y0 = int(min(max(H * 0.42 - h2 / 2, 0), H - h2))
        region = frame[y0:y0 + h2, x0:x0 + w2]
    else:
        region = frame[:, x:x + cw]
    return cv2.resize(region, (fmt.width, fmt.height), interpolation=cv2.INTER_CUBIC)


def split_frame(frame: np.ndarray, v: list[int], fmt: Format = VERTICAL) -> np.ndarray:
    """Two people stacked: left person on top, right person below, with a thin dark seam."""
    import cv2

    xl, yl, xr, yr, hc = v
    half = fmt.height // 2
    wc = min(frame.shape[1], int(round(hc * fmt.width / half / 2)) * 2)
    top = cv2.resize(frame[yl:yl + hc, xl:xl + wc], (fmt.width, half), interpolation=cv2.INTER_CUBIC)
    bottom = cv2.resize(frame[yr:yr + hc, xr:xr + wc], (fmt.width, fmt.height - half), interpolation=cv2.INTER_CUBIC)
    out = np.vstack([top, bottom])
    out[half - 3:half + 3] = 16
    return out


def fit_frame(frame: np.ndarray, fmt: Format = VERTICAL) -> np.ndarray:
    """The whole frame, scaled to fit, over a blurred, darkened copy of itself."""
    import cv2

    OW, OH = fmt.width, fmt.height
    h, w = frame.shape[:2]
    if w / h > OW / OH:
        bw = int(h * OW / OH)
        src = frame[:, (w - bw) // 2:(w - bw) // 2 + bw]
    else:
        bh = int(w * OH / OW)
        src = frame[(h - bh) // 2:(h - bh) // 2 + bh]
    small = cv2.resize(src, (OW // 4, OH // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 10)
    out = cv2.convertScaleAbs(cv2.resize(small, (OW, OH), interpolation=cv2.INTER_LINEAR), alpha=0.55)

    s = min(OW / w, OH / h)
    fw, fh = min(OW, int(round(w * s))), min(OH, int(round(h * s)))
    fg = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0, x0 = (OH - fh) // 2, (OW - fw) // 2
    out[y0:y0 + fh, x0:x0 + fw] = fg
    return out


def vertical_frame(frame: np.ndarray, v, cw: int, fmt: Format = VERTICAL, zoom: float = 1.0) -> np.ndarray:
    if v is None:
        return fit_frame(frame, fmt)
    if isinstance(v, list):
        return split_frame(frame, v, fmt)
    return crop_frame(frame, v, cw, fmt, zoom)


# ---------------------------------------------------------------------------------------------
# Encoding

def _audio_filter(pieces: list[Piece], music: tuple[Path, int] | None = None, duration: float = 0.0) -> list[str]:
    """Join the kept parts of the audio (input n + 1 is piece n), with 10 ms fades so the cuts don't click.
    With music (the input after the pieces), it is mixed in underneath at its volume and faded out at the end."""
    parts, labels = [], []
    for n, piece in enumerate(pieces):
        for i, (a, b) in enumerate(piece.edit.segments):
            fade_out = max(0.0, b - a - 0.01)
            parts.append(f"[{n + 1}:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS,"
                         f"afade=t=in:d=0.01,afade=t=out:st={fade_out:.4f}:d=0.01[a{n}_{i}]")
            labels.append(f"[a{n}_{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]")
    out = "[aout]"
    if music:
        fade = min(1.5, duration / 2)
        parts.append(f"[{len(pieces) + 1}:a]atrim=end={duration:.3f},asetpts=PTS-STARTPTS,"
                     f"volume={music[1] / 100:.3f},afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f}[mus]")
        parts.append("[aout][mus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[amix]")
        out = "[amix]"
    return ["-filter_complex", ";".join(parts), "-map", out]


def _encode(meta: dict, clip: dict, pieces: list[Piece], ass: Path, out: Path, enc_args: list[str],
            cover_frame: Path, fmt: Format, look: Look) -> None:
    fps = Fraction(meta["fps"])
    duration = sum(p.edit.duration for p in pieces)
    inputs = []
    for p in pieces:  # each piece's stretch, in its video file's own time
        inputs += ["-ss", f"{p.start - p.src.offset:.3f}", "-t", f"{p.end - p.start:.3f}", "-i", str(p.src.path.resolve())]
    if look.music:
        inputs += ["-stream_loop", "-1", "-i", str(look.music[0].resolve())]
    single = len(pieces) == 1 and pieces[0].edit.is_identity and not look.music
    audio = ["-map", "1:a:0"] if single else _audio_filter(pieces, look.music, duration)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{fmt.width}x{fmt.height}", "-framerate", str(fps),
        "-i", "pipe:0", *inputs,
        "-map", "0:v:0", *audio,
        "-vf", f"ass={ass.name}",
        *enc_args, "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.3f}",
        "-movflags", "+faststart", str(out.resolve()),
    ]
    total = sum(b - a for p in pieces for a, b in p.edit.frame_ranges(float(fps)))
    cover_index = min(total - 1, round(COVER_AT * float(fps)))
    overlay = Overlay(look, fmt)
    broll = BRoll(look.broll, fmt, fps) if look.broll else None
    joins, at = [], 0.0  # where one part of the clip ends and the next begins, in output time
    for p in pieces[:-1]:
        at += p.edit.duration
        joins.append(at)
    fades = look.fade_between > 0 or look.fade_in > 0 or look.fade_out > 0
    progress = Progress(f"render #{clip['rank']} ({fmt.name})", total)
    with tempfile.TemporaryFile() as err:
        # Run from the clip folder so the subtitle path needs no filter-graph escaping.
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err, cwd=ass.parent)
        written = 0
        try:
            for p in pieces:
                keep, xs, cw = p.edit.frame_ranges(float(fps)), p.plan["x"], p.plan["crop_w"]
                frames = read_frames(p.src.path, p.start - p.src.offset, p.end - p.start,
                                     p.plan["width"], p.plan["height"])
                r = 0
                try:
                    for i, frame in enumerate(frames):
                        while r < len(keep) and i >= keep[r][1]:
                            r += 1
                        if r >= len(keep):
                            break
                        if i < keep[r][0]:
                            continue  # cut out by a jump cut
                        t_out = written / float(fps)
                        zoom = zoom_at(t_out, look.zoom) if look.zoom else 1.0
                        img = vertical_frame(frame, xs[min(i, len(xs) - 1)] if xs else None, cw, fmt, zoom)
                        over = broll.frame(t_out) if broll else None
                        if over is not None:
                            img = over.copy()
                        if fades:
                            f = fade_factor(t_out, duration, joins, look)
                            if f < 0.999:
                                import cv2
                                img = cv2.convertScaleAbs(img, alpha=f)
                        if written == cover_index:
                            import cv2
                            cv2.imwrite(str(cover_frame), img)  # before the logo and bar: the cover stays clean
                        img = overlay.apply(img, (written + 1) / max(total, 1))
                        enc.stdin.write(np.ascontiguousarray(img).data)
                        written += 1
                        progress.update(written)
                finally:
                    frames.close()
            enc.stdin.close()
        except BrokenPipeError:
            pass  # the encoder stopped early; its exit code and stderr say why
        except BaseException:
            enc.kill()
            enc.wait()
            raise
        finally:
            if broll:
                broll.close()
            Progress.clear_line()
        if enc.wait() != 0:
            err.seek(0)
            raise ClipperError("ffmpeg failed while encoding:\n" + err.read().decode(errors="replace")[-1500:])
    if written == 0:
        raise ClipperError("No frames were decoded for this clip")


def make_cover(out_dir: Path) -> Path:
    """cover.jpg: a frame from the start of the clip with the title on it."""
    out = out_dir / "cover.jpg"
    run(["ffmpeg", "-y", "-v", "error", "-i", "cover_frame.png", "-vf", "ass=cover.ass",
         "-q:v", "3", out.name], cwd=out_dir)
    return out


def render(work: Path, meta: dict, clip: dict, pieces: list[Piece], ass: Path, out_dir: Path, *,
           fmt: Format = VERTICAL, look: Look | None = None, encoder: str = "x264", crf: int = 20,
           preset: str = "medium", force: bool = False) -> Path:
    """Render the clip's pieces, one after the other, into out_dir/clip.mp4 (and its cover)."""
    look = look or Look()
    out = out_dir / "clip.mp4"
    stamp = out_dir / "render.json"
    encoder = resolve_encoder(encoder)
    enc_args = encoder_args(encoder, crf, preset)
    params = {
        "version": RENDER_VERSION, "captions": hashlib.sha1(ass.read_bytes()).hexdigest()[:12],
        "cover": hashlib.sha1((out_dir / "cover.ass").read_bytes()).hexdigest()[:12], "encoder": enc_args,
        "format": fmt.name, "look": look.key(),
    }
    if len(pieces) == 1:  # the same stamp as before clips could have earlier parts, so those renders still count
        p = pieces[0]
        params.update({"start": p.start, "end": p.end, "plan": fingerprint(p.plan), "edit": p.edit.segments,
                       "video": p.src.key()})
    else:
        params["pieces"] = [p.key() for p in pieces]
    params = json.loads(json.dumps(params))  # tuples -> lists, to compare with the stored JSON
    if not force and out.exists() and stamp.exists() and read_json(stamp) == params:
        log(f"render #{clip['rank']} ({fmt.name}): reusing {out.parent.parent.name}/{out.parent.name}/{out.name}")
        return out
    duration = sum(p.edit.duration for p in pieces)
    cut = sum(p.end - p.start for p in pieces) - duration
    log(f"render #{clip['rank']}: encoding {duration:.0f}s at {fmt.width}x{fmt.height} ({encoder})"
        + (f" from {len(pieces)} parts of the video" if len(pieces) > 1 else "")
        + (f", {cut:.1f}s of pauses cut" if cut > 0.05 else ""))
    tmp = out_dir / "clip.tmp.mp4"
    _encode(meta, clip, pieces, ass, tmp, enc_args, out_dir / "cover_frame.png", fmt, look)
    os.replace(tmp, out)
    make_cover(out_dir)
    write_json(stamp, params)
    return out
