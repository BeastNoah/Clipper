"""Step 1: get the video, convert it to a standard constant-frame-rate MP4 and pull out 16 kHz mono audio."""

from __future__ import annotations

import hashlib
import os
import re
from fractions import Fraction
from pathlib import Path

from .util import (
    ClipperError, ascii_slug, audio_stream, check_cancel, ffprobe, log, parse_rate, read_json, run,
    video_stream, write_json,
)

MAX_HEIGHT = 1080
# Frame rates we normalise to; anything else (including variable rate phone video) snaps to the closest.
STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                  Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60)]
INGEST_VERSION = 1


def is_url(source: str) -> bool:
    return re.match(r"^https?://", source, re.I) is not None


def local_video_id(path: Path) -> str:
    """Name plus a hash of size and the first/last MB: stable across runs, distinct for different files."""
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read())
    return f"{ascii_slug(path.stem, 32)}-{h.hexdigest()[:8]}"


def pick_frame_rate(stream: dict) -> Fraction:
    rate = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate")) or Fraction(30)
    rate = min(rate, Fraction(60))
    return min(STANDARD_RATES, key=lambda r: abs(float(r) - float(rate)) / float(r))


def _yt_dlp():
    try:
        import yt_dlp
    except ImportError as e:
        raise ClipperError("yt-dlp is needed to download URLs: pip install yt-dlp") from e
    return yt_dlp


def expand_sources(sources: list[str]) -> list[str]:
    """Replace playlist and channel URLs by the URLs of their videos."""
    out = []
    for src in sources:
        if not is_url(src):
            out.append(src)
            continue
        yt_dlp = _yt_dlp()
        # noplaylist: a watch?v=…&list=… link means that one video, not the whole playlist or Mix
        opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(src, download=False)
        if info.get("_type") == "playlist":
            urls = [e.get("url") or e.get("webpage_url") for e in info.get("entries") or []]
            urls = [u for u in urls if u]
            log(f"ingest: playlist '{info.get('title') or src}' has {len(urls)} videos")
            out += urls
        else:
            out.append(src)
    return out


def url_video_id(url: str) -> tuple[str, dict]:
    yt_dlp = _yt_dlp()
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") == "playlist":
        raise ClipperError("That URL is a playlist; pass a single video URL.")
    vid = f"{ascii_slug(info.get('extractor_key') or 'web', 12)}-{ascii_slug(str(info['id']), 40)}"
    return vid, info


def _finished_downloads(dest_dir: Path) -> list[Path]:
    # Skip yt-dlp's partial files and the per-format pieces (source.f137.mp4) it merges afterwards.
    return sorted(
        p for p in dest_dir.glob("source.*")
        if not re.search(r"\.(part|ytdl|temp)$|^source\.(f\d+|temp)\.", p.name)
    )


def download(url: str, dest_dir: Path) -> Path:
    existing = _finished_downloads(dest_dir)
    if existing:
        log(f"ingest: reusing download {existing[0].name}")
        return existing[0]
    yt_dlp = _yt_dlp()
    log(f"ingest: downloading {url}")
    opts = {
        "format": f"bv*[height<={MAX_HEIGHT}]+ba/b[height<={MAX_HEIGHT}]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [lambda _: check_cancel()],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if path.suffix != ".mp4" and path.with_suffix(".mp4").exists():
        path = path.with_suffix(".mp4")
    if not path.exists():
        found = _finished_downloads(dest_dir)
        if not found:
            raise ClipperError("yt-dlp finished but no file was written")
        path = found[0]
    return path


def normalize(src: Path, dest: Path, rate: Fraction) -> None:
    """Re-encode to H.264/AAC at a constant frame rate so audio and picture stay in sync when we cut."""
    gop = max(1, round(float(rate) * 2))  # keyframe every ~2s keeps later seeks fast
    vf = (
        "scale='trunc(iw*sar/2)*2':'trunc(ih/2)*2',setsar=1,"
        f"scale=-2:'min({MAX_HEIGHT},ih)':flags=lanczos,"
        f"fps={rate.numerator}/{rate.denominator},format=yuv420p"
    )
    tmp = dest.with_name("video.tmp.mp4")
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0",
        "-vf", vf, "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", str(gop),
        "-af", "aresample=async=1:first_pts=0", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(tmp),
    ])
    os.replace(tmp, dest)


def extract_audio(video: Path, dest: Path) -> None:
    tmp = dest.with_name("audio.tmp.wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(tmp)])
    os.replace(tmp, dest)


def ingest(source: str, work_root: Path, force: bool = False) -> tuple[Path, dict]:
    """Returns (work dir for this video, meta)."""
    if is_url(source):
        video_id, info = url_video_id(source)
        title = info.get("title") or video_id
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise ClipperError(f"No such file: {source}")
        video_id, title, source = local_video_id(path), path.stem, str(path)

    work = work_root / video_id
    work.mkdir(parents=True, exist_ok=True)
    meta_path, video, audio = work / "meta.json", work / "video.mp4", work / "audio.wav"

    if not force and meta_path.exists() and video.exists() and audio.exists():
        meta = read_json(meta_path)
        if meta.get("version") == INGEST_VERSION:
            log(f"ingest: reusing {work}")
            return work, meta

    src = download(source, work) if is_url(source) else Path(source)
    info = ffprobe(src)
    vs = video_stream(info)
    if vs is None:
        raise ClipperError(f"{src.name} has no video stream")
    if audio_stream(info) is None:
        raise ClipperError(f"{src.name} has no audio stream; clipper needs speech to pick clips")

    rate = pick_frame_rate(vs)
    log(f"ingest: converting to constant {float(rate):.3f} fps H.264 (max {MAX_HEIGHT}p)")
    normalize(src, video, rate)
    extract_audio(video, audio)

    out = ffprobe(video)
    ovs = video_stream(out)
    meta = {
        "version": INGEST_VERSION,
        "video_id": video_id,
        "title": title,
        "source": source,
        "fps": f"{rate.numerator}/{rate.denominator}",
        "width": int(ovs["width"]),
        "height": int(ovs["height"]),
        "duration": float(out["format"]["duration"]),
    }
    write_json(meta_path, meta)
    log(f"ingest: {meta['width']}x{meta['height']}, {meta['duration']:.0f}s -> {work}")
    return work, meta
