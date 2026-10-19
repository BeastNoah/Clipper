"""Step 1: get the sound of the video as 16 kHz mono audio, and (on request) a normalised copy of the video.

By default only the sound is prepared at first: that's all transcription and clip selection need.
The picture is fetched later, only for the stretches that become clips (see clip_video): for links
straight from the stream with ffmpeg, for files on disk straight from the file. A long video then
costs a few minutes of video download and conversion instead of all of it. With download "full",
the whole video is downloaded and converted to a standard constant-frame-rate MP4 up front.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .util import (
    ClipperError, ascii_slug, audio_stream, check_cancel, ffprobe, log, parse_rate, read_json, run,
    video_stream, write_json,
)

MAX_HEIGHT = 1080
# Frame rates we normalise to; anything else (including variable rate phone video) snaps to the closest.
STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                  Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60)]
INGEST_VERSION = 1
VIDEO_FORMAT = f"bv*[height<={MAX_HEIGHT}]+ba/b[height<={MAX_HEIGHT}]/bv*+ba/b"
DOWNLOAD_MODES = ("sections", "full")
SECTION_MARGIN = 3.0   # seconds of extra video fetched around a clip
# Whisper doesn't need music-grade sound: a ~50-70 kbps track is a third of the size of the best one.
SPEECH_AUDIO = "ba[abr<=80][abr>=40]"


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
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True,
                           "format": VIDEO_FORMAT}) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") == "playlist":
        raise ClipperError("That URL is a playlist; pass a single video URL.")
    vid = f"{ascii_slug(info.get('extractor_key') or 'web', 12)}-{ascii_slug(str(info['id']), 40)}"
    return vid, info


def _finished_downloads(dest_dir: Path, stem: str = "source") -> list[Path]:
    # Skip yt-dlp's partial files and the per-format pieces (source.f137.mp4) it merges afterwards.
    return sorted(
        p for p in dest_dir.glob(f"{stem}.*")
        if not re.search(rf"\.(part|ytdl|temp)$|^{stem}\.(f\d+|temp)\.", p.name)
    )


def download(url: str, dest_dir: Path, fmt: str = VIDEO_FORMAT, stem: str = "source",
             force: bool = False) -> Path:
    existing = _finished_downloads(dest_dir, stem)
    if force:  # yt-dlp would skip a file that's already there, so it has to go first
        for p in existing:
            p.unlink()
        existing = []
    if existing:
        log(f"ingest: reusing download {existing[0].name}")
        return existing[0]
    yt_dlp = _yt_dlp()
    log(f"ingest: downloading {'the audio of ' if stem == 'audio_source' else ''}{url}")
    opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / f"{stem}.%(ext)s"),
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
        found = _finished_downloads(dest_dir, stem)
        if not found:
            raise ClipperError("yt-dlp finished but no file was written")
        path = found[0]
    return path


def _encode_args(rate: Fraction) -> list[str]:
    """H.264/AAC at a constant frame rate, so audio and picture stay in sync when we cut."""
    gop = max(1, round(float(rate) * 2))  # keyframe every ~2s keeps later seeks fast
    vf = (
        "scale='trunc(iw*sar/2)*2':'trunc(ih/2)*2',setsar=1,"
        f"scale=-2:'min({MAX_HEIGHT},ih)':flags=lanczos,"
        f"fps={rate.numerator}/{rate.denominator},format=yuv420p"
    )
    return ["-vf", vf, "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", str(gop),
            "-af", "aresample=async=1:first_pts=0", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart"]


def normalize(src: Path, dest: Path, rate: Fraction) -> None:
    tmp = dest.with_name("video.tmp.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0",
         *_encode_args(rate), str(tmp)])
    os.replace(tmp, dest)


def extract_audio(video: Path, dest: Path) -> None:
    tmp = dest.with_name("audio.tmp.wav")
    # The first audio track, like the clips' sound (normalize, fetch_section): left to itself ffmpeg
    # takes the one with the most channels, and the transcript could then be of another track.
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(tmp)])
    os.replace(tmp, dest)


def _stream_choice(info: dict) -> dict:
    """The video and audio formats yt-dlp picked for VIDEO_FORMAT (the same one when it's a single file)."""
    req = info.get("requested_formats") or [info]
    video = next((f for f in req if f.get("vcodec") not in (None, "none")), req[0])
    audio = next((f for f in req if f.get("acodec") not in (None, "none") and f is not video), video)
    return {"video_format": video["format_id"], "audio_format": audio["format_id"],
            "width": video.get("width"), "height": video.get("height"), "fps": video.get("fps")}


def _thumbnail(info: dict, dest: Path) -> None:
    url = info.get("thumbnail")
    if not url or dest.exists():
        return
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            dest.write_bytes(r.read())
    except Exception:  # noqa: BLE001 - a missing thumbnail only affects the web UI's picture
        pass


def _ingest_audio_only(url: str, info: dict, work: Path, video_id: str, title: str, force: bool = False) -> dict:
    choice = _stream_choice(info)
    # the clips' own sound later comes from the best track (choice["audio_format"]), not this one
    src = download(url, work, fmt=f"{SPEECH_AUDIO}/{choice['audio_format']}", stem="audio_source",
                   force=force)
    extract_audio(src, work / "audio.wav")
    _thumbnail(info, work / "thumb.jpg")
    rate = pick_frame_rate({"avg_frame_rate": str(Fraction(choice["fps"]).limit_denominator(1001))
                            if choice["fps"] else None})
    duration = float(ffprobe(work / "audio.wav")["format"]["duration"])
    return {
        "version": INGEST_VERSION, "video_id": video_id, "title": title, "source": url,
        "video": "sections",
        "formats": {"video": choice["video_format"], "audio": choice["audio_format"]},
        "youtube_id": info.get("id") if info.get("extractor_key") == "Youtube" else None,
        "fps": f"{rate.numerator}/{rate.denominator}",
        "width": int(choice["width"] or 1920), "height": int(choice["height"] or 1080),
        "duration": duration,
    }


def display_size(stream: dict) -> tuple[int, int]:
    """Width and height of a video stream as the clips will get it: square pixels, upright, at most
    MAX_HEIGHT tall (what _encode_args makes of it)."""
    w, h = int(stream["width"]), int(stream["height"])
    sar = parse_rate(stream.get("sample_aspect_ratio", "1:1").replace(":", "/")) or Fraction(1)
    w = int(w * sar) // 2 * 2
    rotation = next((d.get("rotation") for d in stream.get("side_data_list") or [] if "rotation" in d),
                    (stream.get("tags") or {}).get("rotate", 0))
    if abs(int(float(rotation or 0))) % 180 == 90:
        w, h = h, w
    if h > MAX_HEIGHT:
        w, h = int(round(w * MAX_HEIGHT / h / 2)) * 2, MAX_HEIGHT
    return w, h


def _ingest_local_audio(path: Path, work: Path, video_id: str, title: str) -> dict:
    probe = ffprobe(path)
    vs = _check_streams(probe, path)
    rate = pick_frame_rate(vs)
    extract_audio(path, work / "audio.wav")
    width, height = display_size(vs)
    return {
        "version": INGEST_VERSION, "video_id": video_id, "title": title, "source": str(path),
        "video": "sections", "local": True,
        "fps": f"{rate.numerator}/{rate.denominator}", "width": width, "height": height,
        "duration": float(ffprobe(work / "audio.wav")["format"]["duration"]),
    }


def _check_streams(probe: dict, path: Path) -> dict:
    vs = video_stream(probe)
    if vs is None:
        raise ClipperError(f"{path.name} has no video stream")
    if audio_stream(probe) is None:
        raise ClipperError(f"{path.name} has no audio stream; clipper needs speech to pick clips")
    return vs


def ingest(source: str, work_root: Path, force: bool = False, download_mode: str = "sections") -> tuple[Path, dict]:
    """Returns (work dir for this video, meta)."""
    if download_mode not in DOWNLOAD_MODES:
        raise ClipperError(f"download must be one of {', '.join(DOWNLOAD_MODES)}")
    info = None
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

    if not force and meta_path.exists() and audio.exists():
        meta = read_json(meta_path)
        have_video = video.exists() or (meta.get("video") == "sections" and download_mode == "sections")
        if meta.get("version") == INGEST_VERSION and have_video:
            if info is None and meta.get("local") and meta.get("source") != source:
                # Same file (the id comes from its content) at a new place: cut the clips from there now.
                meta["source"] = source
                write_json(meta_path, meta)
            log(f"ingest: reusing {work}")
            return work, meta

    if download_mode == "sections":
        if info is not None:
            meta = _ingest_audio_only(source, info, work, video_id, title, force)
            log(f"ingest: audio only for now ({meta['duration']:.0f}s); video is fetched just for the clips")
        else:
            meta = _ingest_local_audio(Path(source), work, video_id, title)
            log(f"ingest: took the sound ({meta['duration']:.0f}s); only the clips' parts of the video "
                "get converted later")
        write_json(meta_path, meta)
        return work, meta

    src = download(source, work, force=force) if is_url(source) else Path(source)
    vs = _check_streams(ffprobe(src), src)

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
        "video": "full",
        "fps": f"{rate.numerator}/{rate.denominator}",
        "width": int(ovs["width"]),
        "height": int(ovs["height"]),
        "duration": float(out["format"]["duration"]),
    }
    write_json(meta_path, meta)
    log(f"ingest: {meta['width']}x{meta['height']}, {meta['duration']:.0f}s -> {work}")
    return work, meta


# ---------------------------------------------------------------------------------------------
# Video for one clip

@dataclass
class ClipVideo:
    """A video file covering a clip. `offset` is where the file's first frame is on the timeline of
    the transcript (0 for the full video, the section start for a section)."""
    path: Path
    offset: float
    width: int
    height: int
    fps: str

    def key(self) -> list:
        return [self.path.name, round(self.offset, 3), self.width, self.height, self.fps]


_inputs_cache: dict[tuple, tuple[float, list]] = {}
_inputs_lock = threading.Lock()  # clips fetch their sections in parallel; look the URLs up only once


def local_source(meta: dict) -> Path | None:
    """The original file, for a video on this computer that was never converted as a whole."""
    return Path(meta["source"]) if meta.get("local") else None


def section_inputs(meta: dict) -> list[tuple[str, dict]]:
    """What to read a section from: the original file, or for links the direct stream URLs (with the
    HTTP headers they need) of the formats chosen at ingest. Those expire after a few hours, so
    they're looked up again when older than an hour."""
    if meta.get("local"):
        path = local_source(meta)
        if not path.is_file():
            raise ClipperError(f"The original video {path} is gone (moved or deleted); clipper needs it to "
                               "cut the clips. Put it back or run clipper on it again.")
        return [(str(path), {})]
    with _inputs_lock:
        return _stream_inputs(meta)


def _stream_inputs(meta: dict) -> list[tuple[str, dict]]:
    fmts = meta["formats"]
    key = (meta["source"], fmts["video"], fmts["audio"])
    hit = _inputs_cache.get(key)
    if hit and time.monotonic() - hit[0] < 3600:
        return hit[1]
    spec = fmts["video"] if fmts["video"] == fmts["audio"] else f"{fmts['video']}+{fmts['audio']}"
    yt_dlp = _yt_dlp()
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True, "format": spec}) as ydl:
        info = ydl.extract_info(meta["source"], download=False)
    req = info.get("requested_formats") or [info]
    inputs = [(f["url"], f.get("http_headers") or {}) for f in req]
    _inputs_cache[key] = (time.monotonic(), inputs)
    return inputs


def fetch_section(inputs: list[tuple[str, dict]], start: float, end: float, rate: Fraction, dest: Path) -> None:
    """Download (or read) and encode only [start, end) of the stream(s). ffmpeg seeks inside remote files
    with HTTP range requests, so only that part (plus up to one keyframe interval) is transferred."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    for url, headers in inputs:
        if str(url).startswith(("http://", "https://")):
            # give up on a stalled connection after 30s instead of hanging; reconnect after drops
            cmd += ["-rw_timeout", "30000000", "-reconnect", "1", "-reconnect_on_network_error", "1",
                    "-reconnect_delay_max", "10"]
            if headers:
                cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
        cmd += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(url)]
    tmp = dest.with_name(dest.stem + ".tmp.mp4")
    try:
        run([*cmd, "-map", "0:v:0", "-map", f"{len(inputs) - 1}:a:0", *_encode_args(rate), str(tmp)])
    except ClipperError as e:
        raise ClipperError(f"Couldn't fetch this part of the video ({e}). If it keeps failing, use "
                           "--download full (or 'the whole video' in the web interface).") from e
    os.replace(tmp, dest)


def _decode_audio(path: Path) -> np.ndarray:
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
               "-f", "s16le", "-"]).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def measure_offset(reference: np.ndarray, ref_start: float, section: np.ndarray, rate: int = 16000) -> float | None:
    """Where (in seconds) the section's audio starts within the reference audio, found by
    cross-correlation; None if there's no clear match."""
    # Compare from 0.5s in until 1s before the end: encoders pad the edges a little, and at the very end
    # of a video the padding would run past the end of the reference.
    head = rate // 2
    probe = section[head: min(len(section) - rate, head + rate * 20)]
    if len(probe) < rate or len(reference) < len(probe):
        return None
    n = 1 << int(np.ceil(np.log2(len(reference) + len(probe))))
    corr = np.fft.irfft(np.fft.rfft(reference, n) * np.conj(np.fft.rfft(probe, n)), n)[: len(reference) - len(probe) + 1]
    lag = int(np.argmax(corr))
    window = reference[lag:lag + len(probe)]
    score = corr[lag] / (np.linalg.norm(window) * np.linalg.norm(probe) + 1e-9)
    return ref_start + (lag - head) / rate if score > 0.5 else None


def clip_video(work: Path, meta: dict, clip: dict, clip_dir: Path) -> ClipVideo:
    """The video to cut a clip from: the full video, or a section fetched just for this clip."""
    if meta.get("video") != "sections":
        return ClipVideo(work / "video.mp4", 0.0, meta["width"], meta["height"], meta["fps"])
    path, info_path = clip_dir / "section.mp4", clip_dir / "section.json"
    want = [round(max(0.0, clip["start"] - SECTION_MARGIN), 3),
            round(min(meta["duration"], clip["end"] + SECTION_MARGIN), 3)]
    info = read_json(info_path) if info_path.exists() and path.exists() else None
    if not info or info.get("range") != want:
        clip_dir.mkdir(parents=True, exist_ok=True)
        rate = Fraction(meta["fps"])
        log(f"clip #{clip['rank']}: {'converting' if meta.get('local') else 'fetching'} "
            f"{want[1] - want[0]:.0f}s of video")
        fetch_section(section_inputs(meta), want[0], want[1], rate, path)
        # Check where the section really starts by matching its sound against the full audio track.
        from .transcribe import load_audio
        ref_start = max(0.0, want[0] - 2.0)
        offset = measure_offset(load_audio(work / "audio.wav", ref_start, want[1] + 2.0), ref_start,
                                _decode_audio(path))
        if offset is None:
            log(f"clip #{clip['rank']}: couldn't confirm the section's timing from its sound; trusting the seek")
            offset = want[0]
        elif abs(offset - want[0]) > 0.02:
            log(f"clip #{clip['rank']}: section starts {offset - want[0]:+.2f}s from where it was asked; corrected")
        vs = video_stream(ffprobe(path))
        info = {"range": want, "offset": round(offset, 4), "width": int(vs["width"]), "height": int(vs["height"])}
        write_json(info_path, info)
    return ClipVideo(path, info["offset"], info["width"], info["height"], meta["fps"])
