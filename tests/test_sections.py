"""Downloading only the audio first, and then only the video sections that become clips."""
import json
import re
import shutil
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from clipper import cli, ingest
from clipper.ingest import clip_video, display_size, extract_audio, measure_offset
from clipper.util import ffprobe, video_stream, write_json
from conftest import make_words

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


class RangeHandler(SimpleHTTPRequestHandler):
    """Static files with HTTP range support (like a video CDN), noting where each request starts."""
    sent: dict = {}
    ranges: list = []

    def log_message(self, *args):
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
        if not m or not path.endswith((".mp4", ".m4a")):
            return super().send_head()
        f = open(path, "rb")
        size = f.seek(0, 2)
        start, end = int(m[1]), min(int(m[2]) if m[2] else size - 1, size - 1)
        RangeHandler.ranges.append((path.rsplit("/", 1)[-1], start / size))
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        name = self.path.rsplit("/", 1)[-1]
        left = getattr(self, "_left", None)
        while left is None or left > 0:
            chunk = source.read(65536 if left is None else min(65536, left))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            RangeHandler.sent[name] = RangeHandler.sent.get(name, 0) + len(chunk)
            if left is not None:
                left -= len(chunk)


@pytest.fixture
def cdn(tmp_path):
    www = tmp_path / "www"
    www.mkdir()
    full = www / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=60",
                    "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=60:amplitude=0.3", "-c:v", "libx264",
                    "-preset", "ultrafast", "-g", "60", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
                    str(full)], check=True)
    # like YouTube: picture and sound as separate files
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(full), "-an", "-c", "copy", "-movflags", "+faststart",
                    str(www / "video_only.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(full), "-vn", "-c", "copy", "-movflags", "+faststart",
                    str(www / "audio_only.m4a")], check=True)
    RangeHandler.sent, RangeHandler.ranges = {}, []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(www)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", www
    httpd.shutdown()


def test_measure_offset_finds_the_start():
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(16000 * 30).astype(np.float32)
    section = ref[16000 * 7 + 123: 16000 * 19]
    assert abs(measure_offset(ref, 100.0, section) - (107 + 123 / 16000)) < 1e-6
    assert measure_offset(ref, 0.0, rng.standard_normal(16000 * 5).astype(np.float32)) is None  # no match


def test_only_the_clip_section_is_downloaded(tmp_path, cdn, monkeypatch):
    base, www = cdn
    work = tmp_path / "work" / "yt-abc"
    work.mkdir(parents=True)
    extract_audio(www / "audio_only.m4a", work / "audio.wav")
    meta = {"video": "sections", "source": "https://example.invalid/watch?v=abc", "fps": "30/1",
            "duration": 60.0, "width": 640, "height": 360, "formats": {"video": "137", "audio": "140"}}
    monkeypatch.setattr(ingest, "section_inputs", lambda meta: [
        (f"{base}/video_only.mp4", {"User-Agent": "clipper-test"}), (f"{base}/audio_only.m4a", {})])
    clip = {"rank": 1, "start": 30.0, "end": 40.0}
    src = clip_video(work, meta, clip, tmp_path / "clip")
    assert abs(src.offset - 27.0) < 0.05             # 3s margin, confirmed against the full audio track
    probe = ffprobe(src.path)
    assert abs(float(probe["format"]["duration"]) - 16.0) < 0.2
    assert (video_stream(probe)["width"], video_stream(probe)["height"]) == (640, 360)
    # ffmpeg read the file's index, then jumped straight to ~27s of 60s instead of reading everything up to it
    jumps = [pos for name, pos in RangeHandler.ranges if name == "video_only.mp4" and pos > 0]
    assert jumps and 0.35 < jumps[0] < 0.5

    requests = len(RangeHandler.ranges)
    assert clip_video(work, meta, clip, tmp_path / "clip").offset == src.offset  # cached: no new download
    assert len(RangeHandler.ranges) == requests


def test_pipeline_from_a_link_without_downloading_the_whole_video(tmp_path, cdn, fake_claude):
    base, _ = cdn
    url = f"{base}/talk.mp4"
    root = tmp_path / "work"
    args = [url, "--work-dir", str(root), "-n", "1", "--min-len", "8", "--max-len", "12",
            "--preset", "ultrafast", "--caption-model", "none", "--no-posts"]
    assert cli.main(args + ["--stop-after", "ingest"]) == 0
    (work,) = root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    assert meta["video"] == "sections" and not (work / "video.mp4").exists()
    write_json(work / "transcript.json", {
        "params": {"version": 2, "model": "small", "backend": "faster-whisper", "language": None,
                   "duration": round(meta["duration"], 2)},
        "result": {"language": "en", "words": make_words([f"Sentence number {i} is said here." for i in range(25)],
                                                         start=0.5)}})
    assert cli.main(args) == 0
    (out,) = (work / "output").glob("*.mp4")
    v = video_stream(ffprobe(out))
    assert (v["width"], v["height"]) == (1080, 1920)
    assert list((work / "clips").glob("*/section.mp4")) and not (work / "video.mp4").exists()


def test_display_size_of_rotated_and_large_videos():
    assert display_size({"width": 1920, "height": 1080}) == (1920, 1080)
    assert display_size({"width": 3840, "height": 2160}) == (1920, 1080)
    phone = {"width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]}
    assert display_size(phone) == (608, 1080)
    assert display_size({"width": 1440, "height": 1080, "sample_aspect_ratio": "4:3"}) == (1920, 1080)


def test_a_moved_video_is_found_at_its_new_place(tmp_path):
    from clipper.ingest import ingest, section_inputs

    first = tmp_path / "Downloads" / "talk.mp4"
    first.parent.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=4",
                    "-f", "lavfi", "-i", "sine=f=300:duration=4", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(first)], check=True)
    work_root = tmp_path / "work"
    work, meta = ingest(str(first), work_root)
    assert meta["source"] == str(first.resolve())

    moved = tmp_path / "Videos" / "talk.mp4"
    moved.parent.mkdir()
    first.rename(moved)
    work2, meta2 = ingest(str(moved), work_root)
    assert work2 == work and meta2["source"] == str(moved.resolve())
    assert section_inputs(meta2) == [(str(moved.resolve()), {})]
    assert json.loads((work / "meta.json").read_text())["source"] == str(moved.resolve())
