"""End to end on a tiny generated video. Whisper is skipped by pre-filling its cached result."""
import json
import shutil
import subprocess

import pytest

from clipper import cli
from clipper.util import ffprobe, video_stream, write_json
from conftest import make_words

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_pipeline_end_to_end(tmp_path, fake_claude):
    src = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=40",
                    "-f", "lavfi", "-i", "sine=f=300:duration=40", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    work_root = tmp_path / "work"
    def args(n, *extra):
        return [str(src), "--work-dir", str(work_root), "-n", str(n), "--min-len", "8", "--max-len", "12",
                "--preset", "ultrafast", *extra]

    assert cli.main(args(2, "--stop-after", "ingest")) == 0
    (work,) = work_root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    assert (meta["width"], meta["height"], meta["fps"]) == (640, 360, "25/1")
    audio = ffprobe(work / "audio.wav")["streams"][0]
    assert (audio["sample_rate"], audio["channels"]) == ("16000", 1)

    words = make_words([f"This is spoken sentence number {i} today." for i in range(14)], start=0.5)
    write_json(work / "transcript.json", {
        "params": {"version": 1, "model": "small", "language": None, "duration": round(meta["duration"], 2)},
        "result": {"language": "en", "model": "small", "words": words},
    })

    assert cli.main(args(2)) == 0
    outputs = sorted((work / "output").glob("*.mp4"))
    assert len(outputs) == 2
    for out in outputs:
        info = ffprobe(out)
        v = video_stream(info)
        assert (v["width"], v["height"]) == (1080, 1920)
        assert any(s["codec_type"] == "audio" for s in info["streams"])
        assert 8 <= float(info["format"]["duration"]) <= 12.5
    clips = json.loads((work / "output" / "clips.json").read_text())
    assert [c["rank"] for c in clips] == [1, 2]

    # No faces in a test pattern: every shot falls back to the full frame on a blurred background.
    plan = json.loads(next((work / "clips").glob("*/reframe.json")).read_text())["result"]
    assert all(s["mode"] == "fit" for s in plan["shots"])

    # A rerun asking for fewer clips reuses everything: no new Claude calls, no re-encoding.
    calls = len(fake_claude.read_text().splitlines())
    mtime = outputs[0].stat().st_mtime_ns
    assert cli.main(args(1)) == 0
    assert len(fake_claude.read_text().splitlines()) == calls
    assert [p.name for p in (work / "output").glob("*.mp4")] == [outputs[0].name]
    assert outputs[0].stat().st_mtime_ns == mtime
