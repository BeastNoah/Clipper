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
                "--preset", "ultrafast", "--caption-model", "none", *extra]

    assert cli.main(args(2, "--stop-after", "ingest")) == 0
    (work,) = work_root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    assert (meta["width"], meta["height"], meta["fps"]) == (640, 360, "25/1")
    audio = ffprobe(work / "audio.wav")["streams"][0]
    assert (audio["sample_rate"], audio["channels"]) == ("16000", 1)
    # only the sound is taken from a file at first; the clips' parts are converted later
    assert meta["video"] == "sections" and not (work / "video.mp4").exists()

    words = make_words([f"This is spoken sentence number {i} today." for i in range(14)], start=0.5)
    write_json(work / "transcript.json", {
        "params": {"version": 2, "model": "small", "backend": "faster-whisper", "language": None,
                   "duration": round(meta["duration"], 2)},
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
    sections = sorted((work / "clips").glob("*/section.mp4"))
    assert len(sections) == 2 and all(float(ffprobe(p)["format"]["duration"]) < 20 for p in sections)
    for c in clips:  # cover image and post text next to each clip
        assert (work / "output" / c["cover"]).stat().st_size > 1000
        assert "#fyp" in (work / "output" / c["file"]).with_suffix(".txt").read_text()
        assert c["post"]["youtube_title"]

    # No faces in a test pattern: every shot falls back to the full frame on a blurred background.
    plan = json.loads(next((work / "clips").glob("*/9x16/reframe.json")).read_text())["result"]
    assert all(s["mode"] == "fit" for s in plan["shots"])

    # A rerun asking for fewer clips reuses everything: no new Claude calls, no re-encoding.
    calls = len(fake_claude.read_text().splitlines())
    mtime = outputs[0].stat().st_mtime_ns
    assert cli.main(args(1)) == 0
    assert len(fake_claude.read_text().splitlines()) == calls
    assert [p.name for p in (work / "output").glob("*.mp4")] == [outputs[0].name]
    assert outputs[0].stat().st_mtime_ns == mtime


def test_pipeline_with_formats_translation_logo_and_focus(tmp_path, fake_claude):
    import cv2
    import numpy as np

    src = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=40",
                    "-f", "lavfi", "-i", "sine=f=300:duration=40", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    logo = np.zeros((30, 60, 4), np.uint8)
    logo[..., 1:] = 255
    cv2.imwrite(str(tmp_path / "logo.png"), logo)
    root = tmp_path / "work"
    args = [str(src), "--work-dir", str(root), "-n", "1", "--min-len", "8", "--max-len", "12", "--preset",
            "ultrafast", "--caption-model", "none", "--format", "9:16,1:1,16:9", "--caption-language", "English",
            "--logo", str(tmp_path / "logo.png"), "--progress-bar", "--prompt", "the best bits",
            "--from", "0:10", "--to", "0:38"]
    assert cli.main(args + ["--stop-after", "ingest"]) == 0
    (work,) = root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    write_json(work / "transcript.json", {
        "params": {"version": 2, "model": "small", "backend": "faster-whisper", "language": None,
                   "duration": round(meta["duration"], 2)},
        "result": {"language": "de", "words": make_words([f"Das ist Satz Nummer {i} heute." for i in range(14)],
                                                         start=0.5)}})
    assert cli.main(args) == 0
    (clip,) = json.loads((work / "output" / "clips.json").read_text())
    assert clip["start"] >= 9.5
    sizes = {}
    for ratio, f in clip["formats"].items():
        v = video_stream(ffprobe(work / "output" / f["file"]))
        sizes[ratio] = (v["width"], v["height"])
    assert sizes == {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}
    srts = [(work / "output" / f).read_text() for f in clip["srt"]]
    assert len(srts) == 2 and "Das ist Satz" in srts[0] and "[EN]" in srts[1]
    ass = next((work / "clips").glob("*/9x16/captions.ass")).read_text()
    assert "[EN]" in ass and "Title,,0,0,0,,{\\fad(150,300)}[EN]" in ass  # translated lines and title
    prompts = [json.loads(l)["prompt"] for l in fake_claude.read_text().splitlines()]
    assert any("looking for: the best bits" in p for p in prompts)


def test_whole_file_conversion_on_request(tmp_path, fake_claude):
    src = tmp_path / "talk.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=30",
                    "-f", "lavfi", "-i", "sine=f=300:duration=30", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    root = tmp_path / "work"
    args = [str(src), "--work-dir", str(root), "-n", "1", "--min-len", "8", "--max-len", "12",
            "--preset", "ultrafast", "--caption-model", "none", "--download", "full"]
    assert cli.main([*args, "--stop-after", "ingest"]) == 0
    (work,) = root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    assert meta["video"] == "full" and (work / "video.mp4").exists()
    write_json(work / "transcript.json", {
        "params": {"version": 2, "model": "small", "backend": "faster-whisper", "language": None,
                   "duration": round(meta["duration"], 2)},
        "result": {"language": "en", "words": make_words(
            [f"This is spoken sentence number {i} today." for i in range(10)], start=0.5)},
    })
    assert cli.main(args) == 0
    assert len(list((work / "output").glob("*.mp4"))) == 1
    assert not list((work / "clips").glob("*/section.mp4"))


def test_new_clips_for_a_video_done_before(tmp_path, fake_claude):
    src = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=75",
                    "-f", "lavfi", "-i", "sine=f=300:duration=75", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    root = tmp_path / "work"
    args = [str(src), "--work-dir", str(root), "-n", "1", "--min-len", "6", "--max-len", "10",
            "--preset", "ultrafast", "--caption-model", "none", "--no-posts"]
    assert cli.main([*args, "--stop-after", "ingest"]) == 0
    (work,) = root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    write_json(work / "transcript.json", {
        "params": {"version": 2, "model": "small", "backend": "faster-whisper", "language": None,
                   "duration": round(meta["duration"], 2)},
        "result": {"language": "en", "words": make_words(
            [f"This is spoken sentence number {i} today." for i in range(22)], start=0.5)},
    })
    assert cli.main(args) == 0
    (first,) = json.loads((work / "output" / "clips.json").read_text())
    first_mp4 = work / "output" / first["file"]
    mtime = first_mp4.stat().st_mtime_ns

    assert cli.main([*args, "--new-clips"]) == 0
    clips = json.loads((work / "output" / "clips.json").read_text())
    assert [c["id"] for c in clips][0] == first["id"] and len(clips) == 2
    new = clips[1]
    assert new["start"] >= first["end"] or new["end"] <= first["start"]
    assert first_mp4.stat().st_mtime_ns == mtime  # the clip made before isn't made again
    assert len(list((work / "output").glob("*.mp4"))) == 2


def test_failed_rerender_keeps_the_clips_already_made(tmp_path, fake_claude, monkeypatch):
    import clipper.pipeline as pipeline
    from clipper.util import ClipperError

    src = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=40",
                    "-f", "lavfi", "-i", "sine=f=300:duration=40", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    work_root = tmp_path / "work"

    def args(*extra):
        return [str(src), "--work-dir", str(work_root), "-n", "2", "--min-len", "8", "--max-len", "12",
                "--preset", "ultrafast", "--caption-model", "none", *extra]

    assert cli.main(args("--stop-after", "ingest")) == 0
    (work,) = work_root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    write_json(work / "transcript.json", {
        "params": {"version": 2, "model": "small", "backend": "faster-whisper", "language": None,
                   "duration": round(meta["duration"], 2)},
        "result": {"language": "en", "model": "small",
                   "words": make_words([f"This is spoken sentence number {i} today." for i in range(14)], start=0.5)},
    })
    assert cli.main(args()) == 0
    out = work / "output"
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert sum(name.endswith(".mp4") for name in before) == 2 and "clips.json" in before

    real_render, calls = pipeline.render, []

    def fails_on_second_clip(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise ClipperError("disk full")
        return real_render(*a, **k)

    monkeypatch.setattr(pipeline, "render", fails_on_second_clip)
    assert cli.main(args("--force", "render")) != 0
    after = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert after == before  # the earlier clips and their list survive a render that didn't finish

    monkeypatch.setattr(pipeline, "render", real_render)
    assert cli.main(args("--force", "render")) == 0
    assert not (out / ".next").exists()
    assert sorted(p.name for p in out.iterdir()) == sorted(before)


def test_clip_with_an_earlier_part(tmp_path, fake_claude, monkeypatch):
    from clipper.pipeline import caption_lines, clip_dir_for

    monkeypatch.setenv("FAKE_CLAUDE_BEFORE", "1")  # the best clip plays the first sentence before it
    src = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=40",
                    "-f", "lavfi", "-i", "sine=f=300:duration=40", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    root = tmp_path / "work"
    args = [str(src), "--work-dir", str(root), "-n", "1", "--min-len", "8", "--max-len", "12",
            "--preset", "ultrafast", "--caption-model", "none", "--jump-cuts", "off", "--no-posts"]
    assert cli.main([*args, "--stop-after", "ingest"]) == 0
    (work,) = root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    write_json(work / "transcript.json", {
        "params": {"version": 2, "model": "small", "backend": "faster-whisper", "language": None,
                   "duration": round(meta["duration"], 2)},
        "result": {"language": "en", "words": make_words(
            [f"This is spoken sentence number {i} today." for i in range(14)], start=0.5)},
    })
    assert cli.main(args) == 0
    (clip,) = json.loads((work / "output" / "clips.json").read_text())
    (part,) = clip["before"]
    assert part["si"] == 0 and part["end"] < clip["start"]
    assert 8 <= clip["duration"] <= 12
    out = work / "output" / clip["file"]
    assert float(ffprobe(out)["format"]["duration"]) == pytest.approx(clip["duration"], abs=0.15)
    # each part's section is fetched on its own (not the stretch in between); the render has its own folder
    sections = sorted((work / "clips").glob("*/section.mp4"))
    assert len(sections) == 2 and all(float(ffprobe(p)["format"]["duration"]) < 16 for p in sections)
    cdir = clip_dir_for(work, clip)
    assert "+" in cdir.name and (cdir / "9x16" / "clip.mp4").exists()
    srt = (work / "output" / clip["srt"][0]).read_text()
    assert "number 0 today" in srt and f"number {clip['si']} today" in srt
    lines = caption_lines(work, clip)
    assert lines[0]["text"] == "This is spoken sentence number 0 today."
    assert lines[1]["start"] >= clip["start"]

    # a second run reuses the render
    mtime = out.stat().st_mtime_ns
    assert cli.main(args) == 0
    assert out.stat().st_mtime_ns == mtime
