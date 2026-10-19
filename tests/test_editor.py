"""The clip editor: your own changes on top of Claude's proposal, applied by the render."""
import json
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from clipper import cli
from clipper.edits import ClipEdits, apply_cuts
from clipper.ui.server import App, Handler
from clipper.util import ClipperError, write_json
from conftest import make_words
from test_ui import call, wait

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_apply_cuts_keeps_and_removes_stretches():
    keep = [(0.0, 2.0), (3.0, 5.0)]  # clipper would cut 2-3s
    assert apply_cuts(keep, [], [(1.9, 3.1)], 5.0, 10) == [(0.0, 5.0)]            # keep the pause
    assert apply_cuts(keep, [(0.5, 1.0)], [], 5.0, 10) == [(0.0, 0.5), (1.0, 2.0), (3.0, 5.0)]
    assert apply_cuts(keep, [(1.5, 3.5)], [(2.0, 3.0)], 5.0, 10) == [(0.0, 1.5), (3.5, 5.0)]  # removed wins
    with pytest.raises(ClipperError):
        apply_cuts(keep, [(0.0, 5.0)], [], 5.0, 10)


def test_edits_are_checked():
    ed = ClipEdits.from_dict({"title": "  My   title ", "keywords": ["x", ""], "layouts": [{"start": 1, "end": 3, "layout": "split"}],
                              "texts": [{"text": "Hi", "start": 0, "end": 2, "position": "middle"}], "zoom": "off"})
    assert ed.title == "My title" and ed.keywords == ["x"] and ed.layouts[0]["layout"] == "split" and not ed.is_empty
    assert ed.layouts_in_part(2.0, 10.0) == [(0.0, 1.0, "split")] and ed.in_part([[1, 3]], 2.0, 10.0) == [(0.0, 1.0)]
    assert ClipEdits.from_dict({}).is_empty and ClipEdits.from_dict(None).to_dict() == ClipEdits().to_dict()
    for bad in ({"layouts": [{"start": 3, "end": 1, "layout": "split"}]}, {"jump_cuts": "sometimes"},
                {"texts": [{"text": "x" * 200, "start": 0, "end": 1}]}, {"music": {"file": ""}}, {"cuts_removed": [[1]]},
                {"transition": "wipe"}, [1, 2]):
        with pytest.raises(ClipperError):
            ClipEdits.from_dict(bad)


@pytest.fixture
def server(tmp_path, fake_claude):
    src = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=25:duration=24",
                    "-f", "lavfi", "-i", "sine=f=300:duration=24", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    root = tmp_path / "work"
    assert cli.main([str(src), "--work-dir", str(root), "--stop-after", "ingest"]) == 0
    (work,) = root.iterdir()
    write_json(work / "transcript.json", {
        "params": {}, "result": {"language": "en", "words": make_words(
            [f"This is spoken sentence number {i} today." for i in range(8)], start=0.5, sentence_gap=1.2)}})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"app": App(root)}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], work
    httpd.shutdown()


def render_first(port, vid, options):
    status, job, _ = call(port, "POST", "/api/jobs", {"kind": "render", "video": vid, "options": options,
                                                     "items": [{"si": 1, "ei": 4, "title": "Claude's pick"}]})
    assert status == 201
    job = wait(port, job["id"])
    assert job["state"] == "done", job


def test_editor_round_trip(server, tmp_path):
    port, work = server
    vid = work.name
    options = {"caption_model": "none", "preset": "ultrafast", "jump_cuts": "on"}
    render_first(port, vid, options)

    status, data, _ = call(port, "GET", f"/api/videos/{vid}/edit/1")
    assert status == 200 and data["clip"]["title"] == "Claude's pick" and data["edits"] == ClipEdits().to_dict()
    (part,) = data["parts"]
    assert part["media"]["url"].endswith("/section.mp4") and part["words"] and part["plan"]["shots"]
    assert part["removed"], "the pauses between the sentences are proposed as jump cuts"
    assert len(part["wave"]) > 10 and all(0 <= v <= 1 for v in part["wave"])
    assert data["jump_cuts"] == "on" and data["sentences"][1]["i"] == 1
    assert call(port, "GET", f"/api/videos/{vid}/edit/1?jump_cuts=off")[1]["parts"][0]["removed"] == []
    assert call(port, "GET", f"/api/videos/{vid}/edit/7")[0] == 404
    # the section plays in the editor, and frames of it make the timeline's thumbnails
    status, body, r = call(port, "GET", part["media"]["url"], headers={"Range": "bytes=0-99"})
    assert status == 206 and r.getheader("Content-Type") == "video/mp4"
    status, _, r = call(port, "GET", f"/api/videos/{vid}/frame?t={part['start'] + 1}&clip={part['dir']}")
    assert status == 200 and r.getheader("Content-Type") == "image/jpeg"
    assert call(port, "GET", f"/media/{vid}/clips/../../meta.json")[0] == 404

    # uploads become assets for B-roll and music
    import cv2
    import numpy as np
    png = tmp_path / "card.png"
    cv2.imwrite(str(png), np.full((40, 80, 3), 200, np.uint8))
    status, up, _ = call(port, "PUT", "/api/upload?name=card.png", body=png.read_bytes())
    assert status == 201
    wav = tmp_path / "beat.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:duration=3", str(wav)], check=True)
    status, music, _ = call(port, "PUT", "/api/upload?name=beat.wav", body=wav.read_bytes())
    assert status == 201
    assets = call(port, "GET", "/api/assets")[1]
    assert {(a["name"], a["kind"]) for a in assets} == {("card.png", "image"), ("beat.wav", "audio")}
    assert call(port, "GET", next(a["url"] for a in assets if a["kind"] == "image"))[0] == 200

    # a file that wasn't uploaded is refused, and so are bad edits
    bad = {"edits": {"broll": [{"file": str(png), "start": part["start"], "end": part["start"] + 1}]}, "options": options}
    assert call(port, "POST", f"/api/videos/{vid}/edit/1", bad)[0] == 400
    assert call(port, "POST", f"/api/videos/{vid}/edit/1", {"edits": {"zoom": "lots"}})[0] == 400

    s0, removed = part["start"], part["removed"][0]
    edits = {
        "title": "My own title", "keywords": ["number"], "zoom": "off", "zooms": [[s0 + 1, s0 + 2]],
        "cuts_removed": [[s0 + 0.5, s0 + 1.0]], "cuts_kept": [removed], "jump_cuts": "on",
        "layouts": [{"start": s0, "end": s0 + 2, "layout": "fit"}],
        "texts": [{"text": "Watch this", "start": s0 + 2, "end": s0 + 4, "position": "middle"}],
        "broll": [{"file": up["path"], "start": s0 + 3, "end": s0 + 4}],
        "music": {"file": music["path"], "volume": 30}, "fade_in": True, "fade_out": True,
    }
    status, reply, _ = call(port, "POST", f"/api/videos/{vid}/edit/1", {"edits": edits, "options": options})
    assert status == 201 and reply["job"]["label"].startswith("Update clip 1") and reply["edits"]["title"] == "My own title"
    job = wait(port, reply["job"]["id"])
    assert job["state"] == "done", job
    data = call(port, "GET", f"/api/videos/{vid}/edit/1")[1]
    assert data["edits"]["cuts_kept"] == [removed] and data["clip"]["title"] == "My own title"
    assert data["clip"]["claude_title"] == "Claude's pick" and data["clip"]["edited"] is True
    clip = call(port, "GET", f"/api/videos/{vid}")[1]["rendered"][0]
    assert clip["title"] == "My own title" and clip["file"].startswith("01_my-own-title")
    # the kept pause is in the clip, the stretch cut out by hand is not
    assert any(a <= removed[0] and b >= removed[1] for a, b, _ in clip["timeline"])
    assert not any(a < s0 + 0.9 and b > s0 + 0.6 for a, b, _ in clip["timeline"])
    ass = next(work.glob("clips/*/9x16/captions.ass")).read_text()
    assert "Watch this" in ass and "Text_middle" in ass and "My own title" in ass
    plan = json.loads(next(work.glob("clips/*/9x16/reframe.json")).read_text())
    assert plan["params"]["layouts"] == [[0.0, 2.0, "fit"]] and plan["result"]["shots"][0]["chosen"] == "fit"
    stamp = json.loads(next(work.glob("clips/*/9x16/render.json")).read_text())
    assert stamp["look"]["music"][1] == 30 and stamp["look"]["broll"] and stamp["look"]["fades"] == [0.0, 0.4, 0.5]
    (zoom,) = stamp["look"]["zoom"]  # only the punch-in you added (the automatic ones are off), in output time:
    assert abs(zoom[0] - 0.5) < 0.1 and abs(zoom[1] - 1.5) < 0.1  # half a second before it was cut out
    assert float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(work / "output" / clip["file"])], capture_output=True, text=True).stdout) > 3

    # clearing the changes brings Claude's proposal back (the title too)
    status, reply, _ = call(port, "POST", f"/api/videos/{vid}/edit/1", {"edits": {}, "options": options})
    assert wait(port, reply["job"]["id"])["state"] == "done"
    clip = call(port, "GET", f"/api/videos/{vid}")[1]["rendered"][0]
    assert clip["title"] == "Claude's pick" and not clip.get("edited") and "claude_title" not in clip
    assert not (work / "edits" / "u1-4.json").exists()


def test_sections_and_hooks(server):
    port, work = server
    vid = work.name
    options = {"caption_model": "none", "preset": "ultrafast"}
    render_first(port, vid, options)
    # play sentence 0 first (a set-up), and run the clip on to sentence 5
    status, reply, _ = call(port, "POST", f"/api/videos/{vid}/edit/1",
                            {"edits": {}, "options": options, "parts": {"si": 1, "ei": 5, "before": [[0, 0]]}})
    assert status == 201 and reply["clip"]["ei"] == 5 and reply["clip"]["before"][0]["si"] == 0
    assert wait(port, reply["job"]["id"])["state"] == "done"
    data = call(port, "GET", f"/api/videos/{vid}/edit/1")[1]
    assert [(p["si"], p["ei"]) for p in data["parts"]] == [(0, 0), (1, 5)]
    assert all(p["media"] and p["plan"] and p["words"] for p in data["parts"])
    # fixing words of the clip right after adding a section: the new section keeps its own words
    lines = [{"start": w["s"], "end": w["e"], "text": "FIXED"} for w in data["parts"][1]["words"][:1]]
    status, job, _ = call(port, "POST", f"/api/videos/{vid}/captions/1", {"lines": lines, "options": options})
    assert status == 201 and wait(port, job["id"])["state"] == "done"
    data = call(port, "GET", f"/api/videos/{vid}/edit/1")[1]
    assert data["parts"][0]["words"][0]["w"].startswith("This") and data["parts"][1]["words"][0]["w"] == "FIXED"
    assert call(port, "POST", f"/api/videos/{vid}/edit/1", {"edits": {}, "parts": {"si": 2, "ei": 1}})[0] == 400

    # AI hook: Claude suggests other titles, kept with the clip
    assert data["hooks"] is None
    status, job, _ = call(port, "POST", f"/api/videos/{vid}/edit/1/hooks", {"options": options})
    assert status == 201 and job["kind"] == "hooks"
    assert wait(port, job["id"])["state"] == "done"
    hooks = call(port, "GET", f"/api/videos/{vid}/edit/1")[1]["hooks"]
    assert [h["title"] for h in hooks][:2] == ["Why Claude's pick matters", "Claude's pick, explained"]
