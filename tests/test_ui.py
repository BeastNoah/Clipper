"""The web UI's HTTP API, against a real (tiny) video."""
import http.client
import json
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from clipper import cli
from clipper.ui.server import App, Handler
from clipper.util import write_json
from conftest import make_words

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


@pytest.fixture
def server(tmp_path, fake_claude):
    src = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=25:duration=20",
                    "-f", "lavfi", "-i", "sine=f=300:duration=20", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    root = tmp_path / "work"
    assert cli.main([str(src), "--work-dir", str(root), "--stop-after", "ingest"]) == 0
    (work,) = root.iterdir()
    meta = json.loads((work / "meta.json").read_text())
    write_json(work / "transcript.json", {
        "params": {}, "result": {"language": "en", "words": make_words(
            [f"This is spoken sentence number {i} today." for i in range(7)], start=0.5)}})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"app": App(root)}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], work, meta
    httpd.shutdown()


def call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    conn.request(method, path, body=data, headers=headers or {})
    r = conn.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw), r
    except ValueError:
        return r.status, raw, r


def test_ui_api(server):
    port, work, meta = server
    vid = work.name

    status, choices, _ = call(port, "GET", "/api/choices")
    assert status == 200 and choices["defaults"]["num_clips"] == 3 and "split" in choices["layouts"]
    status, page, _ = call(port, "GET", "/")
    assert status == 200 and b"/static/app.js" in page

    status, videos, _ = call(port, "GET", "/api/videos")
    assert [v["id"] for v in videos] == [vid] and videos[0]["status"] == "transcribed"
    status, detail, _ = call(port, "GET", f"/api/videos/{vid}")
    assert len(detail["sentences"]) == 7 and detail["candidates"] == []

    status, body, r = call(port, "GET", f"/media/{vid}/video.mp4", headers={"Range": "bytes=0-99"})
    assert status == 206 and len(body) == 100 and r.getheader("Content-Range").startswith("bytes 0-99/")
    status, body, r = call(port, "GET", f"/api/videos/{vid}/frame?t=2")
    assert status == 200 and r.getheader("Content-Type") == "image/jpeg"

    for path in (f"/media/{vid}/transcript.json", f"/media/{vid}/../{vid}/meta.json", "/media/..%2f..%2fetc/passwd",
                 "/static/../server.py", "/api/videos/NOPE"):
        assert call(port, "GET", path)[0] == 404, path

    status, up, _ = call(port, "PUT", "/api/upload?name=../../evil%20name.mp4", body=b"x" * 1000,
                         headers={"Content-Length": "1000"})
    assert status == 201 and up["path"] == str(work.parent / "_uploads" / "evil_name.mp4")

    assert call(port, "POST", "/api/jobs", {"kind": "nope"})[0] == 400
    assert call(port, "POST", "/api/jobs", {"kind": "render", "video": vid, "items": []})[0] == 400
    assert call(port, "POST", "/api/jobs", {"kind": "process", "source": "x", "options": {"layout": "bad"}})[0] == 400

    status, job, _ = call(port, "POST", "/api/jobs", {
        "kind": "render", "video": vid, "items": [{"si": 1, "ei": 3, "title": "Hand picked"}],
        "options": {"caption_model": "none", "preset": "ultrafast"}})
    assert status == 201
    for _ in range(240):
        status, job, _ = call(port, "GET", f"/api/jobs/{job['id']}")
        if job["state"] not in ("queued", "running"):
            break
        time.sleep(0.25)
    assert job["state"] == "done", job
    assert any("render #1" in line for line in job["log"])
    status, detail, _ = call(port, "GET", f"/api/videos/{vid}")
    (clip,) = detail["rendered"]
    assert clip["title"] == "Hand picked" and clip["post"]["tiktok"]
    assert detail["review"]["items"][0]["si"] == 1
    status, body, _ = call(port, "GET", f"/media/{vid}/output/{clip['file']}")
    assert status == 200 and len(body) > 10000
