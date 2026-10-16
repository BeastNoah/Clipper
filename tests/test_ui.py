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
    headers = dict(headers or {})
    data = body
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode()
        headers.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=headers)
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

    # a video on this computer isn't converted as a whole: the player gets the original file
    assert meta["video"] == "sections" and meta["local"] and not (work / "video.mp4").exists()
    status, body, r = call(port, "GET", f"/media/{vid}/source", headers={"Range": "bytes=0-99"})
    assert status == 206 and len(body) == 100 and r.getheader("Content-Range").startswith("bytes 0-99/")
    assert r.getheader("Content-Type") == "video/mp4"
    status, body, r = call(port, "GET", f"/api/videos/{vid}/frame?t=2")
    assert status == 200 and r.getheader("Content-Type") == "image/jpeg"

    for path in (f"/media/{vid}/transcript.json", f"/media/{vid}/../{vid}/meta.json", "/media/..%2f..%2fetc/passwd",
                 "/static/../server.py", "/api/videos/NOPE"):
        assert call(port, "GET", path)[0] == 404, path

    status, up, _ = call(port, "PUT", "/api/upload?name=../../evil%20name.mp4", body=b"x" * 1000,
                         headers={"Content-Length": "1000"})
    assert status == 201 and up["path"] == str(work.parent / "_uploads" / "evil_name.mp4")

    # other web pages can't drive the API: wrong Host (DNS rebinding), foreign Origin, form posts
    assert call(port, "GET", "/api/videos", headers={"Host": "evil.example"})[0] == 403
    assert call(port, "POST", "/api/jobs", {"kind": "nope"}, headers={"Origin": "https://evil.example"})[0] == 403
    assert call(port, "POST", "/api/jobs", b'{"kind": "process", "source": "x"}',
                headers={"Content-Type": "text/plain"})[0] == 415

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

    # fix a misheard word in the captions, which re-renders the clip with the correction
    status, caps, _ = call(port, "GET", f"/api/videos/{vid}/captions/1")
    assert status == 200 and not caps["edited"] and caps["lines"][0]["text"].startswith("This is spoken")
    lines = caps["lines"]
    lines[0]["text"] = lines[0]["text"].replace("spoken", "Spoken-Word")
    status, job, _ = call(port, "POST", f"/api/videos/{vid}/captions/1",
                          {"lines": lines, "options": {"caption_model": "none", "preset": "ultrafast"}})
    assert status == 201
    job = wait(port, job["id"])
    assert job["state"] == "done", job
    assert any("using your corrections" in line for line in job["log"])
    ass = next(work.glob("clips/*/9x16/captions.ass")).read_text()
    assert "Spoken-Word" in ass
    assert call(port, "GET", f"/api/videos/{vid}/captions/1")[1]["edited"] is True
    status, job, _ = call(port, "POST", f"/api/videos/{vid}/captions/1",
                          {"reset": True, "options": {"caption_model": "none", "preset": "ultrafast"}})
    assert wait(port, job["id"])["state"] == "done"
    assert "Spoken-Word" not in next(work.glob("clips/*/9x16/captions.ass")).read_text()
    assert call(port, "GET", f"/api/videos/{vid}/captions/9")[0] == 404

    # ask Claude for a new clip of this already transcribed video, made next to the one made by hand
    status, job, _ = call(port, "POST", "/api/jobs", {"kind": "new_clips", "video": vid, "render": True, "options": {
        "num_clips": 1, "min_len": 4, "max_len": 8, "prompt": "jokes", "caption_model": "none", "preset": "ultrafast"}})
    assert status == 201 and job["kind"] == "render" and job["video"] == vid
    job = wait(port, job["id"])
    assert job["state"] == "done", job
    detail = call(port, "GET", f"/api/videos/{vid}")[1]
    (cand,) = detail["candidates"]
    assert cand["focus"] == "jokes" and cand["search"] == 1
    assert [c["title"] for c in detail["rendered"]][0] == "Hand picked"
    assert [c["id"] for c in detail["rendered"]] == ["u1-3", cand["id"]]
    assert all(c["file"] for c in detail["rendered"])
    assert call(port, "POST", "/api/jobs", {"kind": "new_clips", "video": "nope-12345678"})[0] == 400


def wait(port, job_id):
    for _ in range(240):
        job = call(port, "GET", f"/api/jobs/{job_id}")[1]
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.25)
    return job


def test_phone_mode_needs_the_key(tmp_path):
    from clipper.ui.server import lan_ip, phone_key

    ip = lan_ip()
    if ip is None:
        pytest.skip("no network address to test from")
    root = tmp_path / "work"
    work = root / "talk-1234abcd"
    work.mkdir(parents=True)
    write_json(work / "meta.json", {"title": "Talk", "duration": 10})
    (work / "video.mp4").write_bytes(b"v" * 5000)
    key = phone_key(root)
    assert phone_key(root) == key and phone_key(root, renew=True) != key  # kept, unless renewed
    key = phone_key(root)
    httpd = ThreadingHTTPServer(("0.0.0.0", 0), type("H", (Handler,), {"app": App(root, key, "http://x/?key=" + key)}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def lan(method, path, headers=None):
        conn = http.client.HTTPConnection(ip, port, timeout=10)
        conn.request(method, path, headers=headers or {})
        r = conn.getresponse()
        return r.status, r.read(), r

    try:
        status, body, _ = lan("GET", "/")
        assert status == 403 and b"access link" in body           # a friendly page, not the app
        assert lan("GET", "/api/videos")[0] == 403
        assert lan("GET", "/?key=wrong")[0] == 403
        status, _, r = lan("GET", f"/?key={key}")
        assert status == 303 and r.getheader("Location") == "/"
        cookie = r.getheader("Set-Cookie").split(";")[0]
        assert cookie == f"clipper_key={key}" and "HttpOnly" in r.getheader("Set-Cookie")
        status, body, _ = lan("GET", "/api/videos", {"Cookie": cookie})
        assert status == 200 and json.loads(body)[0]["id"] == "talk-1234abcd"
        assert lan("GET", "/media/talk-1234abcd/video.mp4", {"Cookie": cookie, "Range": "bytes=0-9"})[0] == 206
        assert lan("GET", "/api/videos", {"Cookie": cookie, "Origin": "https://evil.example"})[0] == 403
        status, body, _ = lan("GET", "/api/phone", {"Cookie": cookie})
        info = json.loads(body)
        assert info["url"].endswith(key) and info["svg"].startswith("<svg")
        # the computer itself still works without the key
        assert call(port, "GET", "/api/videos")[0] == 200
    finally:
        httpd.shutdown()
