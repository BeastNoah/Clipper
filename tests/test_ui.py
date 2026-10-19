"""The web UI's HTTP API, against a real (tiny) video."""
import http.client
import json
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

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
    up_path = Path(up["path"])
    assert status == 201 and up_path.name == "evil_name.mp4" and up_path.parent.parent == work.parent / "_uploads"
    # a second file with the same name gets its own place; the first one stays as it was
    status, up2, _ = call(port, "PUT", "/api/upload?name=evil%20name.mp4", body=b"y" * 1000,
                          headers={"Content-Type": "application/octet-stream"})
    assert status == 201 and up2["path"] != up["path"]
    assert up_path.read_bytes() == b"x" * 1000 and Path(up2["path"]).read_bytes() == b"y" * 1000

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
    # each line says where it plays in the finished clip, in order and within its length
    clip_len = call(port, "GET", f"/api/videos/{vid}")[1]["rendered"][0]["final_duration"]
    assert all(0 <= l["at"] <= l["until"] <= clip_len + 0.01 for l in lines)
    assert [l["at"] for l in lines] == sorted(l["at"] for l in lines)
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


def test_key_link_only_sets_the_real_key_as_cookie(tmp_path):
    """A crafted ?key= must not add headers or cookies (it is trusted from this computer)."""
    root = tmp_path / "work"
    root.mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"app": App(root, "realkey123")}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        status, _, r = call(port, "GET", "/?key=x%0d%0aSet-Cookie:%20session=evil;%20Path=/")
        assert status == 303 and r.getheader("Location") == "/"
        assert r.headers.get_all("Set-Cookie") is None
        status, _, r = call(port, "GET", "/?key=realkey123")
        assert status == 303 and r.headers.get_all("Set-Cookie") == [
            "clipper_key=realkey123; Path=/; Max-Age=31536000; HttpOnly; SameSite=Lax"]
    finally:
        httpd.shutdown()


def raw(port, request: bytes) -> bytes:
    import socket
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request)
        out = b""
        while chunk := sock.recv(65536):
            out += chunk
    return out


def test_bad_requests_get_an_answer(server):
    port, work, _ = server
    vid = work.name
    for body in ([1, 2], "text", 3, None):
        status, reply, _ = call(port, "POST", "/api/jobs", json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
        assert status == 400 and "error" in reply, body
    assert call(port, "POST", f"/api/videos/{vid}/captions/1", b"[]",
                headers={"Content-Type": "application/json"})[0] in (400, 404)
    for length in (b"abc", b"-5"):
        reply = raw(port, b"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
                          b"Content-Length: %s\r\n\r\n{}" % (port, length))
        assert reply.startswith(b"HTTP/1.0 400") or reply.startswith(b"HTTP/1.1 400"), reply[:80]
    reply = raw(port, b"PUT /api/upload HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Length: x\r\n\r\n" % port)
    assert b" 400 " in reply.split(b"\r\n")[0]
    for t in ("abc", "inf", "nan", "-inf"):
        assert call(port, "GET", f"/api/videos/{vid}/frame?t={t}")[0] == 400, t


def test_cancelling_a_queued_job_is_immediate():
    from clipper.ui.server import Job, JobManager
    jobs = JobManager()
    release, ran = threading.Event(), []
    first = jobs.submit(Job("process", "first", lambda job: release.wait(10)))
    second = jobs.submit(Job("process", "second", lambda job: ran.append(job)))
    for _ in range(100):
        if first.state == "running":
            break
        time.sleep(0.02)
    jobs.cancel(second)
    assert second.state == "cancelled" and second.finished
    release.set()
    for _ in range(100):
        if first.state == "done":
            break
        time.sleep(0.02)
    time.sleep(0.1)
    assert first.state == "done" and second.state == "cancelled" and not ran


def test_publishing_routes(server, tmp_path, monkeypatch):
    import urllib.parse

    import clipper.publish as pub

    port, work, _ = server
    vid = work.name
    monkeypatch.setenv("CLIPPER_CONFIG", str(tmp_path / "cfg"))
    status, accts, _ = call(port, "GET", "/api/accounts")
    assert status == 200 and [a["name"] for a in accts] == ["youtube", "tiktok", "instagram"]
    assert not any(a["configured"] or a["connected"] for a in accts)

    status, body, _ = call(port, "POST", "/api/accounts/youtube/connect", {})
    assert status == 400 and "isn't set up" in body["error"]
    assert call(port, "POST", "/api/accounts/youtube/app", {"client_id": "cid", "client_secret": "sec"})[0] == 200
    status, body, _ = call(port, "POST", "/api/accounts/youtube/connect", {})
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(body["url"]).query))
    assert status == 200 and q["redirect_uri"] == f"http://localhost:{port}/oauth/youtube/callback"
    assert q["client_id"] == "cid" and q["state"]
    # a callback that this server didn't start, or that the user declined, connects nothing
    status, page, _ = call(port, "GET", "/oauth/youtube/callback?state=forged&code=x")
    assert status == 400 and b"unknown or too old" in page
    status, page, _ = call(port, "GET", "/oauth/youtube/callback?error=access_denied&error_description=%3Cb%3Eno%3C/b%3E")
    assert status == 400 and b"&lt;b&gt;no&lt;/b&gt;" in page  # escaped
    assert call(port, "POST", "/api/accounts/nope/app", {})[0] == 400

    out = work / "output"
    out.mkdir(exist_ok=True)
    (out / "01_hook.mp4").write_bytes(b"v" * 1000)
    clip = {"rank": 1, "id": "c1", "title": "The hook", "file": "01_hook.mp4", "start": 1.0, "end": 9.0,
            "post": {"tiktok": "Watch this", "instagram": "New reel", "youtube_title": "The hook, explained",
                     "youtube_description": "More soon.", "hashtags": ["clips", "talk"]}}
    write_json(out / "clips.json", [clip])
    status, data, _ = call(port, "GET", f"/api/videos/{vid}/publish/1")
    assert status == 200 and data["published"] == {}
    assert data["defaults"]["youtube"] == {"title": "The hook, explained", "description": "More soon.\n\n#clips #talk"}
    assert data["defaults"]["tiktok"]["title"] == "Watch this\n\n#clips #talk"
    assert call(port, "GET", f"/api/videos/{vid}/publish/9")[0] == 404

    calls = []

    def fake_publish(w, c, name, form, again=False):
        calls.append((w, c["file"], name, form, again))
        return {"id": "x", "status": "published"}

    monkeypatch.setattr(pub, "publish_clip", fake_publish)
    status, job, _ = call(port, "POST", f"/api/videos/{vid}/publish/1",
                          {"platform": "tiktok", "form": {"privacy": "SELF_ONLY"}, "again": True})
    assert status == 201 and job["kind"] == "publish" and job["label"] == "Post to TikTok: The hook"
    for _ in range(50):
        state = call(port, "GET", f"/api/jobs/{job['id']}")[1]["state"]
        if state not in ("queued", "running"):
            break
        time.sleep(0.1)
    assert state == "done" and calls == [(work, "01_hook.mp4", "tiktok", {"privacy": "SELF_ONLY"}, True)]
    assert any(j["id"] == job["id"] for j in call(port, "GET", "/api/jobs")[1])
    assert call(port, "POST", f"/api/videos/{vid}/publish/1", {"platform": "myspace"})[0] == 400


def test_campaigns_and_posting_several_clips(server, tmp_path, monkeypatch):
    import clipper.publish as pub

    port, work, _ = server
    vid = work.name
    monkeypatch.setenv("CLIPPER_CONFIG", str(tmp_path / "cfg"))
    assert call(port, "GET", "/api/campaigns")[1] == []
    status, item, _ = call(port, "POST", "/api/campaigns", {"name": "", "brief": "BrandX: tag @brandx", "model": "haiku"})
    assert status == 201 and item["name"] == "BrandX" and item["rules"]["mentions"] == ["@brandx"]
    assert "Use a trending sound" in item["checklist"]
    assert [c["id"] for c in call(port, "GET", "/api/campaigns")[1]] == [item["id"]]
    assert call(port, "POST", "/api/campaigns", {"brief": " "})[0] == 400
    assert call(port, "POST", f"/api/campaigns/{item['id']}/delete", {})[1] == []

    out = work / "output"
    out.mkdir(exist_ok=True)
    clips = []
    for n in (1, 2):
        (out / f"0{n}_c.mp4").write_bytes(b"v" * 100)
        clips.append({"rank": n, "id": f"c{n}", "title": f"Clip {n}", "file": f"0{n}_c.mp4", "start": 1.0, "end": 9.0})
    write_json(out / "clips.json", clips)
    write_json(work / "campaign.json", {"brief": "b", "rules": item["rules"]})
    write_json(out / "published.json", {"02_c.mp4": {"tiktok": {"status": "published", "url": "https://t/1"}}})
    detail = call(port, "GET", f"/api/videos/{vid}")[1]
    assert detail["campaign"]["rules"]["name"] == "BrandX" and detail["campaign"]["checklist"]
    assert call(port, "GET", f"/api/videos/{vid}/publish/1")[1]["campaign"]["rules"]["disclosure"] is True

    calls = []
    monkeypatch.setattr(pub, "publish_clip", lambda w, c, name, form, again=False: calls.append((c["rank"], name, form)) or {})
    posts = [{"rank": r, "platform": p, "form": {"title": f"t{r}"}} for r in (1, 2) for p in ("tiktok", "youtube")]
    status, jobs, _ = call(port, "POST", f"/api/videos/{vid}/publish-all", {"posts": posts})
    assert status == 201 and len(jobs) == 3  # clip 2 is already on TikTok
    for _ in range(50):
        if len(calls) == 3:
            break
        time.sleep(0.1)
    assert calls == [(1, "tiktok", {"title": "t1"}), (1, "youtube", {"title": "t1"}), (2, "youtube", {"title": "t2"})]
    for bad in ({"posts": []}, {"posts": [{"rank": 9, "platform": "tiktok", "form": {}}]},
                {"posts": [{"rank": 1, "platform": "myspace", "form": {}}]},
                {"posts": [{"rank": 2, "platform": "tiktok", "form": {}}]}):
        assert call(port, "POST", f"/api/videos/{vid}/publish-all", bad)[0] == 400
