"""`clipper ui`: a local web interface for clipper.

A small HTTP server (standard library only). By default it only answers this computer; with
--phone it also serves other devices on the network that bring the access key. It runs pipeline
jobs one at a time in a worker thread and serves a single-page app from ./static.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import mimetypes
import queue
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..captions import PRESETS
from ..edit import MODES as JUMP_CUT_MODES
from ..pipeline import Options, load_video, process, produce_reviewed
from ..reframe import LAYOUTS
from ..render import ENCODERS
from ..select import split_sentences
from ..util import Cancelled, ClipperError, read_json, set_reporter

STATIC = Path(__file__).with_name("static")
COOKIE = "clipper_key"
LOCKED_PAGE = """<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1">
<title>clipper</title><body style="font:16px/1.5 system-ui,sans-serif;max-width:32em;margin:15vh auto;padding:0 20px">
<h1>clipper</h1><p>This page needs the access link. On the computer running <code>clipper ui --phone</code>,
scan the QR code shown in the terminal (or under <b>Phone</b> in clipper there), or type the link
printed next to it.</p></body>"""
VIDEO_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
MAX_LOG = 400


# ---------------------------------------------------------------------------------------------
# Jobs

@dataclass
class Job:
    kind: str                       # "process" or "render"
    label: str
    run: object                     # callable doing the work
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    state: str = "queued"           # queued | running | done | failed | cancelled
    step: str = ""
    progress: dict | None = None
    log: list = field(default_factory=list)
    error: str | None = None
    video: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self, with_log: bool = False) -> dict:
        out = {k: getattr(self, k) for k in ("id", "kind", "label", "state", "step", "progress", "error",
                                             "video", "created", "finished")}
        out["log_size"] = len(self.log)
        if with_log:
            out["log"] = self.log[-MAX_LOG:]
        return out


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.queue: "queue.Queue[Job]" = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
        self.queue.put(job)
        return job

    def list(self) -> list[dict]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: -j.created)
        return [j.public() for j in jobs[:30]]

    def _sink(self, job: Job):
        def sink(kind: str, payload) -> None:
            if kind == "log":
                job.log.append(payload)
                del job.log[:-MAX_LOG]
            elif kind == "progress":
                job.progress = payload
            elif kind == "step":
                job.step, job.progress = payload, None
            elif kind == "video":
                job.video = payload
        return sink

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            if job.cancel.is_set():
                job.state, job.finished = "cancelled", time.time()
                continue
            job.state = "running"
            set_reporter(self._sink(job), job.cancel)
            try:
                job.run(job)
                job.state = "done"
            except Cancelled:
                job.state = "cancelled"
            except ClipperError as e:
                job.state, job.error = "failed", str(e)
            except Exception as e:  # noqa: BLE001 - show anything unexpected in the UI
                job.state, job.error = "failed", f"{type(e).__name__}: {e}"
                job.log.append(traceback.format_exc())
            finally:
                set_reporter(None)
                job.finished, job.progress = time.time(), None


# ---------------------------------------------------------------------------------------------
# Data for the UI

def choices() -> dict:
    return {
        "defaults": Options().to_dict(),
        "caption_styles": list(PRESETS), "layouts": list(LAYOUTS), "jump_cuts": list(JUMP_CUT_MODES),
        "encoders": list(ENCODERS), "models": ["opus", "sonnet", "haiku"],
        "whisper_models": ["tiny", "base", "small", "medium", "large-v3", "turbo"],
    }


def video_summary(work: Path) -> dict | None:
    meta_path = work / "meta.json"
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    status = "ingested"
    if (work / "transcript.json").exists():
        status = "transcribed"
    if (work / "select.json").exists():
        status = "selected"
    rendered = work / "output" / "clips.json"
    n_rendered = 0
    if rendered.exists():
        n_rendered = sum(1 for c in read_json(rendered) if c.get("file"))
        if n_rendered:
            status = "rendered"
    return {"id": work.name, "title": meta.get("title"), "duration": meta.get("duration"),
            "source": meta.get("source"), "status": status, "rendered": n_rendered,
            "modified": meta_path.stat().st_mtime}


def video_detail(work: Path) -> dict:
    data = load_video(work)
    tpath = work / "transcript.json"
    if tpath.exists():
        transcript = read_json(tpath)["result"]
        data["language"] = transcript.get("language")
        data["sentences"] = [{"i": s.i, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
                             for s in split_sentences(transcript["words"])]
    else:
        data["sentences"] = []
    posts = work / "posts.json"
    data["has_posts"] = posts.exists()
    return data


def frame_jpeg(work: Path, t: float, width: int = 360) -> Path:
    if not (work / "video.mp4").exists():
        return work / "thumb.jpg"  # only the audio was downloaded: use the video's thumbnail
    thumbs = work / "thumbs"
    thumbs.mkdir(exist_ok=True)
    out = thumbs / f"{int(t * 1000)}_{width}.jpg"
    if not out.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(work / "video.mp4"),
                        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(out)],
                       capture_output=True, check=False)
    return out


# ---------------------------------------------------------------------------------------------
# HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "clipper"
    app: "App"  # set on the subclass made in serve()

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    # -- helpers
    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _work(self, vid: str) -> Path | None:
        if not VIDEO_ID.match(vid):
            return None
        work = self.app.root / vid
        return work if (work / "meta.json").exists() else None

    def _file(self, path: Path, cache: bool = False) -> None:
        if not path.is_file():
            return self._error(404, "not found")
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m[1] or m[2]):
            if m[1]:
                start = int(m[1])
                end = min(int(m[2]), size - 1) if m[2] else size - 1
            else:
                start = max(0, size - int(m[2]))
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-cache")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser stopped reading (normal when seeking a video)

    def _given_key(self) -> str:
        key = parse_qs(urlparse(self.path).query).get("key")
        if key:
            return key[0]
        for part in (self.headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE:
                return value
        return ""

    def _authorized(self) -> bool:
        """Requests from other websites are refused (Origin). This computer is trusted when it uses our
        own host name (so DNS rebinding can't get in); anything else needs the --phone access key."""
        host = self.headers.get("Host") or ""
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{host}":
            return False
        port = self.server.server_address[1]
        loopback = ipaddress.ip_address(self.client_address[0]).is_loopback
        if loopback and host in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return True
        key = self.app.key
        return key is not None and hmac.compare_digest(self._given_key().encode(), key.encode())

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        if not self._authorized():
            if urlparse(self.path).path == "/" and self.command == "GET":
                body = LOCKED_PAGE.encode()
                self.send_response(403)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._error(403, "requests must come from the clipper page itself")
            return False
        return True

    # -- routes
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        qs = parse_qs(url.query)
        if not parts:
            if "key" in qs:  # first visit from the phone link: remember the key, then drop it from the URL
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie", f"{COOKIE}={qs['key'][0]}; Path=/; Max-Age=31536000; "
                                               "HttpOnly; SameSite=Lax")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            return self._file(STATIC / "index.html")
        if parts[0] == "static" and len(parts) == 2 and re.match(r"^[\w.-]+$", parts[1]):
            return self._file(STATIC / parts[1])
        if parts[0] == "media" and len(parts) >= 3:
            work = self._work(parts[1])
            rel = parts[2:]
            if work is None or not (rel in (["video.mp4"], ["audio.wav"], ["thumb.jpg"])
                                    or (len(rel) == 2 and rel[0] == "output"
                                                              and re.match(r"^[\w.-]+$", rel[1]))):
                return self._error(404, "not found")
            return self._file(work.joinpath(*rel))
        if parts[0] != "api":
            return self._error(404, "not found")
        route = parts[1:]
        if route == ["choices"]:
            return self._json(choices())
        if route == ["phone"]:
            url = self.app.phone_url
            return self._json({"url": url, "svg": qr_svg(url) if url else None})
        if route == ["videos"]:
            vids = [v for w in sorted(self.app.root.iterdir()) if w.is_dir() and VIDEO_ID.match(w.name)
                    for v in [video_summary(w)] if v]
            return self._json(sorted(vids, key=lambda v: -v["modified"]))
        if len(route) >= 2 and route[0] == "videos":
            work = self._work(route[1])
            if work is None:
                return self._error(404, "no such video")
            if len(route) == 2:
                return self._json(video_detail(work))
            if route[2:] == ["frame"]:
                t = float((qs.get("t") or ["0"])[0])
                return self._file(frame_jpeg(work, max(0.0, t)), cache=True)
        if route == ["jobs"]:
            return self._json(self.app.jobs.list())
        if len(route) == 2 and route[0] == "jobs":
            job = self.app.jobs.jobs.get(route[1])
            return self._json(job.public(with_log=True)) if job else self._error(404, "no such job")
        return self._error(404, "not found")

    def do_POST(self):
        parts = [p for p in urlparse(self.path).path.strip("/").split("/") if p]
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._error(415, "send JSON")  # also keeps other web pages from posting forms here
        if parts == ["api", "jobs"]:
            try:
                job = self.app.make_job(self._body())
            except (ClipperError, ValueError, KeyError, TypeError) as e:
                return self._error(400, str(e))
            return self._json(job.public(), 201)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
            job = self.app.jobs.jobs.get(parts[2])
            if not job:
                return self._error(404, "no such job")
            job.cancel.set()
            return self._json(job.public())
        return self._error(404, "not found")

    def do_PUT(self):
        url = urlparse(self.path)
        if url.path != "/api/upload":
            return self._error(404, "not found")
        name = (parse_qs(url.query).get("name") or ["upload.mp4"])[0]
        name = re.sub(r"[^\w.-]+", "_", Path(name).name).strip("._") or "upload.mp4"
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._error(400, "empty upload")
        dest_dir = self.app.root / "_uploads"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        tmp = dest.with_name(dest.name + ".part")
        with tmp.open("wb") as f:
            left = length
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            tmp.unlink(missing_ok=True)
            return self._error(400, "upload was interrupted")
        tmp.replace(dest)
        return self._json({"path": str(dest)}, 201)


class App:
    def __init__(self, root: Path, key: str | None = None, phone_url: str | None = None):
        self.root = root
        self.jobs = JobManager()
        self.key, self.phone_url = key, phone_url

    def make_job(self, body: dict) -> Job:
        opts = Options.from_dict({**body.get("options", {}), "work_dir": str(self.root)})
        kind = body.get("kind")
        if kind == "process":
            source = str(body.get("source") or "").strip()
            if not source:
                raise ValueError("Give a video file path or URL")
            if body.get("review", True):
                opts.stop_after = "select"

            def run(job: Job) -> None:
                process(source, opts)
            label = source if len(source) < 70 else "…" + source[-68:]
            return self.jobs.submit(Job("process", label, run))
        if kind == "render":
            vid = str(body.get("video") or "")
            work = self.root / vid
            if not VIDEO_ID.match(vid) or not (work / "transcript.json").exists():
                raise ValueError("unknown video")
            items = body.get("items") or []
            if not items:
                raise ValueError("Select at least one clip")
            opts.stop_after = None

            def run(job: Job) -> None:
                job.video = vid
                produce_reviewed(work, items, opts)
            title = read_json(work / "meta.json").get("title") or vid
            return self.jobs.submit(Job("render", f"Render {len(items)} clip(s): {title}", run, video=vid))
        raise ValueError("kind must be 'process' or 'render'")


def lan_ip() -> str | None:
    """This computer's address on the local network (no packets are sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("10.255.255.255", 1))
            ip = sock.getsockname()[0]
        except OSError:
            return None
    return None if ipaddress.ip_address(ip).is_loopback else ip


def phone_key(root: Path, renew: bool = False) -> str:
    """The access key for other devices, kept in the work folder so phone links survive restarts."""
    path = root / ".ui-key"
    if path.exists() and not renew:
        return path.read_text().strip()
    key = secrets.token_urlsafe(18)
    path.write_text(key)
    os.chmod(path, 0o600)
    return key


def qr_svg(text: str) -> str | None:
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return None
    return qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, border=2).to_string(encoding="unicode")


def print_qr(text: str) -> None:
    try:
        import qrcode
    except ImportError:
        return
    qr = qrcode.QRCode(border=2)
    qr.add_data(text)
    qr.make()
    qr.print_ascii(tty=sys.stdout.isatty())


def serve(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="clipper ui")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--work-dir", default="work", help="where results are kept (default ./work)")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    p.add_argument("--phone", action="store_true",
                   help="also open it to phones and other devices on your network (they need the link with "
                        "the access key, shown as a QR code)")
    p.add_argument("--new-key", action="store_true",
                   help="with --phone: make a new access key, so old phone links stop working")
    args = p.parse_args(argv)
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise ClipperError(f"'{tool}' was not found on PATH. Run `clipper doctor` for help.")
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = App(root)
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("0.0.0.0" if args.phone else "127.0.0.1", port), handler)
            break
        except OSError:
            continue
    if server is None:
        raise ClipperError(f"No free port between {args.port} and {args.port + 19}")
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"clipper ui running at {url}  (work folder: {root})  Press Ctrl+C to stop.")
    if args.phone:
        app.key = phone_key(root, args.new_key)
        ip = lan_ip()
        if ip:
            app.phone_url = f"http://{ip}:{port}/?key={app.key}"
            print("\nOn your phone (same Wi-Fi), scan this or open the link:\n")
            print_qr(app.phone_url)
            print(f"  {app.phone_url}\n")
            print("Anyone with this link can use clipper on this computer, so keep it to yourself.\n"
                  "To lock out old links: clipper ui --phone --new-key. If the phone can't connect, allow\n"
                  "incoming connections for Python when macOS asks (System Settings > Network > Firewall).")
        else:
            print("Couldn't find this computer's network address; is it connected to Wi-Fi?")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
