"""Clipper.app's launcher script (`clipper app`), run with stand-ins for clipper and macOS's `open`."""
import plistlib
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from clipper import app as app_mod

pytestmark = pytest.mark.skipif(not shutil.which("curl"), reason="needs curl")


def _stub(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


@pytest.fixture
def made(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "calls.txt"
    _stub(bin_ / "open", f'echo "open $*" >> {calls}\n')
    _stub(bin_ / "osascript", f'echo "alert" >> {calls}\n')
    fake = bin_ / "clipper"
    _stub(fake, f'echo "clipper $*" >> {calls}\nsleep 5\n')
    monkeypatch.setenv("PATH", f"{bin_}:/usr/bin:/bin")
    monkeypatch.setattr(app_mod, "clipper_command", lambda: str(fake))
    work = tmp_path / "my work"
    app = app_mod.build_app(tmp_path, work)
    return app, work, calls


def test_bundle_layout(made):
    app, _, _ = made
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "clipper"
    assert (app / "Contents" / "MacOS" / "clipper").stat().st_mode & 0o111


def test_starts_the_interface_when_none_is_running(made):
    app, work, calls = made
    subprocess.run([str(app / "Contents" / "MacOS" / "clipper")], check=True, timeout=20)
    assert calls.read_text().splitlines() == [f"clipper ui --work-dir {work}"]


def test_reopens_a_running_interface(made):
    app, work, calls = made

    class Ok(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        work.mkdir()
        (work / ".ui-url").write_text(url)
        subprocess.run([str(app / "Contents" / "MacOS" / "clipper")], check=True, timeout=20)
        assert calls.read_text().splitlines() == [f"open {url}"]
    finally:
        server.shutdown()


def test_reports_a_failed_start(made, tmp_path):
    app, work, calls = made
    _stub(tmp_path / "bin" / "clipper", "echo \"ffmpeg was not found\" >&2\nexit 1\n")
    subprocess.run([str(app / "Contents" / "MacOS" / "clipper")], check=True, timeout=20)
    assert calls.read_text().splitlines() == ["alert"]
    assert "ffmpeg" in (work / ".ui.log").read_text()
