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


# ---------------------------------------------------------------------------------------------
# Self-update: the app pulls from GitHub (here: a local bare repository) each time it is opened

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


def _git(cwd, *args):
    return subprocess.run(GIT + list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A clone (what clipper runs from) of a bare "GitHub" repository, plus a way to publish a change."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    author = tmp_path / "author"
    _git(tmp_path, "clone", "-q", str(origin), str(author))
    (author / "pyproject.toml").write_text("[project]\nname = 'clipper'\n")
    (author / "code.py").write_text("v1\n")
    _git(author, "add", "."), _git(author, "commit", "-q", "-m", "v1"), _git(author, "push", "-q", "-u", "origin", "main")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))

    def publish(**files):
        for name, text in files.items():
            (author / name).write_text(text)
        _git(author, "add", "."), _git(author, "commit", "-q", "-m", "update"), _git(author, "push", "-q")
        return _git(author, "rev-parse", "HEAD")

    return clone, publish


@pytest.fixture
def updating(tmp_path, monkeypatch, repo):
    """Like `made`, but the app pulls into the clone; pip is a stub that records its calls."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "calls.txt"
    _stub(bin_ / "open", f'echo "open $*" >> {calls}\n')
    _stub(bin_ / "osascript", f'echo "alert" >> {calls}\n')
    _stub(bin_ / "clipper", f'echo "clipper $*" >> {calls}\nsleep 5\n')
    _stub(bin_ / "pip", f'echo "pip $*" >> {calls}\n')
    monkeypatch.setenv("PATH", f"{bin_}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")
    monkeypatch.setattr(app_mod, "clipper_command", lambda: str(bin_ / "clipper"))
    monkeypatch.setattr(app_mod, "pip_command", lambda: str(bin_ / "pip"))
    monkeypatch.setattr(app_mod, "install_extras", lambda: "[mac]")
    clone, publish = repo
    work = tmp_path / "work"
    app = app_mod.build_app(tmp_path, work, repo=clone)
    return app / "Contents" / "MacOS" / "clipper", work, calls, clone, publish


def _run(launcher):
    subprocess.run([str(launcher)], check=True, timeout=30)


def _serve(work, jobs_body):
    """A stand-in for a running interface, registered in the work folder like the real one."""
    class Ui(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path.startswith("/api/jobs"):
                self.wfile.write(jobs_body.encode())

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Ui)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    work.mkdir(exist_ok=True)
    (work / ".ui-url").write_text(url)
    return server, url


def test_pulls_the_newest_version_before_starting(updating):
    launcher, work, calls, clone, publish = updating
    old = _git(clone, "rev-parse", "HEAD")
    new = publish(**{"code.py": "v2\n"})
    _run(launcher)
    assert _git(clone, "rev-parse", "HEAD") == new != old
    assert (clone / "code.py").read_text() == "v2\n"
    assert calls.read_text().splitlines() == [f"clipper ui --work-dir {work}"]  # no pip: pyproject unchanged
    assert (work / ".ui.pid").read_text().strip().isdigit()
    assert f"updated {old} -> {new}" in (work / ".update.log").read_text()


def test_reinstalls_when_the_dependencies_change(updating):
    launcher, work, calls, clone, publish = updating
    publish(**{"pyproject.toml": "[project]\nname = 'clipper'\ndependencies = ['x']\n"})
    _run(launcher)
    assert calls.read_text().splitlines() == [f"pip install --quiet -e {clone}[mac]", f"clipper ui --work-dir {work}"]


def test_starts_anyway_when_the_pull_fails(updating):
    launcher, work, calls, clone, publish = updating
    _git(clone, "remote", "set-url", "origin", str(clone.parent / "gone.git"))
    _run(launcher)
    assert calls.read_text().splitlines() == [f"clipper ui --work-dir {work}"]
    assert "update failed" in (work / ".update.log").read_text()


def test_restarts_an_idle_interface_after_an_update(updating):
    launcher, work, calls, clone, publish = updating
    server, url = _serve(work, "[]")
    stale = subprocess.Popen(["sleep", "60"])  # stands in for the interface this app started earlier
    (work / ".ui.pid").write_text(str(stale.pid))
    try:
        publish(**{"code.py": "v2\n"})
        _run(launcher)
        assert stale.wait(timeout=5) != 0  # was told to stop
        assert calls.read_text().splitlines() == [f"clipper ui --work-dir {work}"]
        assert (work / ".ui.pid").read_text().strip() != str(stale.pid)
    finally:
        server.shutdown()
        stale.kill()


def test_leaves_a_busy_interface_alone(updating):
    launcher, work, calls, clone, publish = updating
    server, url = _serve(work, '[{"id": "a", "state": "running"}]')
    busy = subprocess.Popen(["sleep", "60"])
    (work / ".ui.pid").write_text(str(busy.pid))
    try:
        new = publish(**{"code.py": "v2\n"})
        _run(launcher)
        assert busy.poll() is None  # still running
        assert calls.read_text().splitlines() == [f"open {url}"]
        assert _git(clone, "rev-parse", "HEAD") == new  # the code is already pulled for next time
        assert "busy" in (work / ".update.log").read_text()
    finally:
        server.shutdown()
        busy.kill()


def test_nothing_new_just_opens_the_page(updating):
    launcher, work, calls, clone, publish = updating
    server, url = _serve(work, "[]")
    try:
        _run(launcher)
        assert calls.read_text().splitlines() == [f"open {url}"]
    finally:
        server.shutdown()
