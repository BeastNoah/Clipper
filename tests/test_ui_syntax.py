"""The web page's script must at least parse: a syntax error leaves the page blank and unresponsive."""
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parents[1] / "clipper" / "ui" / "static"
APP_JS = STATIC / "app.js"
NODE = shutil.which("node")
JSC = shutil.which("jsc") or "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"


@pytest.mark.skipif(not NODE and not Path(JSC).exists(), reason="needs node or JavaScriptCore (jsc)")
@pytest.mark.parametrize("name", ["app.js", "editor.js"])
def test_app_js_parses(name):
    path = STATIC / name
    if NODE:
        run = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
    else:  # jsc has no check mode: parse it as a function body without running it
        script = f"new Function({path.read_text()!r});"
        run = subprocess.run([JSC, "-e", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
