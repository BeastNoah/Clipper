"""`clipper app`: make a Clipper.app that opens the web interface with a double-click.

The app is a tiny shell-script bundle. It remembers the PATH, the clipper install and the work
folder of the terminal it was made from (apps started from the Finder get a bare PATH without
Homebrew, so ffmpeg and claude would otherwise be missing). Opening it starts `clipper ui` in the
background, or, when that is already running, just opens its page again.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shlex
import shutil
import sys
from pathlib import Path

from .util import ClipperError

LAUNCHER = """\
#!/bin/bash
# Made by `clipper app`. Run it again to update this app.
export PATH={path}
WORK={work}
mkdir -p "$WORK"
URL=$(cat "$WORK/.ui-url" 2>/dev/null)
if [ -n "$URL" ] && curl -s -o /dev/null --max-time 2 "$URL"; then
    open "$URL"
    exit 0
fi
cd "$WORK"
nohup {clipper} ui --work-dir "$WORK" {extra}> "$WORK/.ui.log" 2>&1 &
# If it stops right away (ffmpeg missing, ...), say so instead of failing silently.
sleep 3
if ! kill -0 $! 2>/dev/null; then
    MSG=$(tail -n 3 "$WORK/.ui.log" | tr '"' "'")
    osascript -e "display alert \\"clipper didn't start\\" message \\"$MSG\\n\\nDetails: $WORK/.ui.log\\""
fi
"""


def clipper_command() -> str:
    """How to start this same clipper (the venv's script, or this Python as a fallback)."""
    exe = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if exe and exe.name == "clipper" and exe.is_file():
        return shlex.quote(str(exe))
    return f"{shlex.quote(sys.executable)} -m clipper"


def build_app(dest: Path, work: Path, phone: bool = False) -> Path:
    app = dest / "Clipper.app"
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump({
            "CFBundleName": "Clipper",
            "CFBundleDisplayName": "Clipper",
            "CFBundleIdentifier": "local.clipper.app",
            "CFBundleExecutable": "clipper",
            "CFBundlePackageType": "APPL",
            "CFBundleVersion": "1",
            "LSUIElement": True,  # no Dock icon: the app only starts the server and exits
        }, f)
    script = macos / "clipper"
    script.write_text(LAUNCHER.format(
        path=shlex.quote(os.environ.get("PATH", "/usr/bin:/bin")),
        work=shlex.quote(str(work)),
        clipper=clipper_command(),
        extra="--phone " if phone else "",
    ))
    script.chmod(0o755)
    return app


def make_app(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="clipper app",
                                description="Make a Clipper.app that opens the web interface with a double-click.")
    p.add_argument("--work-dir", default="work", help="where results are kept (default ./work)")
    p.add_argument("--dest", default="~/Applications", help="where to put Clipper.app (default ~/Applications)")
    p.add_argument("--phone", action="store_true", help="the app starts the interface with --phone")
    args = p.parse_args(argv)
    if sys.platform != "darwin":
        raise ClipperError("`clipper app` makes a macOS app; on other systems use `clipper ui`.")
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    app = build_app(dest, Path(args.work_dir).resolve(), args.phone)
    print(f"Made {app}\n"
          "Open it from Launchpad, Spotlight (Cmd+Space, \"Clipper\") or drag it into the Dock.\n"
          "The interface keeps running in the background; opening the app again just shows it.\n"
          "If you move the clipper folder or change your PATH, run `clipper app` again.")
    return 0
