"""`clipper app`: make a Clipper.app that opens the web interface with a double-click.

The app is a tiny shell-script bundle. It remembers the PATH, the clipper install and the work
folder of the terminal it was made from (apps started from the Finder get a bare PATH without
Homebrew, so ffmpeg and claude would otherwise be missing). Opening it first pulls the newest
clipper from GitHub (when clipper runs from a git checkout), then starts `clipper ui` in the
background, or, when that is already running, just opens its page again. When the pull brought
new code and the running interface is idle, it is restarted so the new code takes effect.
"""

from __future__ import annotations

import argparse
import importlib.util
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
export GIT_TERMINAL_PROMPT=0
WORK={work}
REPO={repo}
mkdir -p "$WORK"
LOG="$WORK/.update.log"

# Pull the newest clipper from GitHub. Succeeds only when that changed the code.
update() {{
    [ -n "$REPO" ] && [ -d "$REPO/.git" ] || return 1
    before=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || return 1
    echo "$(date) checking GitHub for a newer clipper" >> "$LOG"
    if ! git -C "$REPO" pull --ff-only --quiet >> "$LOG" 2>&1; then
        echo "update failed (no network, or local changes in $REPO); using the version already here" >> "$LOG"
        return 1
    fi
    after=$(git -C "$REPO" rev-parse HEAD)
    [ "$before" != "$after" ] || return 1
    echo "updated $before -> $after" >> "$LOG"
    if ! git -C "$REPO" diff --quiet "$before" "$after" -- pyproject.toml; then
        echo "dependencies changed, installing" >> "$LOG"
        {pip} install --quiet -e "$REPO{extras}" >> "$LOG" 2>&1 || echo "install failed" >> "$LOG"
    fi
    return 0
}}

URL=$(cat "$WORK/.ui-url" 2>/dev/null)
RUNNING=""
if [ -n "$URL" ] && curl -s -o /dev/null --max-time 2 "$URL"; then
    RUNNING=1
fi
if update && [ -n "$RUNNING" ]; then
    # New code only takes effect in a fresh interface. Restart the one we started, unless it is busy.
    PID=$(cat "$WORK/.ui.pid" 2>/dev/null)
    if curl -s --max-time 2 "$URL/api/jobs" | grep -Eq '"state": ?"(queued|running)"'; then
        echo "the interface is busy; the new version starts next time" >> "$LOG"
    elif [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "restarting the interface" >> "$LOG"
        kill "$PID"
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
        RUNNING=""
    else
        echo "the interface was not started by this app; restart it yourself for the new version" >> "$LOG"
    fi
fi
if [ -n "$RUNNING" ]; then
    open "$URL"
    exit 0
fi
cd "$WORK"
nohup {clipper} ui --work-dir "$WORK" {extra}> "$WORK/.ui.log" 2>&1 &
echo $! > "$WORK/.ui.pid"
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


def repo_dir() -> Path | None:
    """The git checkout this clipper runs from (an editable install), or None."""
    root = Path(__file__).resolve().parents[1]
    return root if (root / ".git").exists() else None


def pip_command() -> str:
    return f"{shlex.quote(sys.executable)} -m pip"


def install_extras() -> str:
    """The extras the install was made with, as far as can be told: [mac] when mlx-whisper is in."""
    return "[mac]" if importlib.util.find_spec("mlx_whisper") else ""


def build_app(dest: Path, work: Path, phone: bool = False, repo: Path | None = None) -> Path:
    """repo: the git checkout to pull from on every start; None for an app that never updates."""
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
        repo=shlex.quote(str(repo)) if repo else '""',
        pip=pip_command(),
        extras=install_extras(),
    ))
    script.chmod(0o755)
    return app


def make_app(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="clipper app",
                                description="Make a Clipper.app that opens the web interface with a double-click.")
    p.add_argument("--work-dir", default="work", help="where results are kept (default ./work)")
    p.add_argument("--dest", default="~/Applications", help="where to put Clipper.app (default ~/Applications)")
    p.add_argument("--phone", action="store_true", help="the app starts the interface with --phone")
    p.add_argument("--no-update", action="store_true",
                   help="don't pull the newest clipper from GitHub each time the app is opened")
    args = p.parse_args(argv)
    if sys.platform != "darwin":
        raise ClipperError("`clipper app` makes a macOS app; on other systems use `clipper ui`.")
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    repo = None if args.no_update else repo_dir()
    app = build_app(dest, Path(args.work_dir).resolve(), args.phone, repo)
    print(f"Made {app}\n"
          "Open it from Launchpad, Spotlight (Cmd+Space, \"Clipper\") or drag it into the Dock.\n"
          "The interface keeps running in the background; opening the app again just shows it.\n"
          + (f"Each time it opens it first pulls the newest clipper into {repo} (log: {args.work_dir}/.update.log).\n"
             if repo else "")
          + "If you move the clipper folder or change your PATH, run `clipper app` again.")
    return 0
