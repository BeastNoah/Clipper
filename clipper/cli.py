"""Command line entry point: runs the six steps in order, reusing whatever is already in work/<video_id>/."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .captions import make_captions
from .ingest import ingest
from .llm import Claude
from .reframe import reframe
from .render import check_ffmpeg_has_libass, render
from .select import select_clips
from .transcribe import transcribe
from .util import ClipperError, fmt_time, log, require_tool, slugify, write_json

STEPS = ["ingest", "transcribe", "select", "reframe", "captions", "render"]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="clipper", description="Turn one long video into short vertical clips.")
    p.add_argument("source", help="a local video file or a URL (anything yt-dlp can download)")
    p.add_argument("-n", "--num-clips", type=int, default=3, help="how many clips to make (default 3)")
    p.add_argument("--min-len", type=float, default=20, help="shortest clip in seconds (default 20)")
    p.add_argument("--max-len", type=float, default=60, help="longest clip in seconds (default 60)")
    p.add_argument("--work-dir", type=Path, default=Path("work"), help="where results are kept (default ./work)")
    p.add_argument("--whisper-model", default="small",
                   help="faster-whisper model: tiny, base, small, medium, large-v3 ... (default small)")
    p.add_argument("--language", default=None, help="spoken language code, e.g. en or de (default: detect)")
    p.add_argument("--model", default="opus", help="Claude model used to pick clips (default opus)")
    p.add_argument("--font", default="Arial Black", help="caption font (default Arial Black)")
    p.add_argument("--crf", type=int, default=20, help="x264 quality, lower is better (default 20)")
    p.add_argument("--preset", default="medium", help="x264 speed preset (default medium)")
    p.add_argument("--force", choices=STEPS, metavar="STEP",
                   help=f"redo this step and every step after it ({', '.join(STEPS)})")
    p.add_argument("--stop-after", choices=STEPS, metavar="STEP", help="stop after this step")
    args = p.parse_args(argv)
    if args.num_clips < 1:
        p.error("--num-clips must be at least 1")
    if args.min_len <= 0 or args.max_len < args.min_len:
        p.error("--min-len must be positive and no larger than --max-len")
    return args


def _publish(src: Path, dest: Path) -> None:
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def run(args: argparse.Namespace) -> list[dict]:
    for tool in ("ffmpeg", "ffprobe"):
        require_tool(tool)

    def forced(step: str) -> bool:
        return args.force is not None and STEPS.index(step) >= STEPS.index(args.force)

    def stop(step: str) -> bool:
        return args.stop_after == step

    work, meta = ingest(args.source, args.work_dir, force=forced("ingest"))
    if stop("ingest"):
        return []
    transcript = transcribe(work, meta, args.whisper_model, args.language, force=forced("transcribe"))
    if stop("transcribe"):
        return []
    clips = select_clips(
        work, meta, transcript, num_clips=args.num_clips, min_len=args.min_len, max_len=args.max_len,
        model=args.model, claude_factory=lambda: Claude(args.model, cwd=work), force=forced("select"),
    )
    if not clips:
        raise ClipperError("Claude didn't find any usable clips in this video.")
    if stop("select"):
        return clips

    do_render = args.stop_after not in ("reframe", "captions")
    if do_render:
        check_ffmpeg_has_libass()
    out_dir = work / "output"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.mp4"):
        old.unlink()

    for clip in clips:
        clip_dir = work / "clips" / f"{round(clip['start'] * 1000):08d}-{round(clip['end'] * 1000):08d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        plan = reframe(work, meta, clip, clip_dir, force=forced("reframe"))
        ass = make_captions(transcript, clip, clip_dir, args.font)
        if not do_render:
            continue
        mp4 = render(work, meta, clip, plan, ass, clip_dir, crf=args.crf, preset=args.preset,
                     force=forced("captions"))
        final = out_dir / f"{clip['rank']:02d}_{slugify(clip['title'])}.mp4"
        _publish(mp4, final)
        clip["file"] = str(final)

    write_json(out_dir / "clips.json", clips)
    if do_render:
        log(f"done: {len(clips)} clip(s) in {out_dir}")
        for c in clips:
            print(f"{c['rank']:2d}. [{c['score']:3d}] {fmt_time(c['start'])}-{fmt_time(c['end'])}  "
                  f"{c['title']}\n    {c['file']}")
    return clips


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except ClipperError as e:
        print(f"clipper: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("clipper: interrupted; finished steps are saved and will be reused", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
