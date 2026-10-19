"""Command line: `clipper SOURCE...`, `clipper ui`, `clipper app` and `clipper doctor`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .captions import PRESETS
from .edit import MODES as JUMP_CUT_MODES
from .ingest import expand_sources
from .pipeline import STEPS, Options, process
from .reframe import LAYOUTS
from .render import ENCODERS
from .select import clip_spans
from .util import ClipperError, fmt_time, log

USAGE = """\
clipper SOURCE [SOURCE ...] [options]   make clips from files, URLs or playlists
clipper ui [--port N] [--work-dir DIR]   open the web interface
clipper trending [--refresh] [--all]     what the creators you clip for posted lately, best first
clipper app                              make a Clipper.app (macOS) that opens the interface
clipper doctor                           check that everything clipper needs is installed"""


def _bool(value: str) -> bool:
    return value.lower() in ("1", "yes", "on", "true")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    d = Options()
    p = argparse.ArgumentParser(prog="clipper", usage=USAGE,
                                description="Turn long videos into short vertical clips.")
    p.add_argument("sources", nargs="+", metavar="SOURCE",
                   help="video files or URLs (anything yt-dlp can download, including playlists)")
    g = p.add_argument_group("clips")
    g.add_argument("-n", "--num-clips", type=int, default=d.num_clips, help="clips per video (default 3)")
    g.add_argument("--min-len", type=float, default=d.min_len, help="shortest clip in seconds (default 20)")
    g.add_argument("--max-len", type=float, default=d.max_len, help="longest clip in seconds (default 60)")
    g.add_argument("--model", default=d.model, help="Claude model that picks the clips (default opus)")
    g.add_argument("--prompt", default=None, metavar="TEXT",
                   help="what Claude should look for, e.g. 'funny moments' or 'everything about pricing'")
    g.add_argument("--new-clips", action="store_true",
                   help="ask for clips at moments Claude hasn't suggested before; they're made alongside the "
                        "clips already made (for a video that's been through clipper before)")
    g.add_argument("--no-earlier-parts", action="store_true",
                   help="don't let Claude put earlier parts of the video (a joke's set-up, the question being "
                        "answered) in front of a clip")
    g.add_argument("--from", dest="range_start", default=None, metavar="TIME",
                   help="only use the video from this time on (seconds, 12:30 or 1:02:03)")
    g.add_argument("--to", dest="range_end", default=None, metavar="TIME", help="only use the video up to this time")
    g = p.add_argument_group("speech")
    g.add_argument("--whisper-model", default=d.whisper_model,
                   help="model for the full transcript: tiny, base, small, medium, large-v3, turbo (default small)")
    g.add_argument("--caption-model", default=d.caption_model,
                   help="bigger model that re-transcribes just the chosen clips for accurate captions, "
                        "or 'none' (default turbo)")
    g.add_argument("--caption-language", default=None, metavar="LANG",
                   help="translate the captions, e.g. en or German (also writes a translated .srt)")
    g.add_argument("--whisper-backend", default=d.whisper_backend, choices=("auto", "faster-whisper", "mlx"),
                   help="auto uses mlx-whisper on Apple Silicon when installed (default auto)")
    g.add_argument("--language", default=None, help="spoken language code, e.g. en or de (default: detect)")
    g = p.add_argument_group("look")
    g.add_argument("--caption-style", default=d.caption_style, choices=tuple(PRESETS), help="(default bold)")
    g.add_argument("--font", default=None, help="caption font (default depends on the style)")
    g.add_argument("--caption-size", type=int, default=None, help="caption font size in pixels")
    g.add_argument("--highlight-color", default=None, help="colour of the spoken word, e.g. '#FFE600'")
    g.add_argument("--uppercase", action="store_true", help="captions in capital letters")
    g.add_argument("--caption-position", default=d.caption_position, choices=("lower", "middle"))
    g.add_argument("--no-title", action="store_true", help="don't show the clip title at the start")
    g.add_argument("--format", dest="formats", default="9:16", metavar="RATIOS",
                   help="output shapes, comma separated: 9:16, 4:5, 1:1, 16:9 (default 9:16)")
    g.add_argument("--no-zoom", action="store_true", help="no automatic zoom-ins on the hook and keywords")
    g.add_argument("--progress-bar", action="store_true", help="a progress bar along the top")
    g.add_argument("--logo", default=None, metavar="FILE", help="your logo on every clip (PNG with transparency)")
    g.add_argument("--logo-position", default=d.logo_position,
                   choices=("top-left", "top-right", "bottom-left", "bottom-right"))
    g.add_argument("--jump-cuts", default=d.jump_cuts, choices=JUMP_CUT_MODES,
                   help="cut out long pauses and filler words (on), only pauses, or nothing (default on)")
    g.add_argument("--layout", default=d.layout, choices=LAYOUTS,
                   help="two people in a shot: follow whoever talks, split screen, auto, or only the "
                        "main person (default auto)")
    g = p.add_argument_group("output")
    g.add_argument("--encoder", default=d.encoder, choices=ENCODERS,
                   help="auto uses the Mac's hardware encoder (videotoolbox: much faster, slightly bigger "
                        "files) when there is one, else x264 (best quality per MB) (default auto)")
    g.add_argument("--crf", type=int, default=d.crf, help="x264 quality, lower is better (default 20)")
    g.add_argument("--preset", default=d.preset, help="x264 speed preset (default medium)")
    g.add_argument("--no-posts", action="store_true", help="don't write post text and hashtags")
    g.add_argument("--download", default=d.download, choices=("sections", "full"),
                   help="'sections' takes only the sound at first, then just the video of the chosen "
                        "clips (much faster for long videos); 'full' downloads and converts the whole video "
                        "first (default sections)")
    g.add_argument("--work-dir", default=d.work_dir, help="where results are kept (default ./work)")
    g.add_argument("--force", choices=STEPS, metavar="STEP",
                   help=f"redo this step and every step after it ({', '.join(STEPS)})")
    g.add_argument("--stop-after", choices=STEPS, metavar="STEP", help="stop after this step")
    return p.parse_args(argv)


def options_from_args(args: argparse.Namespace) -> Options:
    return Options(
        num_clips=args.num_clips, min_len=args.min_len, max_len=args.max_len, work_dir=args.work_dir,
        download=args.download, prompt=args.prompt, range_start=args.range_start, range_end=args.range_end,
        caption_language=args.caption_language, formats=args.formats, zoom=not args.no_zoom,
        progress_bar=args.progress_bar, logo=args.logo, logo_position=args.logo_position,
        model=args.model, whisper_model=args.whisper_model, whisper_backend=args.whisper_backend,
        language=args.language, caption_model=args.caption_model, caption_style=args.caption_style,
        font=args.font, caption_size=args.caption_size, highlight_color=args.highlight_color,
        uppercase=args.uppercase, caption_position=args.caption_position, title_overlay=not args.no_title,
        earlier_parts=not args.no_earlier_parts, jump_cuts=args.jump_cuts, layout=args.layout, encoder=args.encoder, crf=args.crf, preset=args.preset,
        posts=not args.no_posts, force=args.force, stop_after=args.stop_after,
    ).validate()


def _summary(work: Path, clips: list[dict]) -> None:
    out_dir = work / "output"
    for c in clips:
        line = f"{c['rank']:2d}. [{c['score'] if c.get('score') is not None else '-':>3}] " \
               f"{' + '.join(f'{fmt_time(s)}-{fmt_time(e)}' for s, e in clip_spans(c))}  {c['title']}"
        if c.get("file"):
            line += f"\n    {out_dir / c['file']}"
        print(line)


def run_sources(args: argparse.Namespace) -> int:
    opts = options_from_args(args)
    sources = expand_sources(args.sources)
    failed = []
    for n, source in enumerate(sources, 1):
        if len(sources) > 1:
            log(f"=== video {n} of {len(sources)}: {source}")
        try:
            work, clips = process(source, opts, new_clips=args.new_clips)
        except ClipperError as e:
            if len(sources) == 1:
                raise
            log(f"error: {e}")
            failed.append(source)
            continue
        if clips:
            if clips[0].get("file"):
                log(f"done: {len(clips)} clip(s) in {work / 'output'}")
            _summary(work, clips)
    if failed:
        log(f"{len(failed)} of {len(sources)} videos failed: " + ", ".join(failed))
        return 1
    return 0


def show_trending(argv: list[str]) -> int:
    """`clipper trending`: the creators' latest videos ranked by how much they're worth clipping."""
    from .trending import CLIPPING_LABELS, creators_path, load_creators, trending

    p = argparse.ArgumentParser(prog="clipper trending")
    p.add_argument("--work-dir", default="work", help="where results are kept (default ./work)")
    p.add_argument("--refresh", action="store_true", help="look the creators up again instead of using the last hour's result")
    p.add_argument("--all", action="store_true", help="also show creators not marked as allowing clipping")
    args = p.parse_args(argv)
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = trending(root, refresh=args.refresh)
    videos = [v for v in data["videos"] if args.all or v["clipping"] in ("allowed", "program")]
    for e in data["errors"]:
        log(f"{e['creator']}: {e['error']}")
    if not videos:
        print(f"Nothing to show. Add creators in the web interface (Creators) or in {creators_path(root)}"
              + ("" if args.all else ", or use --all to include creators not marked as allowing clipping") + ".")
        return 0
    for v in videos:
        print(f"[{v['score']:3d}] {v['verdict']:<8} {v['creator']} ({CLIPPING_LABELS[v['clipping']]})\n"
              f"      {v['title']}\n      {' · '.join(v['reasons'])}\n      {v['url']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv[:1] == ["doctor"]:
            from .doctor import doctor
            return doctor(argv[1:])
        if argv[:1] == ["ui"]:
            from .ui.server import serve
            return serve(argv[1:])
        if argv[:1] == ["trending"]:
            return show_trending(argv[1:])
        if argv[:1] == ["app"]:
            from .app import make_app
            return make_app(argv[1:])
        return run_sources(parse_args(argv))
    except ClipperError as e:
        print(f"clipper: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nclipper: interrupted; finished steps are saved and will be reused", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
