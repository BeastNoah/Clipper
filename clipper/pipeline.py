"""The six steps wired together, shared by the command line and the web UI."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Callable

from .audio import Loudness
from .captions import PRESETS, make_captions, make_style
from .edit import MODES as JUMP_CUT_MODES
from .edit import EditMap, plan_cuts
from .ingest import ingest
from .llm import Claude
from .posts import format_post, write_posts
from .reframe import LAYOUTS, reframe
from .render import ENCODERS, check_ffmpeg_has_libass, render
from .select import clips_from_review, select_clips
from .transcribe import clip_words, transcribe
from .util import ClipperError, check_cancel, log, read_json, report, require_tool, slugify, write_json

STEPS = ["ingest", "transcribe", "select", "reframe", "captions", "render"]


@dataclass
class Options:
    num_clips: int = 3
    min_len: float = 20
    max_len: float = 60
    work_dir: str = "work"
    model: str = "opus"                 # Claude model that picks the clips
    whisper_model: str = "small"
    whisper_backend: str = "auto"       # auto | faster-whisper | mlx
    language: str | None = None
    caption_model: str = "turbo"        # re-transcribes the chosen clips; "none" to skip
    caption_style: str = "bold"         # bold | clean | boxed
    font: str | None = None
    caption_size: int | None = None
    highlight_color: str | None = None
    uppercase: bool = False
    caption_position: str = "lower"     # lower | middle
    title_overlay: bool = True
    jump_cuts: str = "on"               # on | pauses | off
    layout: str = "auto"                # auto | follow | split | main
    encoder: str = "x264"               # x264 | videotoolbox | auto
    crf: int = 20
    preset: str = "medium"
    posts: bool = True
    force: str | None = None
    stop_after: str | None = None

    def validate(self) -> "Options":
        problems = []
        if self.num_clips < 1:
            problems.append("number of clips must be at least 1")
        if self.min_len <= 0 or self.max_len < self.min_len:
            problems.append("minimum length must be positive and no larger than the maximum")
        for name, value, allowed in [
            ("caption style", self.caption_style, tuple(PRESETS)), ("jump cuts", self.jump_cuts, JUMP_CUT_MODES),
            ("layout", self.layout, LAYOUTS), ("encoder", self.encoder, ENCODERS),
            ("caption position", self.caption_position, ("lower", "middle")),
            ("whisper backend", self.whisper_backend, ("auto", "faster-whisper", "mlx")),
            ("force", self.force, (None, *STEPS)), ("stop after", self.stop_after, (None, *STEPS)),
        ]:
            if value not in allowed:
                problems.append(f"{name} must be one of {', '.join(str(a) for a in allowed if a)}")
        if not problems:
            try:
                self.style  # noqa: B018 - builds the caption style, which checks font size and colour
            except ValueError as e:
                problems.append(str(e))
        if problems:
            raise ClipperError("; ".join(problems))
        return self

    @classmethod
    def from_dict(cls, data: dict) -> "Options":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names}).validate()

    def to_dict(self) -> dict:
        return asdict(self)

    def forced(self, step: str) -> bool:
        return self.force is not None and STEPS.index(step) >= STEPS.index(self.force)

    @property
    def style(self):
        return make_style(self.caption_style, font=self.font, size=self.caption_size,
                          highlight=self.highlight_color, uppercase=self.uppercase,
                          position=self.caption_position)


def claude_factory(opts: Options, work: Path) -> Callable[[], Claude]:
    return lambda: Claude(opts.model, cwd=work)


def _publish(src: Path, dest: Path) -> None:
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def clip_dir_for(work: Path, clip: dict) -> Path:
    return work / "clips" / f"{round(clip['start'] * 1000):08d}-{round(clip['end'] * 1000):08d}"


def produce(work: Path, meta: dict, transcript: dict, clips: list[dict], opts: Options) -> list[dict]:
    """Steps 4-6 (reframe, captions, render) for clips that have already been chosen."""
    do_render = opts.stop_after not in ("reframe", "captions")
    if do_render:
        check_ffmpeg_has_libass()
    style = opts.style
    fps = float(Fraction(meta["fps"]))
    loud = Loudness.from_work(work) if opts.jump_cuts != "off" else None
    out_dir = work / "output"
    out_dir.mkdir(exist_ok=True)
    if do_render:  # the list goes first, so an interrupted render never points at deleted files
        for old in out_dir.iterdir():
            if old.suffix in (".mp4", ".jpg", ".txt") or old.name == "clips.json":
                old.unlink()

    for clip in clips:
        check_cancel()
        report("step", f"clip {clip['rank']} of {len(clips)}")
        cdir = clip_dir_for(work, clip)
        cdir.mkdir(parents=True, exist_ok=True)
        words = clip_words(work, transcript, clip, cdir, opts.caption_model, opts.whisper_backend,
                           force=opts.forced("captions"))
        plan = reframe(work, meta, clip, cdir, words, opts.layout, force=opts.forced("reframe"))
        edit = EditMap(plan_cuts(words, clip["start"], clip["end"], fps, opts.jump_cuts,
                                 loud.is_quiet if loud else None))
        ass = make_captions(clip, words, edit, cdir, style, opts.title_overlay)
        clip["final_duration"] = round(edit.duration, 2)
        if not do_render:
            continue
        mp4 = render(work, meta, clip, plan, edit, ass, cdir, encoder=opts.encoder, crf=opts.crf,
                     preset=opts.preset, force=opts.forced("render"))
        name = f"{clip['rank']:02d}_{slugify(clip['title'])}"
        _publish(mp4, out_dir / f"{name}.mp4")
        _publish(cdir / "cover.jpg", out_dir / f"{name}.jpg")
        clip["file"], clip["cover"] = f"{name}.mp4", f"{name}.jpg"

    if do_render and opts.posts:
        try:
            posts = write_posts(work, meta, clips, claude_factory(opts, work), force=opts.forced("render"))
        except ClipperError as e:
            log(f"posts: skipped ({e})")
            posts = {}
        for clip in clips:
            if clip["id"] in posts:
                clip["post"] = posts[clip["id"]]
                text_file = out_dir / f"{clip['rank']:02d}_{slugify(clip['title'])}.txt"
                text_file.write_text(format_post(clip, posts[clip["id"]]), encoding="utf-8")
    write_json(out_dir / "clips.json", clips)
    return clips


def process(source: str, opts: Options) -> tuple[Path, list[dict]]:
    """Run one video through the pipeline (or up to opts.stop_after)."""
    for tool in ("ffmpeg", "ffprobe"):
        require_tool(tool)
    work_root = Path(opts.work_dir)
    report("step", "ingest")
    work, meta = ingest(source, work_root, force=opts.forced("ingest"))
    report("video", work.name)
    if opts.stop_after == "ingest":
        return work, []
    check_cancel()
    report("step", "transcribe")
    transcript = transcribe(work, meta, opts.whisper_model, opts.language, opts.whisper_backend,
                            force=opts.forced("transcribe"))
    if opts.stop_after == "transcribe":
        return work, []
    check_cancel()
    report("step", "select")
    clips = select_clips(work, meta, transcript, num_clips=opts.num_clips, min_len=opts.min_len,
                         max_len=opts.max_len, model=opts.model, claude_factory=claude_factory(opts, work),
                         force=opts.forced("select"))
    if not clips:
        raise ClipperError("Claude didn't find any usable clips in this video.")
    check_cancel()
    if opts.stop_after == "select":
        return work, clips
    return work, produce(work, meta, transcript, clips, opts)


def load_video(work: Path) -> dict:
    """Everything the UI shows about one processed video."""
    out = {"id": work.name, "meta": read_json(work / "meta.json") if (work / "meta.json").exists() else None}
    for name in ("clips", "review"):
        path = work / f"{name}.json"
        out[name] = read_json(path) if path.exists() else None
    out["transcribed"] = (work / "transcript.json").exists()
    sel = work / "select.json"
    out["candidates"] = read_json(sel)["candidates"] if sel.exists() else []
    rendered = work / "output" / "clips.json"
    out["rendered"] = read_json(rendered) if rendered.exists() else []
    return out


def produce_reviewed(work: Path, items: list[dict], opts: Options) -> list[dict]:
    """Render clips chosen (and possibly adjusted) by hand in the review screen."""
    meta = read_json(work / "meta.json")
    transcript = read_json(work / "transcript.json")["result"]
    candidates = read_json(work / "select.json")["candidates"] if (work / "select.json").exists() else []
    clips = clips_from_review(transcript, meta, candidates, items)
    if not clips:
        raise ClipperError("No clips were selected.")
    sig = ",".join(f"{c['id']}:{c['si']}-{c['ei']}" for c in candidates)  # same format as the UI's
    write_json(work / "review.json", {"items": items, "candidates": sig, "options": opts.to_dict()})
    return produce(work, meta, transcript, clips, opts)
