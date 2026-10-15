"""The six steps wired together, shared by the command line and the web UI."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
from pathlib import Path
from typing import Callable

from .audio import Loudness
from .captions import PRESETS, _phrases, build_srt, keyword_positions, make_captions, make_style, map_words
from .edit import MODES as JUMP_CUT_MODES
from .edit import EditMap, plan_cuts
from .formats import get_format, parse_formats, parse_time
from .ingest import DOWNLOAD_MODES, clip_video, ingest
from .llm import Claude
from .posts import format_post, write_posts
from .reframe import LAYOUTS, reframe
from .render import ENCODERS, LOGO_POSITIONS, Look, check_ffmpeg_has_libass, render
from .select import clips_from_review, select_clips
from .transcribe import clip_words, transcribe
from .translate import spread, translate_clip
from .util import ClipperError, check_cancel, log, read_json, report, require_tool, slugify, write_json

STEPS = ["ingest", "transcribe", "select", "reframe", "captions", "render"]


@dataclass
class Options:
    num_clips: int = 3
    min_len: float = 20
    max_len: float = 60
    prompt: str | None = None           # what Claude should look for, e.g. "funny moments"
    range_start: float | str | None = None  # only use this part of the video (seconds or "12:30")
    range_end: float | str | None = None
    work_dir: str = "work"
    download: str = "sections"          # links: sections (audio first, video only for the clips) | full
    model: str = "opus"                 # Claude model that picks the clips
    whisper_model: str = "small"
    whisper_backend: str = "auto"       # auto | faster-whisper | mlx
    language: str | None = None
    caption_model: str = "turbo"        # re-transcribes the chosen clips; "none" to skip
    caption_language: str | None = None  # translate the captions, e.g. "en" or "English"
    caption_style: str = "bold"         # bold | clean | boxed
    font: str | None = None
    caption_size: int | None = None
    highlight_color: str | None = None
    uppercase: bool = False
    caption_position: str = "lower"     # lower | middle
    title_overlay: bool = True
    formats: list[str] = field(default_factory=lambda: ["9:16"])  # 9:16, 4:5, 1:1, 16:9
    zoom: bool = True                   # punch in on the hook and the keywords
    progress_bar: bool = False
    logo: str | None = None             # image file, ideally a PNG with transparency
    logo_position: str = "top-right"
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
            ("download", self.download, DOWNLOAD_MODES), ("logo position", self.logo_position, LOGO_POSITIONS),
            ("whisper backend", self.whisper_backend, ("auto", "faster-whisper", "mlx")),
            ("force", self.force, (None, *STEPS)), ("stop after", self.stop_after, (None, *STEPS)),
        ]:
            if value not in allowed:
                problems.append(f"{name} must be one of {', '.join(str(a) for a in allowed if a)}")
        try:
            self.formats = parse_formats(self.formats)
            self.range_start, self.range_end = parse_time(self.range_start), parse_time(self.range_end)
            if self.range_start is not None and self.range_end is not None and self.range_end <= self.range_start:
                problems.append("the end of the time range must come after its start")
            self.style  # noqa: B018 - builds the caption style, which checks font size and colour
        except ValueError as e:
            problems.append(str(e))
        for name in ("prompt", "caption_language", "logo", "language"):
            value = getattr(self, name)
            setattr(self, name, value.strip() if isinstance(value, str) and value.strip() else None)
        if self.logo and not Path(self.logo).expanduser().is_file():
            problems.append(f"no logo file at {self.logo}")
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


def zoom_windows(words: list[dict], keywords: list[str], duration: float) -> list[tuple[float, float]]:
    """When to punch in: on the opening line (the hook) and on each of Claude's keywords."""
    windows = [(0.0, min(1.4, duration))]
    for i in sorted(keyword_positions(words, keywords)):
        windows.append((max(0.0, words[i]["s"] - 0.05), min(duration, words[i]["e"] + 1.0)))
    merged: list[tuple[float, float]] = []
    for a, b in sorted(windows):
        if merged and a - merged[-1][1] < 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


# ---------------------------------------------------------------------------------------------
# Caption corrections from the web UI

def _edits_path(work: Path, clip: dict) -> Path:
    return clip_dir_for(work, clip) / "words_edited.json"


def edited_words(work: Path, clip: dict) -> list[dict] | None:
    path = _edits_path(work, clip)
    if path.exists():
        data = read_json(path)
        if data.get("span") == [clip["start"], clip["end"]]:
            return data["words"]
    return None


def caption_lines(work: Path, clip: dict) -> list[dict]:
    """The clip's words as editable sentences (source times)."""
    words = edited_words(work, clip)
    if words is None:
        path = clip_dir_for(work, clip) / "words.json"
        if path.exists():
            words = read_json(path)["result"]
        else:
            transcript = read_json(work / "transcript.json")["result"]
            words = [w for w in transcript["words"] if clip["start"] <= (w["s"] + w["e"]) / 2 < clip["end"]]
    words = [{**w, "w": w["w"].strip()} for w in words if w["w"].strip()]
    return [{"start": p[0]["s"], "end": p[-1]["e"], "text": " ".join(w["w"] for w in p)} for p in _phrases(words)]


def save_caption_lines(work: Path, clip: dict, lines: list[dict]) -> None:
    """Store corrected sentences; each one's time is spread over its (possibly changed) words."""
    words = []
    for line in lines:
        words += spread(float(line["start"]), float(line["end"]), str(line["text"]))
    if not words:
        raise ClipperError("The captions can't be empty.")
    write_json(_edits_path(work, clip), {"span": [clip["start"], clip["end"]], "words": words})


# ---------------------------------------------------------------------------------------------

def produce(work: Path, meta: dict, transcript: dict, clips: list[dict], opts: Options) -> list[dict]:
    """Steps 4-6 (reframe, captions, render) for clips that have already been chosen."""
    do_render = opts.stop_after not in ("reframe", "captions")
    if do_render:
        check_ffmpeg_has_libass()
    style = opts.style
    formats = [get_format(n) for n in opts.formats]
    fps = float(Fraction(meta["fps"]))
    loud = Loudness.from_work(work) if opts.jump_cuts != "off" else None
    claude = claude_factory(opts, work)
    out_dir = work / "output"
    out_dir.mkdir(exist_ok=True)
    if do_render:  # the list goes first, so an interrupted render never points at deleted files
        for old in out_dir.iterdir():
            if old.suffix in (".mp4", ".jpg", ".txt", ".srt") or old.name == "clips.json":
                old.unlink()

    for clip in clips:
        check_cancel()
        report("step", f"clip {clip['rank']} of {len(clips)}")
        cdir = clip_dir_for(work, clip)
        cdir.mkdir(parents=True, exist_ok=True)
        words = clip_words(work, transcript, clip, cdir, opts.caption_model, opts.whisper_backend,
                           force=opts.forced("captions"))
        fixed = edited_words(work, clip)
        if fixed is not None:
            log(f"captions #{clip['rank']}: using your corrections")
            words = fixed
        src = clip_video(work, meta, clip, cdir)
        edit = EditMap(plan_cuts(words, clip["start"], clip["end"], fps, opts.jump_cuts,
                                 loud.is_quiet if loud else None))
        spoken = map_words(words, clip["start"], edit)
        cap_words, cap_title, keywords = spoken, clip["title"], clip.get("keywords")
        if opts.caption_language:
            cap_words, cap_title = translate_clip(cdir, spoken, clip["title"], opts.caption_language, claude)
            keywords = None  # the keywords were picked in the spoken language
        look = Look(zoom=zoom_windows(spoken, clip.get("keywords") or [], edit.duration) if opts.zoom else [],
                    logo=Path(opts.logo).expanduser() if opts.logo else None, logo_position=opts.logo_position,
                    progress=style.highlight if opts.progress_bar else None)
        clip["final_duration"] = round(edit.duration, 2)
        base = f"{clip['rank']:02d}_{slugify(clip['title'])}"
        outputs = {}
        for n, fmt in enumerate(formats):
            fdir = cdir / fmt.tag
            fdir.mkdir(exist_ok=True)
            plan = reframe(work, meta, clip, fdir, words, opts.layout, force=opts.forced("reframe"), src=src, fmt=fmt)
            ass = make_captions(cap_words, edit.duration, fdir, style, fmt,
                                title=cap_title if opts.title_overlay else None, cover_title=cap_title,
                                keywords=keywords)
            if not do_render:
                continue
            mp4 = render(work, meta, clip, plan, edit, ass, fdir, fmt=fmt, look=look, encoder=opts.encoder,
                         crf=opts.crf, preset=opts.preset, force=opts.forced("render"), src=src)
            name = base if n == 0 else f"{base}_{fmt.tag}"
            _publish(mp4, out_dir / f"{name}.mp4")
            _publish(fdir / "cover.jpg", out_dir / f"{name}.jpg")
            outputs[fmt.name] = {"file": f"{name}.mp4", "cover": f"{name}.jpg"}
        if do_render:
            clip["file"], clip["cover"] = outputs[formats[0].name]["file"], outputs[formats[0].name]["cover"]
            clip["formats"] = outputs
            (out_dir / f"{base}.srt").write_text(build_srt(spoken), encoding="utf-8")
            clip["srt"] = [f"{base}.srt"]
            if opts.caption_language:
                tag = slugify(opts.caption_language, 12)
                (out_dir / f"{base}.{tag}.srt").write_text(build_srt(cap_words), encoding="utf-8")
                clip["srt"].append(f"{base}.{tag}.srt")

    if do_render and opts.posts:
        try:
            posts = write_posts(work, meta, clips, claude, force=opts.forced("render"))
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
    work, meta = ingest(source, work_root, force=opts.forced("ingest"), download_mode=opts.download)
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
                         force=opts.forced("select"), focus=opts.prompt,
                         time_range=(opts.range_start, opts.range_end))
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


def rerender(work: Path, opts: Options) -> list[dict]:
    """Render the current clips again (after caption corrections, or with other options)."""
    meta = read_json(work / "meta.json")
    transcript = read_json(work / "transcript.json")["result"]
    path = work / "output" / "clips.json"
    if not path.exists():
        raise ClipperError("There are no rendered clips to update.")
    keep = ("rank", "id", "title", "hook", "score", "scores", "start", "end", "duration", "si", "ei",
            "text", "keywords")
    clips = [{k: c[k] for k in keep if k in c} for c in read_json(path)]
    return produce(work, meta, transcript, clips, opts)
