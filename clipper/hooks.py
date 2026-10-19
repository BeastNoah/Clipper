"""AI hook: ask Claude for other ways to open a clip, as titles shown at its start.

The suggestions are kept in work/<video>/edits/<clip id>.hooks.json, next to your own changes to the
clip, so the editor can show them again without asking twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from .llm import Claude, InvalidReply, extract_json
from .util import log, read_json, write_json

SYSTEM_PROMPT = """\
You are a short-form video editor who writes the on-screen title that opens a clip on TikTok, \
Reels and Shorts: the hook that makes a stranger stop scrolling in the first second. You write in \
the language the clip is spoken in, without clickbait clichés, emoji or exclamation marks. \
You answer with JSON only."""


class Hook(BaseModel):
    title: str = Field(min_length=1, max_length=60)
    why: str = Field(default="", max_length=200)


class HookReply(BaseModel):
    hooks: list[Hook] = Field(min_length=1, max_length=8)


def parse_hooks(reply: str) -> list[dict]:
    try:
        parsed = HookReply.model_validate(extract_json(reply))
    except ValidationError as e:
        raise InvalidReply(str(e)) from e
    seen, out = set(), []
    for h in parsed.hooks:
        title = " ".join(h.title.split())
        if title.lower() in seen:
            continue
        seen.add(title.lower())
        out.append({"title": title, "why": " ".join(h.why.split())})
    return out


def _prompt(video_title: str, clip: dict) -> str:
    return f"""\
Video title: {video_title}

This clip opens with the title "{clip['title']}". Here is what is said in it:

<clip_text>
{clip.get('text', '')}
</clip_text>

Suggest 5 different titles to open this clip with instead. Each one at most 50 characters, in the \
language of the clip, and true to what is actually said (no promises the clip doesn't keep). Mix the \
angles: a question, a bold claim, the surprising detail, the number or name that matters, the \
emotion. For each, say in a few words why it would stop a viewer.

Reply with only a JSON object in exactly this format:
{{"hooks": [{{"title": "...", "why": "..."}}]}}"""


def hooks_path(work: Path, clip: dict) -> Path:
    return work / "edits" / f"{clip['id']}.hooks.json"


def load_hooks(work: Path, clip: dict) -> list[dict] | None:
    path = hooks_path(work, clip)
    if not path.exists():
        return None
    try:
        return read_json(path)["hooks"]
    except (KeyError, ValueError, TypeError):
        return None


def suggest_hooks(work: Path, meta: dict, clip: dict, claude_factory: Callable[[], Claude]) -> list[dict]:
    log(f"hooks #{clip['rank']}: asking Claude for other ways to open the clip")
    hooks = claude_factory().ask(SYSTEM_PROMPT, _prompt(meta.get("title", ""), clip), parse_hooks, what="hooks")
    write_json(hooks_path(work, clip), {"for": clip["title"], "hooks": hooks})
    return hooks
