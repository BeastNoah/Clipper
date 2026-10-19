"""Post-ready text for each clip: captions and hashtags for TikTok, Instagram and YouTube Shorts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from .campaign import caption_guide, enforce_post
from .llm import Claude, InvalidReply, extract_json
from .util import fingerprint, log, read_json, write_json

POSTS_VERSION = 1

SYSTEM_PROMPT = """\
You are a social media manager who writes the text that goes with short vertical videos on \
TikTok, Instagram Reels and YouTube Shorts. You write naturally, without clickbait clichés or \
emoji spam, in the language the video is spoken in. You answer with JSON only."""


class Post(BaseModel):
    id: str
    tiktok: str = Field(min_length=1, max_length=2200)
    instagram: str = Field(min_length=1, max_length=2200)
    youtube_title: str = Field(min_length=1, max_length=100)
    youtube_description: str = Field(min_length=1, max_length=5000)
    hashtags: list[str] = Field(max_length=12)


class PostReply(BaseModel):
    posts: list[Post]


def parse_posts(reply: str, ids: list[str]) -> dict[str, dict]:
    try:
        parsed = PostReply.model_validate(extract_json(reply))
    except ValidationError as e:
        raise InvalidReply(str(e)) from e
    got = {p.id: p for p in parsed.posts}
    missing = [i for i in ids if i not in got]
    if missing:
        raise InvalidReply("missing posts for: " + ", ".join(missing))
    return {i: {**got[i].model_dump(exclude={"id"}),
                "hashtags": [h.lstrip("#").strip() for h in got[i].hashtags if h.strip("# ")]}
            for i in ids}


def _prompt(title: str, clips: list[dict], campaign: dict | None = None) -> str:
    blocks = "\n\n".join(f"[{c['id']}] \"{c['title']}\"\n{c['text']}" for c in clips)
    return f"""\
Video title: {title}

Write the text to post with each of these short clips, in the language the clip is spoken in.

<clips>
{blocks}
</clips>

For each clip:
- tiktok: one or two punchy sentences that make people want to watch, then 3-5 hashtags
- instagram: two or three sentences, then 5-8 hashtags
- youtube_title: at most 70 characters, no hashtags
- youtube_description: one or two sentences, then 3 hashtags
- hashtags: the 5 most relevant hashtags, without the # sign{caption_guide(campaign)}

Reply with only a JSON object in exactly this format, one entry per clip id:
{{"posts": [{{"id": "{clips[0]['id']}", "tiktok": "...", "instagram": "...", "youtube_title": "...", \
"youtube_description": "...", "hashtags": ["..."]}}]}}"""


def write_posts(work: Path, meta: dict, clips: list[dict], claude_factory: Callable[[], Claude],
                force: bool = False, campaign: dict | None = None) -> dict[str, dict]:
    """Post text per clip id. Stored per clip, so only clips without text yet go to Claude. With a
    campaign, Claude follows its caption rules and its required hashtags and mentions are always added."""
    path = work / "posts.json"
    store = read_json(path) if path.exists() and not force else {}
    if store.get("version") != POSTS_VERSION:
        store = {"version": POSTS_VERSION, "posts": {}}
    key = lambda c: fingerprint([c["title"], c["text"]] + ([campaign] if campaign else []))  # noqa: E731
    todo = [c for c in clips if key(c) not in store["posts"]]
    if todo:
        log(f"posts: asking Claude for post text for {len(todo)} clip(s)")
        ids = [c["id"] for c in todo]
        got = claude_factory().ask(SYSTEM_PROMPT, _prompt(meta.get("title", ""), todo, campaign),
                                   lambda r: parse_posts(r, ids), what="post text")
        for c in todo:
            store["posts"][key(c)] = got[c["id"]]
        write_json(path, store)
    return {c["id"]: enforce_post(store["posts"][key(c)], campaign) for c in clips}


def format_post(clip: dict, post: dict) -> str:
    return (f"{clip['title']}\n\n"
            f"TikTok\n{post['tiktok']}\n\n"
            f"Instagram\n{post['instagram']}\n\n"
            f"YouTube Shorts\n{post['youtube_title']}\n{post['youtube_description']}\n\n"
            f"Hashtags\n{' '.join('#' + h for h in post['hashtags'])}\n")
