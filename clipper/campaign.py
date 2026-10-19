"""Clipping campaigns (Content Rewards, Whop, Vyro and the like): a brand's brief, turned into rules.

You paste the campaign's instructions as they are. Claude reads them once and says what they mean for
clipper: what to look for, how long clips may be, what every post must say (hashtags, @mentions, a
call to action), whether the post must be marked as an ad, which platforms count, and what's left for
you to do by hand. Those rules then steer the clip picking, the look, the post text and the publish
form. The hashtags, mentions and phrases are added to the post text by clipper itself, not only asked
for, so no post goes out without them.

Saved campaigns live in <work>/campaigns.json; each video remembers its campaign in campaign.json.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError, field_validator

from .llm import Claude, InvalidReply, extract_json
from .util import ClipperError, fingerprint, log, read_json, write_json

CAMPAIGN_VERSION = 1
PLATFORM_NAMES = ("tiktok", "instagram", "youtube")
MAX_BRIEF = 20000

SYSTEM_PROMPT = """\
You help a clipper (someone who cuts short vertical clips out of long videos and posts them for a \
brand's paid clipping campaign) follow the campaign's rules exactly. You read campaign briefs \
carefully and literally, never invent requirements that aren't there, and answer with JSON only."""


class Rules(BaseModel):
    name: str = Field(default="", max_length=80)
    focus: str | None = Field(default=None, max_length=600)
    avoid: list[str] = Field(default_factory=list, max_length=15)
    min_len: float | None = Field(default=None, ge=3, le=600)
    max_len: float | None = Field(default=None, ge=3, le=600)
    hashtags: list[str] = Field(default_factory=list, max_length=15)
    mentions: list[str] = Field(default_factory=list, max_length=10)
    must_include: list[str] = Field(default_factory=list, max_length=8)
    caption_notes: str | None = Field(default=None, max_length=600)
    disclosure: bool = False
    platforms: list[str] = Field(default_factory=list)
    on_screen_text: str | None = Field(default=None, max_length=80)
    no_watermark: bool = False
    manual: list[str] = Field(default_factory=list, max_length=15)

    @field_validator("hashtags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        return _unique(re.sub(r"\s+", "", t).lstrip("#") for t in v)

    @field_validator("mentions")
    @classmethod
    def _mentions(cls, v: list[str]) -> list[str]:
        return _unique("@" + re.sub(r"\s+", "", m).lstrip("@") for m in v)

    @field_validator("platforms")
    @classmethod
    def _platforms(cls, v: list[str]) -> list[str]:
        names = {"tiktok": "tiktok", "instagram": "instagram", "reels": "instagram", "youtube": "youtube",
                 "shorts": "youtube", "youtube shorts": "youtube", "instagram reels": "instagram"}
        return _unique(names[p.strip().lower()] for p in v if p.strip().lower() in names)

    @field_validator("avoid", "must_include", "manual")
    @classmethod
    def _strip(cls, v: list[str]) -> list[str]:
        return _unique(s.strip() for s in v)


def _unique(items) -> list[str]:
    out = []
    for s in items:
        if s and s.strip("#@") and s.lower() not in (o.lower() for o in out):
            out.append(s)
    return out


def _prompt(brief: str) -> str:
    return f"""\
Here is a clipping campaign's brief, exactly as the brand wrote it:

<brief>
{brief}
</brief>

Say what it requires, as JSON with these fields. Use null, false or [] for anything the brief \
doesn't ask for; don't add requirements of your own.

- name: a short name for the campaign (the brand or creator), at most 40 characters
- focus: what kind of moments to clip, in one or two sentences (topics, people, products), or null
- avoid: content the clips must not contain (e.g. "swearing", "competitor names")
- min_len, max_len: the allowed clip length in seconds, if the brief sets one
- hashtags: hashtags every post must have, without the # sign
- mentions: accounts every post must tag, e.g. "@brand"
- must_include: other exact text every caption must contain (a link, a promo code, "link in bio"), \
copied exactly from the brief
- caption_notes: how the caption should be written (tone, call to action, language), or null
- disclosure: true if posts must be marked as an ad, sponsored or a paid partnership (e.g. #ad, \
TikTok's branded content switch)
- platforms: where the clips may be posted, from "tiktok", "instagram", "youtube"; [] if any
- on_screen_text: text the brand wants shown on the video itself the whole time (e.g. their handle), \
or null
- no_watermark: true if the clips must not carry the clipper's own watermark or logo
- manual: requirements clipper can't take care of itself, one short sentence each, for the clipper's \
checklist (e.g. "Use a trending sound", "Submit the post link within 48 hours", "Account needs 1,000 \
followers", "Keep the post up for 30 days")

Reply with only the JSON object:
{{"name": "...", "focus": null, "avoid": [], "min_len": null, "max_len": null, "hashtags": [], \
"mentions": [], "must_include": [], "caption_notes": null, "disclosure": false, "platforms": [], \
"on_screen_text": null, "no_watermark": false, "manual": []}}"""


def parse_rules(reply: str) -> dict:
    try:
        rules = Rules.model_validate(extract_json(reply))
    except ValidationError as e:
        raise InvalidReply(str(e)) from e
    if rules.min_len and rules.max_len and rules.max_len < rules.min_len:
        raise InvalidReply("max_len is shorter than min_len")
    return rules.model_dump()


def clean_brief(brief) -> str:
    brief = str(brief or "").strip()
    if not brief:
        raise ClipperError("Paste the campaign's instructions first.")
    if len(brief) > MAX_BRIEF:
        raise ClipperError(f"The campaign brief is too long (at most {MAX_BRIEF} characters).")
    return brief


def interpret(brief: str, cache_dir: Path, claude_factory: Callable[[], Claude]) -> dict:
    """The rules in a brief. Each brief is read by Claude once; the answer is kept in cache_dir."""
    brief = clean_brief(brief)
    path = cache_dir / ".campaigns" / f"{fingerprint([CAMPAIGN_VERSION, brief])}.json"
    if path.exists():
        return read_json(path)["rules"]
    log("campaign: asking Claude what the campaign brief requires")
    rules = claude_factory().ask(SYSTEM_PROMPT, _prompt(brief), parse_rules, what="campaign rules")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"brief": brief, "rules": rules})
    return rules


# ---------------------------------------------------------------------------------------------
# What the rules change

def focus_with(focus: str | None, rules: dict | None) -> str | None:
    """Your own "look for" text plus what the campaign wants and rules out."""
    if not rules:
        return focus
    parts = [p for p in (focus, rules.get("focus")) if p]
    if rules.get("avoid"):
        parts.append("Never pick moments with: " + "; ".join(rules["avoid"]))
    return ". ".join(p.rstrip(". ") for p in parts) or None


def lengths_with(min_len: float, max_len: float, rules: dict | None) -> tuple[float, float]:
    """The campaign's clip lengths win over your own; a campaign that sets only one end keeps the other
    where it fits."""
    if not rules:
        return min_len, max_len
    lo, hi = rules.get("min_len") or min_len, rules.get("max_len") or max_len
    if hi < lo:
        lo, hi = (lo, lo + max(5.0, max_len - min_len)) if rules.get("min_len") else (max(3.0, hi / 2), hi)
    return float(lo), float(hi)


def caption_guide(rules: dict | None) -> str:
    """For Claude's post text prompt."""
    if not rules:
        return ""
    lines = []
    if rules.get("caption_notes"):
        lines.append(f"- how to write it: {rules['caption_notes']}")
    if rules.get("mentions"):
        lines.append(f"- tag {' '.join(rules['mentions'])}")
    if rules.get("hashtags"):
        lines.append(f"- use the hashtags {' '.join('#' + t for t in rules['hashtags'])}")
    if rules.get("must_include"):
        lines.append("- include, word for word: " + " | ".join(rules["must_include"]))
    if rules.get("disclosure"):
        lines.append("- it's a paid partnership: say so clearly (e.g. #ad)")
    if not lines:
        return ""
    return ("\n\nThese posts are for a paid clipping campaign" + (f" ({rules['name']})" if rules.get("name") else "")
            + ". Every caption must follow its rules:\n" + "\n".join(lines))


def _missing(text: str, needed: list[str]) -> list[str]:
    low = text.lower()
    return [n for n in needed if n.lower() not in low]


def enforce_post(post: dict, rules: dict | None) -> dict:
    """The post with every required hashtag, mention and phrase in it, whatever Claude wrote."""
    if not rules:
        return post
    post = dict(post)
    tags = [t for t in rules.get("hashtags", [])] + (["ad"] if rules.get("disclosure") else [])
    post["hashtags"] = _unique([*tags, *post.get("hashtags", [])])
    required = [*rules.get("mentions", []), *rules.get("must_include", [])]

    def fix(text: str, limit: int) -> str:
        extra = _missing(text, required)
        tag_text = " ".join(f"#{t}" for t in tags if not re.search(rf"#{re.escape(t)}\b", text, re.I))
        add = " ".join(p for p in (" ".join(extra), tag_text) if p)
        if not add:
            return text
        room = limit - len(add) - 2
        return f"{text[:max(0, room)].rstrip()}\n\n{add}".strip()

    for key, limit in (("tiktok", 2200), ("instagram", 2200), ("youtube_description", 5000)):
        if post.get(key) is not None:
            post[key] = fix(post[key], limit)
    return post


def checklist(rules: dict | None) -> list[str]:
    """What the clipper should check before submitting, in plain words."""
    if not rules:
        return []
    out = []
    if rules.get("platforms"):
        out.append("Post only to " + ", ".join(
            {"tiktok": "TikTok", "instagram": "Instagram", "youtube": "YouTube"}[p] for p in rules["platforms"]))
    if rules.get("disclosure"):
        out.append("Mark the posts as a paid partnership (clipper adds #ad and turns on TikTok's branded content switch)")
    return out + list(rules.get("manual", []))


# ---------------------------------------------------------------------------------------------
# Saved campaigns

def campaigns_path(root: Path) -> Path:
    return root / "campaigns.json"


def load_campaigns(root: Path) -> list[dict]:
    path = campaigns_path(root)
    data = read_json(path) if path.exists() else []
    return data if isinstance(data, list) else []


def save_campaign(root: Path, name: str, brief: str, rules: dict | None = None, cid: str | None = None) -> dict:
    brief = clean_brief(brief)
    items = load_campaigns(root)
    old = next((c for c in items if c.get("id") == cid), None) if cid else None
    item = {"id": cid if old else uuid.uuid4().hex[:10],
            "name": (str(name or "").strip() or (rules or {}).get("name") or "Campaign")[:80],
            "brief": brief, "rules": rules, "created": (old or {}).get("created") or time.time()}
    items = [item if c is old else c for c in items] if old else [item, *items]
    write_json(campaigns_path(root), items)
    return item


def delete_campaign(root: Path, cid: str) -> None:
    write_json(campaigns_path(root), [c for c in load_campaigns(root) if c.get("id") != cid])


def video_campaign(work: Path) -> dict | None:
    """{"brief", "rules"} of the campaign this video's clips were made for, if any."""
    path = work / "campaign.json"
    return read_json(path) if path.exists() else None
