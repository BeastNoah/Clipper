"""Translated captions: Claude translates a clip sentence by sentence, and each sentence's time is
spread over the translated words, so the word-by-word highlight still roughly follows the speech."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from .captions import _WIDE, _phrases, text_width
from .llm import Claude, InvalidReply, extract_json
from .util import fingerprint, log, read_json, write_json

SYSTEM_PROMPT = """\
You translate the captions of short social media videos. You keep the tone and energy of spoken \
language, keep names and terms that shouldn't be translated, and keep each line short. You answer \
with JSON only."""


class Translation(BaseModel):
    title: str
    lines: list[str]


def parse_translation(reply: str, n: int) -> dict:
    try:
        t = Translation.model_validate(extract_json(reply))
    except ValidationError as e:
        raise InvalidReply(str(e)) from e
    if len(t.lines) != n:
        raise InvalidReply(f"expected exactly {n} lines in the same order, got {len(t.lines)}")
    if any(not line.strip() for line in t.lines):
        raise InvalidReply("every line needs a translation; none may be empty")
    return {"title": t.title.strip(), "lines": [line.strip() for line in t.lines]}


def _prompt(language: str, title: str, lines: list[str]) -> str:
    numbered = "\n".join(f"{n}. {line}" for n, line in enumerate(lines, 1))
    return f"""\
Translate the title and the caption lines of this short video into {language}.

Title: {title}

<lines>
{numbered}
</lines>

Translate each numbered line on its own, in the same order, without the numbers. Reply with only a \
JSON object in exactly this format, with {len(lines)} lines:
{{"title": "...", "lines": ["...", "..."]}}"""


CHUNK_WIDTH = 8  # a caption "word" in Chinese or Japanese: about four characters


def _chunks(token: str) -> list[str]:
    """Short pieces of a run of text without spaces (Chinese, Japanese); other tokens stay whole.
    Latin words and numbers inside stay whole, and punctuation stays with the text before it."""
    if not _WIDE.search(token) or text_width(token) <= CHUNK_WIDTH:
        return [token]
    out: list[str] = []
    for atom in re.findall(rf"(?:(?!{_WIDE.pattern})[^\W_])+|.", token, re.S):
        if out and (not re.match(r"\w", atom) or text_width(out[-1]) + text_width(atom) <= CHUNK_WIDTH
                    and not re.search(r"\W$", out[-1])):
            out[-1] += atom
        else:
            out.append(atom)
    return out


def spread(start: float, end: float, text: str) -> list[dict]:
    """Words of `text` with times spread over [start, end] in proportion to their length."""
    tokens = [c for tok in text.split() for c in _chunks(tok)]
    if not tokens:
        return []
    weights = [text_width(t) + 1 for t in tokens]
    total, t, out = sum(weights), start, []
    for tok, w in zip(tokens, weights):
        d = (end - start) * w / total
        out.append({"w": tok, "s": round(t, 3), "e": round(t + d, 3)})
        t += d
    return out


def translate_clip(cdir: Path, words: list[dict], title: str, language: str,
                   claude_factory: Callable[[], Claude]) -> tuple[list[dict], str]:
    """(translated words with times in the finished clip, translated title)."""
    phrases = _phrases(words)
    if not phrases:  # nothing is said in the clip: no captions to translate
        return [], title
    texts = [" ".join(w["w"] for w in p) for p in phrases]
    key = fingerprint([language, title, texts])
    path = cdir / f"translation-{fingerprint(language)[:6]}.json"
    stored = read_json(path) if path.exists() else None
    if not stored or stored.get("key") != key:
        log(f"captions: translating into {language}")
        result = claude_factory().ask(SYSTEM_PROMPT, _prompt(language, title, texts),
                                      lambda r: parse_translation(r, len(texts)), what="translation")
        stored = {"key": key, "language": language, **result}
        write_json(path, stored)
    out = []
    for p, line in zip(phrases, stored["lines"]):
        out += spread(p[0]["s"], p[-1]["e"], line)
    return out, stored["title"]
