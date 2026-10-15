"""Output formats (aspect ratios) and small helpers for parsing them and times."""

from __future__ import annotations

import re
from dataclasses import dataclass

SIZES = {"9:16": (1080, 1920), "4:5": (1080, 1350), "1:1": (1080, 1080), "16:9": (1920, 1080)}


@dataclass(frozen=True)
class Format:
    name: str
    width: int
    height: int

    @property
    def tag(self) -> str:
        return self.name.replace(":", "x")

    @property
    def ratio(self) -> float:
        return self.width / self.height

    def v(self, px_at_1920: float) -> int:
        """A vertical distance designed for a 1920px-tall frame, scaled to this format."""
        return int(round(px_at_1920 * self.height / 1920))


VERTICAL = Format("9:16", *SIZES["9:16"])


def get_format(name: str) -> Format:
    if name not in SIZES:
        raise ValueError(f"unknown format '{name}' (choose from {', '.join(SIZES)})")
    return Format(name, *SIZES[name])


def parse_formats(text: str | list[str]) -> list[str]:
    names = text if isinstance(text, list) else [t.strip() for t in str(text).split(",")]
    out = []
    for n in names:
        n = n.strip().replace("x", ":")
        get_format(n)
        if n not in out:
            out.append(n)
    if not out:
        raise ValueError("give at least one format, e.g. 9:16")
    return out


def parse_time(text: str | float | int | None) -> float | None:
    """'75', '1:15', '1:02:03' or a number -> seconds."""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        return float(text)
    if not re.fullmatch(r"\d+(\.\d+)?|(\d+:)?\d{1,2}:\d{1,2}(\.\d+)?", text.strip()):
        raise ValueError(f"times look like 75, 1:15 or 1:02:03, not {text!r}")
    total = 0.0
    for part in text.strip().split(":"):
        total = total * 60 + float(part)
    return total
