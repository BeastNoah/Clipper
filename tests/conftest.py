import sys
from pathlib import Path

import pytest

FAKE_CLAUDE = Path(__file__).with_name("fake_claude.py")


def make_words(sentences, word_s=0.3, word_gap=0.1, sentence_gap=0.5, start=0.0):
    """Words with timings for a list of sentence strings."""
    words, t = [], start
    for sent in sentences:
        for tok in sent.split():
            words.append({"w": " " + tok, "s": round(t, 3), "e": round(t + word_s, 3), "p": 0.9})
            t += word_s + word_gap
        t += sentence_gap - word_gap
    return words


@pytest.fixture
def fake_claude(monkeypatch, tmp_path):
    log = tmp_path / "claude_calls.jsonl"
    wrapper = tmp_path / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE_CLAUDE} \"$@\"\n")
    wrapper.chmod(0o755)
    monkeypatch.setenv("CLIPPER_CLAUDE_BIN", str(wrapper))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    return log
