"""Talks to Claude through the `claude` command-line tool (so it runs on your Claude subscription).

Every answer is parsed and validated; if it doesn't match the expected format, the validation
error is sent back in the same conversation and Claude gets another try.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Callable, TypeVar

from .util import ClipperError, log

T = TypeVar("T")


class InvalidReply(ValueError):
    """Raised by a parse function when Claude's answer doesn't match the required format."""


def extract_json(text: str) -> object:
    """Pull the JSON object out of a reply, tolerating code fences or a stray sentence around it."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise InvalidReply("The reply did not contain a JSON object.")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise InvalidReply(f"The reply is not valid JSON: {e}") from e


class Claude:
    def __init__(self, model: str = "opus", cwd: Path | None = None, max_attempts: int = 3,
                 timeout: int = 900, binary: str | None = None):
        self.model = model
        self.cwd = cwd
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.binary = binary or os.environ.get("CLIPPER_CLAUDE_BIN") or shutil.which("claude")
        if not self.binary:
            raise ClipperError("The 'claude' command was not found. Install Claude Code and log in "
                               "(https://claude.com/claude-code), then run clipper again.")

    def _call(self, system: str, message: str, session_id: str, resume: bool) -> tuple[str, str | None]:
        cmd = [self.binary, "-p", "--output-format", "json", "--model", self.model,
               "--system-prompt", system, "--tools", "",
               "--resume" if resume else "--session-id", session_id]
        # When clipper itself runs inside Claude Code, don't let the child join the parent's session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_SESSION_ID"}
        try:
            proc = subprocess.run(cmd, input=message.encode(), capture_output=True,
                                  cwd=self.cwd, env=env, timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise ClipperError(f"claude did not answer within {self.timeout}s") from e
        out = proc.stdout.decode(errors="replace")
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            err = (proc.stderr.decode(errors="replace") or out).strip()[-1500:]
            raise ClipperError(f"claude failed (exit {proc.returncode}): {err}")
        if data.get("is_error") or proc.returncode != 0:
            raise ClipperError(f"claude returned an error: {str(data.get('result') or data)[:1500]}")
        return str(data.get("result", "")), data.get("session_id")

    def ask(self, system: str, prompt: str, parse: Callable[[str], T], what: str = "request") -> T:
        session_id, message = str(uuid.uuid4()), prompt
        for attempt in range(1, self.max_attempts + 1):
            reply, returned_id = self._call(system, message, session_id, resume=attempt > 1)
            session_id = returned_id or session_id
            try:
                return parse(reply)
            except InvalidReply as e:
                log(f"select: {what}: answer didn't match the format (attempt {attempt}): {str(e)[:300]}")
                message = (
                    "Your answer could not be used because it does not match the required format:\n"
                    f"{e}\n\nReply again with only the corrected JSON object and nothing else."
                )
        raise ClipperError(f"Claude's answer for {what} still didn't match the format after "
                           f"{self.max_attempts} tries.")
