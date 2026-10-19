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
import time
import uuid
from pathlib import Path
from typing import Callable, TypeVar

from .util import Cancelled, ClipperError, check_cancel, log, run_proc

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


class ClaudeFailed(ClipperError):
    """The claude command itself failed (timed out, crashed, or reported an error); worth another try."""


class Claude:
    retry_delays = (5, 20)  # seconds to wait before the 2nd and 3rd try after a failed call

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
            code, stdout, stderr = run_proc(cmd, input=message.encode(), cwd=self.cwd, env=env,
                                            timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise ClaudeFailed(f"claude did not answer within {self.timeout}s") from e
        out = stdout.decode(errors="replace")
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            err = (stderr.decode(errors="replace") or out).strip()[-1500:]
            raise ClaudeFailed(f"claude failed (exit {code}): {err}")
        if not isinstance(data, dict) or data.get("is_error") or code != 0:
            detail = data.get("result") or data if isinstance(data, dict) else data
            raise ClaudeFailed(f"claude returned an error: {str(detail)[:1500]}")
        return str(data.get("result", "")), data.get("session_id")

    def _call_retrying(self, system: str, message: str, session_id: str, resume: bool,
                       what: str) -> tuple[str, str | None]:
        for attempt, delay in enumerate((*self.retry_delays, None), 1):
            try:
                return self._call(system, message, session_id, resume)
            except Cancelled:
                raise
            except ClaudeFailed as e:
                if delay is None:
                    raise
                log(f"select: {what}: {str(e)[:300]}; trying again in {delay}s")
                end = time.monotonic() + delay
                while time.monotonic() < end:
                    check_cancel()
                    time.sleep(min(0.25, max(0.0, end - time.monotonic())))
                if not resume:  # the failed call may have started a conversation with this id
                    session_id = str(uuid.uuid4())
        raise AssertionError("unreachable")

    def ask(self, system: str, prompt: str, parse: Callable[[str], T], what: str = "request") -> T:
        session_id, message = str(uuid.uuid4()), prompt
        for attempt in range(1, self.max_attempts + 1):
            reply, returned_id = self._call_retrying(system, message, session_id, attempt > 1, what)
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
