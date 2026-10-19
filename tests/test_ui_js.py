"""The web page's job polling and video loading, run in JavaScriptCore with the page's helpers stubbed."""
import shutil
import subprocess
from pathlib import Path

import pytest

JSC = shutil.which("jsc") or "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
APP_JS = Path(__file__).parents[1] / "clipper" / "ui" / "static" / "app.js"

pytestmark = pytest.mark.skipif(not Path(JSC).exists(), reason="needs JavaScriptCore (jsc)")

PRELUDE = r'''
var out = [];
var timers = [];
function setTimeout(f, ms) { timers.push(f); return timers.length; }
function clearTimeout() {}
var resolvers = [];
var jobsAnswer = [{id: "a", state: "done", kind: "render", video: "v1"}];
var api = { get: function (url) {
  return new Promise(function (res) { resolvers.push(function () { res(url === "/api/jobs" ? jobsAnswer : {id: url, sentences: [], rendered: []}); }); });
}};
var S = { jobs: [{id: "a", state: "running", video: "v1"}], openLogs: [], route: {name: "video", id: "v1"}, tab: "clips" };
function updateJobsPill() {} function renderVideoJob() {} function renderJobs() {} function refreshLibrary() {}
function toast(m) { out.push("toast:" + m); }
function renderVideo() { out.push("render:" + S.video.id); }
function initReview() { return null; } function persistReview() {}
function $() { return { replaceChildren: function () { out.push("error-page"); } }; }
function h() {}
var location = { hash: "" };
'''

TAIL = r'''async function flush() { for (var k = 0; k < 20; k++) { while (resolvers.length) resolvers.shift()(); await null; await null; } }
async function main() {
  pollJobs(); pollJobs();          // e.g. a cancel click while a poll is on its way
  await flush();
  var toasts = out.filter(function (x) { return x.indexOf("toast") === 0; }).length;
  out.push("TOASTS=" + toasts);
  // navigating away while a video is loading
  var p = loadVideo("v2", null); S.route = {name: "home"}; await flush(); await p;
  out.push("AFTER-NAV=" + out.filter(function (x) { return x === "render:/api/videos/v2"; }).length);
}
main().then(function () { print(out.join("\n")); });
'''


def _between(src, start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


def test_finished_jobs_are_reported_once_and_stale_videos_dropped(tmp_path):
    src = APP_JS.read_text()
    code = (_between(src, "let pollTimer = null", "function updateJobsPill")
            + _between(src, "function span(it)", "function renderVideoJob"))
    script = tmp_path / "harness.js"
    script.write_text(PRELUDE + code + TAIL)
    out = subprocess.run([JSC, str(script)], capture_output=True, text=True, timeout=30).stdout
    assert "TOASTS=1" in out, out  # two polls at once (a click while one is on its way) report a job once
    assert "AFTER-NAV=0" in out, out  # a video that arrives after leaving its page isn't shown
