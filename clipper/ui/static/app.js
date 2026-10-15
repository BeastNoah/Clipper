"use strict";

// ---------------------------------------------------------------------------------------------
// Small helpers

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.setAttribute("style", v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const api = {
  async get(path) {
    const r = await fetch(path);
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  },
  async post(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  },
};

function fmt(t) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  if (m >= 60) return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtEta(sec) {
  if (sec == null) return "";
  sec = Math.round(sec);
  return sec < 60 ? `${sec}s left` : `${Math.floor(sec / 60)}m ${String(sec % 60).padStart(2, "0")}s left`;
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), 2600);
}

const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
  },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
};

// ---------------------------------------------------------------------------------------------
// State

const S = {
  choices: null,
  jobs: [],
  route: null,
  video: null,        // detail of the open video
  review: null,       // review state of the open video
  tab: "review",
  openLogs: new Set(),
  logs: {},
};

function options() {
  return { ...S.choices.defaults, ...store.get("clipper.options", {}) };
}
function saveOption(key, value) {
  const o = store.get("clipper.options", {});
  o[key] = value;
  store.set("clipper.options", o);
}

// ---------------------------------------------------------------------------------------------
// Jobs polling

let pollTimer = null;
async function pollJobs() {
  clearTimeout(pollTimer);
  try {
    const before = new Map(S.jobs.map(j => [j.id, j.state]));
    S.jobs = await api.get("/api/jobs");
    for (const id of S.openLogs) {
      const job = await api.get(`/api/jobs/${id}`).catch(() => null);
      if (job) S.logs[id] = job.log;
    }
    updateJobsPill();
    const finished = S.jobs.filter(j => before.get(j.id) && before.get(j.id) !== j.state && ["done", "failed", "cancelled"].includes(j.state));
    if (S.route?.name === "home") {
      renderJobs();
      if (finished.length) refreshLibrary();
    } else if (S.route?.name === "video") {
      renderVideoJob();
      if (finished.some(j => j.video === S.route.id)) {
        const job = finished.find(j => j.video === S.route.id);
        if (job.state === "done") toast(job.kind === "render" ? "Clips are ready" : "Clips are ready to review");
        await loadVideo(S.route.id, job.kind === "render" ? "clips" : "review");
      }
    }
    for (const job of finished) {
      if (job.state === "done" && job.kind === "process" && job.video && S.route?.name === "home" && S.followJob === job.id) {
        location.hash = `#/video/${job.video}`;
      }
    }
  } catch (e) { /* server restarting: try again */ }
  const busy = S.jobs.some(j => j.state === "running" || j.state === "queued");
  pollTimer = setTimeout(pollJobs, busy ? 1000 : 4000);
}

function updateJobsPill() {
  const pill = $("#jobs-pill");
  const running = S.jobs.filter(j => j.state === "running" || j.state === "queued");
  pill.hidden = running.length === 0;
  pill.replaceChildren(h("span", { class: "dot" }), `${running.length} job${running.length > 1 ? "s" : ""} running`);
  pill.onclick = () => { location.hash = "#/"; };
}

function jobView(job) {
  const active = job.state === "running" || job.state === "queued";
  const p = job.progress;
  const bar = h("div", { class: "bar" + (active && !p ? " indeterminate" : "") }, h("i", { style: p ? `width:${(p.fraction * 100).toFixed(1)}%` : "" }));
  const open = S.openLogs.has(job.id);
  return h("div", { class: "panel job" },
    h("div", { class: "job-head" },
      h("span", { class: `badge ${job.state}` }, job.state),
      h("div", { class: "job-title grow", title: job.label }, job.label),
      job.video && S.route?.id !== job.video ? h("a", { class: "btn ghost", href: `#/video/${job.video}` }, "Open") : null,
      h("button", { class: "btn ghost", onclick: async () => {
        if (open) S.openLogs.delete(job.id); else { S.openLogs.add(job.id); S.logs[job.id] = (await api.get(`/api/jobs/${job.id}`)).log; }
        renderJobs(); renderVideoJob();
      } }, open ? "Hide log" : "Log"),
      active ? h("button", { class: "btn ghost danger", onclick: async () => { await api.post(`/api/jobs/${job.id}/cancel`, {}); toast("Cancelling…"); pollJobs(); } }, "Cancel") : null,
    ),
    active ? h("div", { class: "small muted", style: "margin-top:8px" },
      [job.step && `Step: ${job.step}`, p && `${p.label} ${Math.round(p.fraction * 100)}%`, p && fmtEta(p.eta)].filter(Boolean).join(" · ") || "Waiting…") : null,
    active ? bar : null,
    job.error ? h("div", { class: "error" }, job.error) : null,
    open ? h("div", { class: "log" }, (S.logs[job.id] || []).join("\n") || "No output yet.") : null,
  );
}

function renderJobs() {
  const box = $("#jobs");
  if (!box) return;
  const recent = S.jobs.slice(0, 8);
  $("#jobs-section").hidden = recent.length === 0;
  box.replaceChildren(...recent.map(jobView));
}

// ---------------------------------------------------------------------------------------------
// Home: new video, jobs, library

function settingsFields(opts, keys) {
  const C = S.choices;
  const sel = (key, label, values, names) => h("label", { class: "field" }, h("span", {}, label),
    h("select", { onchange: e => saveOption(key, e.target.value) },
      values.map((v, i) => h("option", { value: v, selected: String(opts[key]) === String(v) }, names ? names[i] : v))));
  const num = (key, label, step = 1) => h("label", { class: "field" }, h("span", {}, label),
    h("input", { type: "number", min: 1, step, value: opts[key], onchange: e => saveOption(key, Number(e.target.value)) }));
  const chk = (key, label) => h("label", { class: "check" },
    h("input", { type: "checkbox", checked: !!opts[key], onchange: e => saveOption(key, e.target.checked) }), label);
  const fields = {
    num_clips: () => num("num_clips", "Clips"),
    min_len: () => num("min_len", "Min length (s)"),
    max_len: () => num("max_len", "Max length (s)"),
    model: () => sel("model", "Claude model", C.models),
    whisper_model: () => sel("whisper_model", "Transcript model", C.whisper_models),
    caption_model: () => sel("caption_model", "Caption model", ["turbo", "large-v3", "medium", "small", "none"],
      ["turbo (accurate)", "large-v3", "medium", "small", "none (reuse transcript)"]),
    caption_style: () => sel("caption_style", "Caption style", C.caption_styles),
    caption_position: () => sel("caption_position", "Caption position", ["lower", "middle"]),
    layout: () => sel("layout", "Two people", C.layouts,
      ["auto", "follow the speaker", "split screen", "main person only"]),
    jump_cuts: () => sel("jump_cuts", "Jump cuts", C.jump_cuts, ["off", "pauses only", "pauses + filler words"]),
    encoder: () => sel("encoder", "Encoder", C.encoders, ["x264 (best quality)", "VideoToolbox (fast, Mac)", "auto"]),
    language: () => h("label", { class: "field" }, h("span", {}, "Language"),
      h("input", { type: "text", placeholder: "detect", value: opts.language || "", onchange: e => saveOption("language", e.target.value.trim() || null) })),
    uppercase: () => chk("uppercase", "Capital letters"),
    title_overlay: () => chk("title_overlay", "Title at the start"),
    posts: () => chk("posts", "Write post text"),
  };
  return keys.map(k => fields[k]());
}

async function startJob(source) {
  const review = $("#review-first")?.checked ?? true;
  store.set("clipper.reviewFirst", review);
  try {
    const job = await api.post("/api/jobs", { kind: "process", source, review, options: options() });
    S.followJob = job.id;
    S.openLogs.add(job.id);
    toast("Started");
    pollJobs();
  } catch (e) { toast(e.message); }
}

async function upload(file) {
  const label = $("#drop-label");
  label.textContent = `Uploading ${file.name}…`;
  try {
    const r = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", `/api/upload?name=${encodeURIComponent(file.name)}`);
      xhr.upload.onprogress = e => { if (e.lengthComputable) label.textContent = `Uploading ${file.name}… ${Math.round(e.loaded / e.total * 100)}%`; };
      xhr.onload = () => xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(xhr.responseText));
      xhr.onerror = () => reject(new Error("upload failed"));
      xhr.send(file);
    });
    label.textContent = "Drop a video file here, or click to choose one";
    await startJob(r.path);
  } catch (e) {
    label.textContent = "Upload failed. Try again.";
    toast(e.message);
  }
}

async function renderHome() {
  const opts = options();
  const input = h("input", { type: "text", id: "source", placeholder: "Paste a YouTube link, or the path of a video file on this computer", autocomplete: "off" });
  const go = () => { const v = input.value.trim(); if (v) { startJob(v); input.value = ""; } else input.focus(); };
  input.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
  const fileInput = h("input", { type: "file", accept: "video/*", hidden: true, onchange: e => e.target.files[0] && upload(e.target.files[0]) });
  const drop = h("div", { class: "drop", role: "button", tabindex: 0, onclick: () => fileInput.click(),
    onkeydown: e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); },
    ondragover: e => { e.preventDefault(); drop.classList.add("over"); },
    ondragleave: () => drop.classList.remove("over"),
    ondrop: e => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) upload(f); } },
    h("span", { id: "drop-label" }, "Drop a video file here, or click to choose one"), fileInput);

  $("#app").replaceChildren(h("div", { class: "stack" },
    h("section", { class: "panel hero" },
      h("div", {}, h("h1", {}, "Make clips"), h("div", { class: "muted" }, "From a long video to short vertical clips with captions. Claude picks the best moments.")),
      h("div", { class: "source-row" }, input, h("button", { class: "btn primary big", onclick: go }, "Start")),
      drop,
      h("div", { class: "quick" },
        ...settingsFields(opts, ["num_clips", "min_len", "max_len"]),
        h("label", { class: "check", style: "margin-bottom:9px" },
          h("input", { type: "checkbox", id: "review-first", checked: store.get("clipper.reviewFirst", true) }),
          "Let me review the clips before rendering")),
      h("details", {}, h("summary", {}, "More settings"),
        h("div", { class: "settings" }, ...settingsFields(opts, ["model", "whisper_model", "caption_model", "language",
          "caption_style", "caption_position", "layout", "jump_cuts", "encoder"])),
        h("div", { class: "row", style: "margin-top:14px; gap:22px" }, ...settingsFields(opts, ["uppercase", "title_overlay", "posts"]))),
    ),
    h("section", { id: "jobs-section", hidden: true }, h("h2", {}, "Jobs"), h("div", { class: "jobs", id: "jobs" })),
    h("section", {}, h("h2", {}, "Library"), h("div", { id: "library" }, h("div", { class: "muted" }, "Loading…"))),
  ));
  renderJobs();
  refreshLibrary();
}

const STATUS = { ingested: "Downloaded", transcribed: "Transcribed", selected: "Ready to review", rendered: "Clips ready" };

async function refreshLibrary() {
  const box = $("#library");
  if (!box) return;
  const vids = await api.get("/api/videos").catch(() => []);
  if (!vids.length) {
    box.replaceChildren(h("div", { class: "panel empty" }, "No videos yet. Paste a link above to make your first clips."));
    return;
  }
  box.replaceChildren(h("div", { class: "grid" }, vids.map(v => h("a", { class: "card", href: `#/video/${v.id}` },
    h("div", { class: "thumb", style: `background-image:url('/api/videos/${v.id}/frame?t=${Math.round((v.duration || 0) * 0.1)}')` }),
    h("div", { class: "card-body" },
      h("div", { class: "card-title" }, v.title || v.id),
      h("div", { class: "meta" }, h("span", {}, fmt(v.duration)), h("span", { class: "badge" + (v.status === "rendered" ? " done" : "") },
        v.status === "rendered" ? `${v.rendered} clip${v.rendered === 1 ? "" : "s"} ready` : STATUS[v.status] || v.status)))))));
}

// ---------------------------------------------------------------------------------------------
// Video page

function reviewKey(id) { return `clipper.review.${id}`; }

function candidatesSig(cands) {
  return cands.map(c => `${c.id}:${c.si}-${c.ei}`).join(",");
}

function initReview(v) {
  const sentences = v.sentences;
  const sig = candidatesSig(v.candidates);
  const cands = [...v.candidates].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const picked = new Set((v.clips || []).map(c => c.id));
  const items = {};
  for (const c of cands) items[c.id] = { key: c.id, id: c.id, si: c.si, ei: c.ei, title: c.title, selected: picked.has(c.id), cand: c };
  // Edits only make sense for the suggestions they were made on.
  const saved = store.get(reviewKey(v.id), null);
  const fromServer = v.review?.candidates === sig ? v.review.items : null;
  const apply = list => {
    for (const k of Object.keys(items)) items[k].selected = false;
    for (const it of list) {
      const key = it.key || it.id || `u${it.si}-${it.ei}`;
      if (!items[key]) items[key] = { key, id: null, cand: null, title: it.title || sentences[it.si]?.text.slice(0, 50) };
      Object.assign(items[key], { si: it.si, ei: it.ei, title: it.title ?? items[key].title, selected: it.selected ?? true });
    }
  };
  if (saved && saved.items && saved.sig === sig) apply(saved.items);
  else if (fromServer) apply(fromServer);
  const order = [...cands.map(c => c.id), ...Object.keys(items).filter(k => !items[k].cand)];
  return { items, order, sig, mark: null, markEnd: null, focus: null };
}

function persistReview() {
  const r = S.review;
  store.set(reviewKey(S.video.id), { sig: r.sig, items: r.order.map(k => r.items[k]).filter(it => it.selected || !it.cand || it.si !== it.cand.si || it.ei !== it.cand.ei || it.title !== it.cand.title)
    .map(({ key, id, si, ei, title, selected }) => ({ key, id, si, ei, title, selected })) });
}

function span(it) {
  const s = S.video.sentences;
  return { start: s[it.si].start, end: s[it.ei].end };
}

async function loadVideo(id, tab) {
  try {
    const v = await api.get(`/api/videos/${id}`);
    S.video = v;
    S.video.version = Date.now();
    S.review = v.sentences.length ? initReview(v) : null;
    if (tab) S.tab = tab;
    else if (!S.tab) S.tab = v.rendered?.some(c => c.file) ? "clips" : "review";
    renderVideo();
  } catch (e) {
    $("#app").replaceChildren(h("div", { class: "panel empty" }, `Couldn't open this video: ${e.message}`));
  }
}

function renderVideoJob() {
  const box = $("#video-job");
  if (!box || !S.video) return;
  const jobs = S.jobs.filter(j => j.video === S.video.id && (j.state === "running" || j.state === "queued" || (j.state === "failed" && Date.now() / 1000 - j.finished < 600)));
  box.replaceChildren(...jobs.slice(0, 2).map(jobView));
}

function renderVideo() {
  const v = S.video;
  const nRendered = (v.rendered || []).filter(c => c.file).length;
  const tabs = h("div", { class: "tabs", role: "tablist" },
    h("button", { class: S.tab === "review" ? "active" : "", onclick: () => { S.tab = "review"; renderVideo(); } }, `Review (${v.candidates.length})`),
    h("button", { class: S.tab === "clips" ? "active" : "", onclick: () => { S.tab = "clips"; renderVideo(); } }, `Clips (${nRendered})`));
  const head = h("div", { class: "video-head" },
    h("div", {}, h("a", { href: "#/", class: "small muted" }, "← Library"),
      h("h1", {}, v.meta?.title || v.id),
      h("div", { class: "muted small" }, [fmt(v.meta?.duration), v.language && `language: ${v.language}`, v.meta?.source].filter(Boolean).join(" · "))),
    tabs);
  const body = S.tab === "clips" ? clipsView() : reviewView();
  $("#app").replaceChildren(head, h("div", { id: "video-job", class: "jobs", style: "margin-bottom:16px" }), body);
  renderVideoJob();
}

// -- review

function reviewView() {
  const v = S.video, r = S.review;
  if (!r) return h("div", { class: "panel empty" }, "This video hasn't been transcribed yet.");
  const player = h("video", { class: "player", id: "player", src: `/media/${v.id}/video.mp4`, controls: true, preload: "metadata" });
  player.addEventListener("timeupdate", () => {
    if (S.stopAt != null && player.currentTime >= S.stopAt) { player.pause(); S.stopAt = null; }
  });
  const side = h("div", { class: "side" },
    h("div", { class: "panel player-panel", style: "padding:12px" }, player, h("div", { class: "preview-label", id: "preview-label", style: "margin-top:8px" }, "Click ▶ on a clip to preview it here.")),
    h("div", { class: "panel transcript-panel", style: "padding:12px 8px" },
      h("div", { class: "row", style: "padding:0 8px 8px" }, h("h2", { style: "margin:0" }, "Transcript"),
        h("span", { class: "small muted grow" }, "Click a sentence to start your own clip, then click where it should end.")),
      h("div", { class: "transcript", id: "transcript" }),
      h("div", { class: "custom-bar", id: "custom-bar", style: "padding:0 8px" })));

  const list = h("div", { class: "cands", id: "cands" });
  const wrap = h("div", { class: "review" }, side, h("div", { class: "cands-col" }, list));
  setTimeout(() => { renderCands(); renderTranscript(); renderCustomBar(); renderRenderBar(); }, 0);
  return h("div", {}, wrap, h("div", { class: "renderbar", id: "renderbar" }));
}

function preview(it, n) {
  const p = $("#player");
  if (!p) return;
  const { start, end } = span(it);
  p.currentTime = Math.max(0, start - 0.2);
  S.stopAt = end + 0.4;
  p.play().catch(() => {});
  $("#preview-label").textContent = `Previewing ${n ? `#${n}` : ""} “${it.title}” · ${fmt(start)}–${fmt(end)}`;
  S.review.focus = it.key;
  renderTranscript(true);
}

function candView(it, n) {
  const c = it.cand;
  const { start, end } = span(it);
  const nS = S.video.sentences.length;
  const change = (fn) => () => { fn(); persistReview(); renderCands(); renderRenderBar(); if (S.review.focus === it.key) renderTranscript(); };
  const subs = c?.scores ? h("div", { class: "subs" }, [["hook", "Hook"], ["standalone", "Standalone"], ["emotion", "Emotion"], ["clarity", "Clarity"]].map(([k, label]) =>
    h("div", { class: "sub" }, `${label} ${c.scores[k] ?? "–"}`, h("div", { class: "bar" }, h("i", { style: `width:${((c.scores[k] ?? 0) / 25) * 100}%` }))))) : null;
  const moved = c && (it.si !== c.si || it.ei !== c.ei);
  return h("div", { class: "cand" + (it.selected ? " selected" : "") + (c?.duplicate_of ? " dimmed" : "") },
    h("input", { type: "checkbox", "aria-label": "Use this clip", checked: it.selected, onchange: e => { it.selected = e.target.checked; persistReview(); renderCands(); renderRenderBar(); } }),
    h("div", { class: "thumb", style: `background-image:url('/api/videos/${S.video.id}/frame?t=${Math.round(start + 1)}')` }),
    h("div", {},
      h("input", { class: "cand-title", value: it.title, "aria-label": "Clip title", onchange: e => { it.title = e.target.value; persistReview(); } }),
      h("div", { class: "meta" },
        c?.score != null ? h("span", { class: "score", title: "Claude's score out of 100" }, c.score) : h("span", { class: "badge" }, "your clip"),
        h("span", {}, `${fmt(start)}–${fmt(end)}`), h("span", {}, `${Math.round(end - start)}s`),
        moved ? h("span", { class: "chip" }, "adjusted") : null),
      subs,
      c?.hook ? h("div", { class: "hook" }, `“${c.hook}”`) : null,
      h("div", { class: "chips" },
        ...(c?.keywords || []).map(k => h("span", { class: "chip key", title: "Highlighted in the captions" }, k)),
        c?.duplicate_of ? h("span", { class: "chip warn" }, `Similar to a better clip (${c.duplicate_of})`) : null,
        c?.trimmed_by_claude ? h("span", { class: "chip", title: "Claude chose which sentences to keep" }, "shortened by Claude") : null),
      h("div", { class: "actions" },
        h("button", { class: "btn icon", onclick: () => preview(it, n), "aria-label": "Preview" }, "▶ Preview"),
        h("span", { class: "nudge" }, "Start",
          h("button", { class: "btn", title: "One sentence earlier", disabled: it.si <= 0, onclick: change(() => it.si--) }, "◀"),
          h("button", { class: "btn", title: "One sentence later", disabled: it.si >= it.ei, onclick: change(() => it.si++) }, "▶")),
        h("span", { class: "nudge" }, "End",
          h("button", { class: "btn", title: "One sentence earlier", disabled: it.ei <= it.si, onclick: change(() => it.ei--) }, "◀"),
          h("button", { class: "btn", title: "One sentence later", disabled: it.ei >= nS - 1, onclick: change(() => it.ei++) }, "▶")),
        moved ? h("button", { class: "btn ghost small", onclick: change(() => { it.si = c.si; it.ei = c.ei; }) }, "Reset") : null,
        !c ? h("button", { class: "btn ghost danger", onclick: change(() => { delete S.review.items[it.key]; S.review.order = S.review.order.filter(k => k !== it.key); }) }, "Remove") : null),
      h("details", { class: "cand-text" }, h("summary", {}, "Transcript"),
        S.video.sentences.slice(it.si, it.ei + 1).map(s => s.text).join(" "))));
}

function renderCands() {
  const box = $("#cands");
  if (!box) return;
  const r = S.review;
  if (!r.order.length) {
    box.replaceChildren(h("div", { class: "panel empty" }, "No clip suggestions yet. Make your own from the transcript."));
    return;
  }
  let n = 0;
  box.replaceChildren(...r.order.map(k => { const it = r.items[k]; if (it.selected) n++; return candView(it, it.selected ? n : null); }));
}

function renderTranscript(scroll) {
  const box = $("#transcript");
  if (!box) return;
  const r = S.review;
  const focus = r.focus && r.items[r.focus];
  const lo = r.mark ?? -1, hi = r.markEnd ?? r.mark ?? -1;
  let target = null;
  box.replaceChildren(...S.video.sentences.map(s => {
    const inFocus = focus && s.i >= focus.si && s.i <= focus.ei;
    const inMark = s.i >= lo && s.i <= hi && lo >= 0;
    const el = h("button", { class: "sent" + (inFocus || inMark ? " in" : "") + (s.i === r.mark ? " mark" : ""),
      onclick: () => clickSentence(s.i) }, h("span", { class: "t" }, fmt(s.start)), s.text);
    if ((inFocus && focus.si === s.i) || (inMark && s.i === lo)) target = target || el;
    return el;
  }));
  if (scroll && target) box.scrollTop = target.offsetTop - box.offsetTop - 40;
}

function clickSentence(i) {
  const r = S.review;
  const p = $("#player");
  if (p) { p.currentTime = S.video.sentences[i].start; S.stopAt = null; }
  if (r.mark == null || r.markEnd != null || i < r.mark) { r.mark = i; r.markEnd = null; }
  else r.markEnd = i;
  r.focus = null;
  renderTranscript();
  renderCustomBar();
}

function renderCustomBar() {
  const box = $("#custom-bar");
  if (!box) return;
  const r = S.review, s = S.video.sentences;
  if (r.mark == null) { box.replaceChildren(); return; }
  const end = r.markEnd ?? r.mark;
  const dur = s[end].end - s[r.mark].start;
  box.replaceChildren(
    h("span", { class: "small" }, r.markEnd == null ? `Start: ${fmt(s[r.mark].start)} · now click the last sentence` : `${fmt(s[r.mark].start)}–${fmt(s[end].end)} · ${Math.round(dur)}s`),
    r.markEnd != null ? h("button", { class: "btn primary", onclick: () => {
      const key = `u${r.mark}-${end}`;
      r.items[key] = { key, id: null, cand: null, si: r.mark, ei: end, title: s[r.mark].text.slice(0, 50), selected: true };
      if (!r.order.includes(key)) r.order.push(key);
      r.mark = r.markEnd = null;
      persistReview(); renderCands(); renderTranscript(); renderCustomBar(); renderRenderBar();
      toast("Clip added");
    } }, "Add as clip") : null,
    h("button", { class: "btn ghost", onclick: () => { r.mark = r.markEnd = null; renderTranscript(); renderCustomBar(); } }, "Clear"));
}

function renderRenderBar() {
  const box = $("#renderbar");
  if (!box) return;
  const r = S.review;
  const chosen = r.order.map(k => r.items[k]).filter(it => it.selected);
  const total = chosen.reduce((a, it) => { const { start, end } = span(it); return a + end - start; }, 0);
  const opts = options();
  const running = S.jobs.some(j => j.video === S.video.id && (j.state === "running" || j.state === "queued"));
  box.replaceChildren(h("div", { class: "inner" },
    h("div", { class: "grow" }, h("strong", {}, `${chosen.length} clip${chosen.length === 1 ? "" : "s"}`),
      h("span", { class: "muted" }, chosen.length ? ` · ${fmt(total)} in total` : "")),
    ...settingsFields(opts, ["caption_style", "layout", "jump_cuts"]).map(f => {
      const label = f.querySelector("span").textContent, sel = f.querySelector("select");
      sel.title = label; sel.setAttribute("aria-label", label); sel.classList.add("bar-option");
      return sel;
    }),
    h("button", { class: "btn primary big", disabled: !chosen.length || running, onclick: () => renderClips(chosen) },
      running ? "Working…" : `Render ${chosen.length || ""}`)));
}

async function renderClips(chosen) {
  try {
    await api.post("/api/jobs", { kind: "render", video: S.video.id, options: options(),
      items: chosen.map(it => ({ id: it.id, si: it.si, ei: it.ei, title: it.title })) });
    toast("Rendering started");
    S.tab = "clips";
    renderVideo();
    pollJobs();
  } catch (e) { toast(e.message); }
}

// -- rendered clips

function clipsView() {
  const v = S.video;
  const clips = (v.rendered || []).filter(c => c.file);
  if (!clips.length) return h("div", { class: "panel empty" }, "No clips rendered yet. Pick clips in Review and press Render.");
  return h("div", { class: "clips" }, clips.map(c => {
    const src = `/media/${v.id}/output/${encodeURIComponent(c.file)}?v=${v.version}`;
    const cover = c.cover ? `/media/${v.id}/output/${encodeURIComponent(c.cover)}?v=${v.version}` : null;
    const text = h("div", { class: "post-text" });
    const platforms = c.post ? [["TikTok", c.post.tiktok], ["Instagram", c.post.instagram],
      ["YouTube", `${c.post.youtube_title}\n\n${c.post.youtube_description}`]] : [];
    const tabs = h("div", { class: "post-tabs" });
    const show = i => {
      text.textContent = platforms[i][1];
      [...tabs.children].forEach((b, j) => b.classList.toggle("active", i === j));
    };
    platforms.forEach(([name], i) => tabs.append(h("button", { onclick: () => show(i) }, name)));
    if (platforms.length) setTimeout(() => show(0), 0);
    return h("div", { class: "panel clip-card" },
      h("video", { class: "vertical", src, poster: cover, controls: true, preload: "metadata", playsinline: true }),
      h("div", { style: "margin-top:12px; font-weight:650" }, `${c.rank}. ${c.title}`),
      h("div", { class: "meta" }, h("span", {}, `${Math.round(c.final_duration || c.duration)}s`),
        c.score != null ? h("span", { class: "score" }, c.score) : null, h("span", {}, `from ${fmt(c.start)}`)),
      h("div", { class: "row", style: "gap:8px" },
        h("a", { class: "btn primary", href: src, download: c.file }, "Download"),
        cover ? h("a", { class: "btn", href: cover, download: c.cover }, "Cover") : null),
      platforms.length ? h("div", { class: "post" }, tabs, text,
        h("button", { class: "btn ghost small", style: "margin-top:6px", onclick: async () => {
          try { await navigator.clipboard.writeText(text.textContent); toast("Copied"); } catch { toast("Couldn't copy"); }
        } }, "Copy text")) : null);
  }));
}

// ---------------------------------------------------------------------------------------------
// Router

async function route() {
  const m = location.hash.match(/^#\/video\/([a-z0-9-]+)(?:\/(clips|review))?/);
  document.querySelectorAll("[data-nav]").forEach(a => a.classList.toggle("active", !m));
  if (m) {
    const same = S.route?.name === "video" && S.route.id === m[1];
    S.route = { name: "video", id: m[1] };
    if (!same) S.tab = m[2] || null;
    await loadVideo(m[1], m[2]);
  } else {
    S.route = { name: "home" };
    S.video = null;
    await renderHome();
  }
  window.scrollTo(0, 0);
}

(async function init() {
  S.choices = await api.get("/api/choices");
  window.addEventListener("hashchange", route);
  await route();
  pollJobs();
})();
