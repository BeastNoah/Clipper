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

let pollTimer = null, polling = false, pollAgain = false;
async function pollJobs() {
  clearTimeout(pollTimer);
  // One poll at a time: two at once would both see a job finish and report it twice.
  if (polling) { pollAgain = true; return; }
  polling = true;
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
        if (job.kind === "publish") {
          toast(job.state === "done" ? job.label.replace(/^Post to/, "Posted to") : job.state === "failed" ? `Posting failed: ${job.error}` : "Posting cancelled");
        } else if (job.state === "done") {
          toast(job.kind === "render" ? "Clips are ready" : job.kind === "suggest" ? "New suggestions are ready" : "Clips are ready to review");
        }
        await loadVideo(S.route.id, job.kind === "render" || job.kind === "publish" ? "clips" : "review");
      }
    }
    for (const job of finished) {
      if (job.state === "done" && job.kind === "process" && job.video && S.route?.name === "home" && S.followJob === job.id) {
        location.hash = `#/video/${job.video}`;
      }
    }
  } catch (e) { /* server restarting: try again */ }
  polling = false;
  if (pollAgain) { pollAgain = false; return pollJobs(); }
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
  const txt = (key, label, placeholder) => h("label", { class: "field" }, h("span", {}, label),
    h("input", { type: "text", placeholder, value: opts[key] ?? "", onchange: e => saveOption(key, e.target.value.trim() || null) }));
  const formats = () => h("div", { class: "field" }, h("span", {}, "Formats"),
    h("div", { class: "row", style: "gap:12px" }, C.formats.map(f => h("label", { class: "check" },
      h("input", { type: "checkbox", checked: (opts.formats || ["9:16"]).includes(f), onchange: e => {
        const now = new Set(options().formats || ["9:16"]);
        e.target.checked ? now.add(f) : now.delete(f);
        if (!now.size) { e.target.checked = true; return toast("Keep at least one format"); }
        saveOption("formats", C.formats.filter(x => now.has(x)));
      } }), f))));
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
    encoder: () => sel("encoder", "Encoder", C.encoders, ["x264 (best quality)", "VideoToolbox (fast, Mac)", "auto (VideoToolbox on a Mac)"]),
    download: () => sel("download", "Video", ["sections", "full"], ["only the clips' parts (fast)", "convert the whole video first"]),
    language: () => txt("language", "Spoken language", "detect"),
    caption_language: () => txt("caption_language", "Translate captions to", "e.g. English"),
    range_start: () => txt("range_start", "Only from", "0:00"),
    range_end: () => txt("range_end", "Until", "the end"),
    logo: () => txt("logo", "Logo file", "/path/to/logo.png"),
    logo_position: () => sel("logo_position", "Logo position", ["top-right", "top-left", "bottom-right", "bottom-left"]),
    formats,
    zoom: () => chk("zoom", "Auto-zoom on key moments"),
    progress_bar: () => chk("progress_bar", "Progress bar"),
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
    const focus = $("#focus")?.value.trim() || null;
    const job = await api.post("/api/jobs", { kind: "process", source, review, options: { ...options(), prompt: focus } });
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
    label.textContent = DROP_TEXT;
    await startJob(r.path);
  } catch (e) {
    label.textContent = "Upload failed. Try again.";
    toast(e.message);
  }
}

const TOUCH = window.matchMedia("(pointer: coarse)").matches;
const DROP_TEXT = TOUCH ? "Tap to choose a video from this phone" : "Drop a video file here, or click to choose one";

async function renderHome() {
  const opts = options();
  const input = h("input", { type: "text", id: "source", autocomplete: "off",
    placeholder: TOUCH ? "Paste a YouTube link" : "Paste a YouTube link, or the path of a video file on this computer" });
  const go = () => { const v = input.value.trim(); if (v) { startJob(v); input.value = ""; } else input.focus(); };
  input.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
  const fileInput = h("input", { type: "file", accept: "video/*", hidden: true, onchange: e => e.target.files[0] && upload(e.target.files[0]) });
  const drop = h("div", { class: "drop", role: "button", tabindex: 0, onclick: () => fileInput.click(),
    onkeydown: e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); },
    ondragover: e => { e.preventDefault(); drop.classList.add("over"); },
    ondragleave: () => drop.classList.remove("over"),
    ondrop: e => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) upload(f); } },
    h("span", { id: "drop-label" }, DROP_TEXT), fileInput);

  $("#app").replaceChildren(h("div", { class: "stack" },
    h("section", { class: "panel hero" },
      h("div", {}, h("h1", {}, "Make clips"), h("div", { class: "muted" }, "From a long video to short vertical clips with captions. Claude picks the best moments.")),
      h("div", { class: "source-row" }, input, h("button", { class: "btn primary big", onclick: go }, "Start")),
      h("input", { type: "text", id: "focus", autocomplete: "off",
        placeholder: "Optional: what should Claude look for? e.g. funny moments, everything about pricing" }),
      drop,
      h("div", { class: "quick" },
        ...settingsFields(opts, ["num_clips", "min_len", "max_len"]),
        h("label", { class: "check", style: "margin-bottom:9px" },
          h("input", { type: "checkbox", id: "review-first", checked: store.get("clipper.reviewFirst", true) }),
          "Let me review the clips before rendering")),
      h("details", {}, h("summary", {}, "More settings"),
        h("div", { class: "settings" }, ...settingsFields(opts, ["range_start", "range_end", "model", "whisper_model",
          "caption_model", "language", "caption_language", "caption_style", "caption_position", "layout", "jump_cuts",
          "encoder", "download", "logo", "logo_position"])),
        h("div", { style: "margin-top:14px" }, ...settingsFields(opts, ["formats"])),
        h("div", { class: "row", style: "margin-top:14px; gap:22px" }, ...settingsFields(opts, ["uppercase", "title_overlay",
          "zoom", "progress_bar", "posts"]))),
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
  // "Find new clips" only adds suggestions, so a review made on an earlier list still applies as long as
  // every suggestion it saw is still there unchanged.
  const current = new Set(sig.split(",").filter(Boolean));
  const covers = old => old != null && old.split(",").filter(Boolean).every(x => current.has(x));
  const latest = Math.max(1, ...v.candidates.map(c => c.search || 1));
  const order = c => c.order ?? Infinity;  // Claude's ranking of all candidates, where there is one
  const cands = [...v.candidates].sort((a, b) => (b.search || 1) - (a.search || 1) || order(a) - order(b) || (b.score ?? 0) - (a.score ?? 0));
  const picked = new Set((v.clips || []).map(c => c.id));
  const items = {};
  for (const c of cands) items[c.id] = { key: c.id, id: c.id, si: c.si, ei: c.ei, title: c.title, selected: picked.has(c.id), cand: c };
  // Edits only make sense for the suggestions they were made on.
  const saved = store.get(reviewKey(v.id), null);
  const fromServer = covers(v.review?.candidates) ? v.review.items : null;
  const apply = list => {
    for (const k of Object.keys(items)) items[k].selected = false;
    for (const it of list) {
      const key = it.key || it.id || `u${it.si}-${it.ei}`;
      if (!items[key]) items[key] = { key, id: null, cand: null, title: it.title || sentences[it.si]?.text.slice(0, 50) };
      Object.assign(items[key], { si: it.si, ei: it.ei, title: it.title ?? items[key].title, selected: it.selected ?? true });
    }
  };
  let seen = null;
  if (saved && saved.items && covers(saved.sig)) { apply(saved.items); seen = saved.sig; }
  else if (fromServer) { apply(fromServer); seen = v.review.candidates; }
  if (seen != null) {  // Claude's picks from a search made after that review start ticked
    const before = new Set(seen.split(",").filter(Boolean).map(x => x.split(":")[0]));
    for (const id of picked) if (items[id] && !before.has(id)) items[id].selected = true;
  }
  const order = [...cands.map(c => c.id), ...Object.keys(items).filter(k => !items[k].cand)];
  return { items, order, sig, latest, mark: null, markEnd: null, focus: null };
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

let videoRequest = 0;
async function loadVideo(id, tab) {
  const request = ++videoRequest;
  // An answer that arrives after the user went elsewhere (or asked again) is dropped.
  const stale = () => request !== videoRequest || S.route?.name !== "video" || S.route.id !== id;
  try {
    const v = await api.get(`/api/videos/${id}`);
    if (stale()) return;
    S.video = v;
    S.video.version = Date.now();
    S.review = v.sentences.length ? initReview(v) : null;
    if (S.review) persistReview();  // so the ticks survive when "Find new clips" adds suggestions
    if (tab) S.tab = tab;
    else if (!S.tab) S.tab = v.rendered?.some(c => c.file) ? "clips" : "review";
    renderVideo();
  } catch (e) {
    if (stale()) return;
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
  const player = makePlayer(v);
  const side = h("div", { class: "side" },
    h("div", { class: "panel player-panel", style: "padding:12px" }, player, h("div", { class: "preview-label", id: "preview-label", style: "margin-top:8px" }, "Click ▶ on a clip to preview it here.")),
    h("div", { class: "panel transcript-panel", style: "padding:12px 8px" },
      h("div", { class: "row", style: "padding:0 8px 8px" }, h("h2", { style: "margin:0" }, "Transcript"),
        h("span", { class: "small muted grow" }, "Click a sentence to start your own clip, then click where it should end.")),
      h("div", { class: "transcript", id: "transcript" }),
      h("div", { class: "custom-bar", id: "custom-bar", style: "padding:0 8px" })));

  const list = h("div", { class: "cands", id: "cands" });
  const wrap = h("div", { class: "review" }, side, h("div", { class: "cands-col" }, findPanel(), list));
  setTimeout(() => { renderCands(); renderTranscript(); renderCustomBar(); renderRenderBar(); }, 0);
  return h("div", {}, wrap, h("div", { class: "renderbar", id: "renderbar" }));
}

// Ask Claude for (more) clips of a video that's already transcribed. Earlier suggestions and finished clips stay.
function findPanel() {
  const v = S.video, has = v.candidates.length > 0;
  const running = S.jobs.some(j => j.video === v.id && (j.state === "running" || j.state === "queued"));
  S.findFocus = S.findFocus || {};
  const focus = h("input", { type: "text", class: "find-focus", autocomplete: "off", value: S.findFocus[v.id] || "",
    placeholder: "Optional: what should Claude look for? e.g. funny moments", oninput: e => { S.findFocus[v.id] = e.target.value; } });
  const render = h("input", { type: "checkbox", checked: store.get("clipper.findRender", false) });
  const button = h("button", { class: "btn primary", disabled: running, onclick: () => go() },
    running ? "Working…" : has ? "Find new clips" : "Find clips");
  const go = async () => {
    if (button.disabled) return;
    button.disabled = true;
    button.textContent = "Working…";
    store.set("clipper.findRender", render.checked);
    try {
      await api.post("/api/jobs", { kind: "new_clips", video: v.id, render: render.checked,
        options: { ...options(), prompt: focus.value.trim() || null } });
      toast(render.checked ? "Finding and rendering new clips" : "Asking Claude for new clips");
      pollJobs();
    } catch (e) {
      toast(e.message);
      button.disabled = false;
      button.textContent = has ? "Find new clips" : "Find clips";
    }
  };
  focus.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
  return h("div", { class: "panel find" },
    h("div", {}, h("h2", {}, has ? "Find new clips" : "Find clips"),
      h("div", { class: "small muted" }, has
        ? "Claude looks for moments it hasn't suggested yet. The suggestions below and your finished clips stay."
        : "Claude reads the transcript and suggests the best moments.")),
    focus,
    h("div", { class: "quick" }, ...settingsFields(options(), ["num_clips", "min_len", "max_len"])),
    h("details", {}, h("summary", {}, "More"),
      h("div", { class: "settings" }, ...settingsFields(options(), ["range_start", "range_end", "model"]))),
    h("div", { class: "row" },
      h("label", { class: "check grow" }, render, "Render them right away"), button));
}

// Without a downloaded video (only the clip sections get fetched), YouTube's own player previews the
// moments; for other sites the downloaded sound does. A video on this computer plays from the original
// file, or from its sound if the browser can't play that file type.
function makePlayer(v) {
  const sections = v.meta?.video === "sections";
  if (sections && v.meta.youtube_id) {
    return h("iframe", { class: "player", id: "player", title: "Preview", allowfullscreen: true,
      allow: "autoplay; encrypted-media; picture-in-picture", src: youtubeSrc(v.meta.youtube_id, 0, null, false) });
  }
  const audio = `/media/${v.id}/audio.wav`;
  const el = sections && !v.meta.local
    ? h("audio", { class: "player audio", id: "player", src: audio, controls: true, preload: "metadata" })
    : h("video", { class: "player", id: "player", controls: true, preload: "metadata",
        src: `/media/${v.id}/${sections ? "source" : "video.mp4"}` });
  if (sections && v.meta.local) {
    el.addEventListener("error", () => { if (!el.src.endsWith("audio.wav")) el.src = audio; });
  }
  el.addEventListener("timeupdate", () => {
    if (S.stopAt != null && el.currentTime >= S.stopAt) { el.pause(); S.stopAt = null; }
  });
  return el;
}

function youtubeSrc(id, start, end, autoplay) {
  const q = new URLSearchParams({ start: Math.floor(Math.max(0, start)), autoplay: autoplay ? 1 : 0, rel: 0, playsinline: 1 });
  if (end != null) q.set("end", Math.ceil(end));
  return `https://www.youtube-nocookie.com/embed/${encodeURIComponent(id)}?${q}`;
}

function seekPlayer(t, stopAt, play) {
  const p = $("#player");
  if (!p) return;
  if (p.tagName === "IFRAME") {
    p.src = youtubeSrc(S.video.meta.youtube_id, t, stopAt, play);
    return;
  }
  p.currentTime = Math.max(0, t);
  S.stopAt = stopAt;
  if (play) p.play().catch(() => {});
}

function preview(it, n) {
  const { start, end } = span(it);
  seekPlayer(start - 0.2, end + 0.4, true);
  $("#preview-label").textContent = `Previewing ${n ? `#${n}` : ""} “${it.title}” · ${fmt(start)}–${fmt(end)}`;
  S.review.focus = it.key;
  renderTranscript(true);
}

function candView(it, n) {
  const c = it.cand;
  const { start, end } = span(it);
  const nS = S.video.sentences.length;
  const change = (fn) => () => { fn(); persistReview(); renderCands(); renderRenderBar(); if (S.review.focus === it.key) renderTranscript(); };
  const subs = c?.scores ? h("div", { class: "subs" }, [["hook", "Hook"], ["payoff", "Payoff"], ["reaction", "Reaction"], ["pace", "Pace"]].map(([k, label]) =>
    h("div", { class: "sub" }, `${label} ${c.scores[k] ?? "–"}`, h("div", { class: "bar" }, h("i", { style: `width:${((c.scores[k] ?? 0) / 10) * 100}%` }))))) : null;
  const moved = c && (it.si !== c.si || it.ei !== c.ei);
  return h("div", { class: "cand" + (it.selected ? " selected" : "") + (c?.duplicate_of ? " dimmed" : "") },
    h("input", { type: "checkbox", "aria-label": "Use this clip", checked: it.selected, onchange: e => { it.selected = e.target.checked; persistReview(); renderCands(); renderRenderBar(); } }),
    h("div", { class: "thumb", style: `background-image:url('/api/videos/${S.video.id}/frame?t=${Math.round(start + 1)}')` }),
    h("div", {},
      h("input", { class: "cand-title", value: it.title, "aria-label": "Clip title", onchange: e => { it.title = e.target.value; persistReview(); } }),
      h("div", { class: "meta" },
        c?.score != null ? h("span", { class: "score", title: "Claude's score out of 100: hook × payoff × (reaction and pace)" }, c.score) : h("span", { class: "badge" }, "your clip"),
        c?.type && c.type !== "other" ? h("span", { class: "chip", title: "The kind of moment" }, c.type) : null,
        h("span", {}, `${fmt(start)}–${fmt(end)}`), h("span", {}, `${Math.round(end - start)}s`),
        moved ? h("span", { class: "chip" }, "adjusted") : null),
      subs,
      c?.hook ? h("div", { class: "hook" }, `“${c.hook}”`) : null,
      h("div", { class: "chips" },
        c && S.review.latest > 1 && (c.search || 1) === S.review.latest ? h("span", { class: "chip new" }, "new") : null,
        c?.focus ? h("span", { class: "chip", title: "What Claude was asked to look for" }, `for: ${c.focus}`) : null,
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
    box.replaceChildren(h("div", { class: "panel empty" }, "No clip suggestions yet. Ask Claude above, or make your own from the transcript."));
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
  seekPlayer(S.video.sentences[i].start, null, false);
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

const RATIO_CSS = { "9:16": "9 / 16", "4:5": "4 / 5", "1:1": "1 / 1", "16:9": "16 / 9" };

function clipsView() {
  const v = S.video;
  const clips = (v.rendered || []).filter(c => c.file);
  if (!clips.length) return h("div", { class: "panel empty" }, "No clips rendered yet. Pick clips in Review and press Render.");
  return h("div", { class: "clips" }, clips.map(c => clipCard(v, c)));
}

function clipCard(v, c) {
  const media = name => `/media/${v.id}/output/${encodeURIComponent(name)}?v=${v.version}`;
  const formats = c.formats ? Object.entries(c.formats) : [["9:16", { file: c.file, cover: c.cover }]];
  const player = h("video", { class: "vertical", controls: true, preload: "metadata", playsinline: true });
  const download = h("a", { class: "btn primary" }, "Download");
  const coverLink = h("a", { class: "btn" }, "Cover");
  const pills = h("div", { class: "post-tabs" });
  const pick = i => {
    const [ratio, f] = formats[i];
    player.src = media(f.file);
    player.poster = f.cover ? media(f.cover) : "";
    player.style.aspectRatio = RATIO_CSS[ratio] || "9 / 16";
    download.href = media(f.file); download.download = f.file;
    coverLink.href = f.cover ? media(f.cover) : "#"; coverLink.download = f.cover || "";
    [...pills.children].forEach((b, j) => b.classList.toggle("active", i === j));
  };
  formats.forEach(([ratio], i) => pills.append(h("button", { onclick: () => pick(i) }, ratio)));
  setTimeout(() => pick(0), 0);

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

  const editor = h("div", { class: "caption-editor", hidden: true });
  const publishBox = h("div", { class: "publish", hidden: true });
  const posted = Object.entries((v.published || {})[c.file] || {}).filter(([, r]) => r.status !== "failed");
  const card = h("div", { class: "panel clip-card" },
    player,
    formats.length > 1 ? h("div", { style: "margin-top:8px" }, pills) : null,
    h("div", { style: "margin-top:12px; font-weight:650" }, `${c.rank}. ${c.title}`),
    h("div", { class: "meta" }, h("span", {}, `${Math.round(c.final_duration || c.duration)}s`),
      c.score != null ? h("span", { class: "score" }, c.score) : null, h("span", {}, `from ${fmt(c.start)}`)),
    h("div", { class: "row", style: "gap:8px" }, download, coverLink,
      ...(c.srt || []).map(f => h("a", { class: "btn ghost", href: media(f), download: f, title: "Caption file" },
        f.split(".").length > 2 ? `Captions (${f.split(".").slice(-2, -1)[0]})` : "Captions")),
      h("button", { class: "btn ghost", onclick: () => openCaptionEditor(v, c, editor) }, "Fix captions"),
      h("button", { class: "btn", onclick: () => openPublish(v, c, publishBox) }, "Publish")),
    posted.length ? h("div", { class: "posted-list" }, posted.map(([name, r]) =>
      r.url ? h("a", { class: "posted", href: r.url, target: "_blank", rel: "noopener" }, `Posted to ${PLATFORM_NAMES[name]}`)
            : h("span", { class: "posted" }, `Posted to ${PLATFORM_NAMES[name]}`))) : null,
    publishBox,
    editor,
    platforms.length ? h("div", { class: "post" }, tabs, text,
      h("button", { class: "btn ghost small", style: "margin-top:6px", onclick: async () => {
        try { await navigator.clipboard.writeText(text.textContent); toast("Copied"); } catch { toast("Couldn't copy"); }
      } }, "Copy text")) : null);
  return card;
}

async function openCaptionEditor(v, c, box) {
  if (!box.hidden) { box.hidden = true; return; }
  box.hidden = false;
  box.replaceChildren(h("div", { class: "small muted" }, "Loading…"));
  let data;
  try { data = await api.get(`/api/videos/${v.id}/captions/${c.rank}`); }
  catch (e) { box.replaceChildren(h("div", { class: "error" }, e.message)); return; }
  const inputs = data.lines.map(l => h("textarea", { rows: 2, "aria-label": `Caption at ${fmt(l.start)}` }, l.text));
  const send = async body => {
    try {
      await api.post(`/api/videos/${v.id}/captions/${c.rank}`, { ...body, options: options() });
      box.hidden = true;
      toast("Saved. Updating the clips…");
      pollJobs();
    } catch (e) { toast(e.message); }
  };
  box.replaceChildren(
    h("div", { class: "small muted" }, "Correct any misheard words, then save. The clips are rendered again with your text."),
    ...data.lines.map((l, i) => h("label", { class: "caption-line" }, h("span", { class: "t" }, fmt(l.start)), inputs[i])),
    h("div", { class: "row", style: "gap:8px; margin-top:8px" },
      h("button", { class: "btn primary", onclick: () => send({ lines: data.lines.map((l, i) => ({ ...l, text: inputs[i].value })) }) }, "Save and update"),
      data.edited ? h("button", { class: "btn ghost", onclick: () => send({ reset: true }) }, "Undo my corrections") : null,
      h("button", { class: "btn ghost", onclick: () => { box.hidden = true; } }, "Cancel")));
}

// ---------------------------------------------------------------------------------------------
// Publishing

const PLATFORM_NAMES = { youtube: "YouTube", tiktok: "TikTok", instagram: "Instagram" };
const KEY_LABELS = { client_id: "Client ID", client_secret: "Client secret", client_key: "Client key",
  app_id: "App ID", app_secret: "App secret", config_id: "Configuration ID (optional)" };
const TIKTOK_PRIVACY = { PUBLIC_TO_EVERYONE: "Everyone", MUTUAL_FOLLOW_FRIENDS: "Friends",
  FOLLOWER_OF_CREATOR: "Followers", SELF_ONLY: "Only me" };
const GUIDE_URL = "https://github.com/BeastNoah/Clipper/blob/claude/clipper-pipeline/docs/publishing.md";

async function showAccounts() {
  const dialog = $("#accounts-dialog"), body = $("#accounts-body");
  if (!dialog.open) dialog.showModal();
  let list;
  try { list = await api.get("/api/accounts"); }
  catch (e) { body.replaceChildren(h("div", { class: "error" }, e.message)); return; }
  body.replaceChildren(
    h("p", { class: "small muted" }, "Clips are posted with each platform's official API, through a free developer app you create once. ",
      h("a", { href: GUIDE_URL, target: "_blank", rel: "noopener" }, "How to set them up")),
    ...list.map(accountRow));
}

function accountRow(p) {
  const fields = p.name === "instagram" ? [...p.app_fields, "config_id"] : p.app_fields;
  const keys = h("form", { class: "account-keys", hidden: p.configured, onsubmit: async e => {
    e.preventDefault();
    try { await api.post(`/api/accounts/${p.name}/app`, Object.fromEntries(new FormData(keys))); toast("Saved"); showAccounts(); }
    catch (err) { toast(err.message); }
  } },
    ...fields.map(f => h("label", {}, KEY_LABELS[f] || f,
      h("input", { name: f, type: f.endsWith("secret") ? "password" : "text", autocomplete: "off", spellcheck: "false", required: f !== "config_id" }))),
    h("div", { class: "row", style: "gap:8px" },
      h("button", { class: "btn primary small", type: "submit" }, "Save keys"),
      h("a", { class: "btn ghost small", href: p.setup_url, target: "_blank", rel: "noopener" }, "Create the app")));
  const state = p.connected ? `Connected as ${p.account}` : p.configured ? "Not connected" : "Not set up yet";
  return h("div", { class: "account" },
    h("div", { class: "row", style: "justify-content:space-between; gap:8px; flex-wrap:nowrap" },
      h("div", {}, h("b", {}, p.label), h("div", { class: "small muted" }, state)),
      h("div", { class: "row", style: "gap:6px" },
        p.configured ? h("button", { class: "btn ghost small", onclick: () => { keys.hidden = !keys.hidden; } }, "Keys") : null,
        p.configured && !p.connected ? h("button", { class: "btn primary small", onclick: () => connectAccount(p) }, "Connect") : null,
        p.connected ? h("button", { class: "btn ghost small", onclick: () => disconnectAccount(p) }, "Disconnect") : null)),
    keys);
}

async function connectAccount(p) {
  try {
    const { url } = await api.post(`/api/accounts/${p.name}/connect`, {});
    window.open(url, "_blank", "noopener");
    toast("Allow access in the new tab, then come back here");
  } catch (e) { toast(e.message); }
}

async function disconnectAccount(p) {
  if (!confirm(`Disconnect ${p.label} (${p.account})? You can connect it again any time.`)) return;
  try { await api.post(`/api/accounts/${p.name}/disconnect`, {}); showAccounts(); } catch (e) { toast(e.message); }
}

async function openPublish(v, c, box, platform) {
  if (!box.hidden && !platform) { box.hidden = true; return; }
  box.hidden = false;
  box.replaceChildren(h("div", { class: "small muted" }, "Loading…"));
  let data;
  try { data = await api.get(`/api/videos/${v.id}/publish/${c.rank}`); }
  catch (e) { box.replaceChildren(h("div", { class: "error" }, e.message)); return; }
  platform = platform || (data.accounts.find(a => a.connected) || data.accounts[0]).name;
  const acct = data.accounts.find(a => a.name === platform);
  const area = h("div", { class: "publish-form" });
  box.replaceChildren(
    h("div", { class: "post-tabs" }, data.accounts.map(a =>
      h("button", { class: a.name === platform ? "active" : null, onclick: () => openPublish(v, c, box, a.name) }, a.label))),
    area);
  if (!acct.connected) {
    area.append(h("p", { class: "small muted" }, acct.configured ? `Connect your ${acct.label} account first.` : `${acct.label} isn't set up yet.`),
      h("button", { class: "btn small", onclick: showAccounts }, "Open Accounts"));
    return;
  }
  const done = data.published[platform];
  const build = again => publishForm(v, c, box, area, acct, data.defaults[platform] || {}, again);
  if (done && done.status !== "failed") {
    area.append(
      h("p", { class: "small" }, `Already posted to ${acct.label}`, done.url ? [": ", h("a", { href: done.url, target: "_blank", rel: "noopener" }, done.url)] : ".",
        done.note ? h("span", { class: "muted" }, ` ${done.note}`) : null),
      h("button", { class: "btn ghost small", onclick: () => { area.replaceChildren(); build(true); } }, "Post it again"));
    return;
  }
  build(false);
}

async function publishForm(v, c, box, area, acct, defaults, again) {
  area.append(h("div", { class: "small muted" }, "Loading your account…"));
  let opts;
  try { opts = await api.get(`/api/accounts/${acct.name}/options`); }
  catch (e) { area.replaceChildren(h("div", { class: "error" }, e.message), h("button", { class: "btn small", onclick: showAccounts }, "Open Accounts")); return; }
  const duration = c.final_duration || c.duration;
  const button = h("button", { class: "btn primary", type: "submit" }, `Post to ${acct.label}`);
  const field = (label, input, hint) => h("label", { class: "field" }, h("span", {}, label), input, hint ? h("span", { class: "small muted" }, hint) : null);
  const check = (label, attrs = {}) => { const box = h("input", { type: "checkbox", ...attrs }); return [box, h("label", { class: "check" }, box, label)]; };
  let collect, valid = () => true;
  const parts = [h("div", { class: "small muted" }, `Posting to ${opts.account || acct.account}`)];

  if (acct.name === "youtube") {
    const title = h("input", { maxlength: 100, value: defaults.title || c.title });
    const desc = h("textarea", { rows: 4 }, defaults.description || "");
    const privacy = h("select", {}, opts.privacy_options.map(p => h("option", { value: p }, p[0].toUpperCase() + p.slice(1))));
    const [kids, kidsLabel] = check("Made for kids");
    parts.push(field("Title", title), field("Description", desc), field("Who can see it", privacy), kidsLabel);
    collect = () => ({ title: title.value, description: desc.value, privacy: privacy.value, made_for_kids: kids.checked });
  } else if (acct.name === "instagram") {
    const caption = h("textarea", { rows: 5 }, defaults.caption || "");
    const [feed, feedLabel] = check("Also show it in the profile grid", { checked: true });
    parts.push(field("Caption", caption), feedLabel);
    collect = () => ({ caption: caption.value, share_to_feed: feed.checked });
  } else {
    // TikTok: follows TikTok's content-sharing guidelines (no defaults for privacy or interactions).
    const title = h("textarea", { rows: 4, maxlength: 2200 }, defaults.title || "");
    const privacy = h("select", {}, h("option", { value: "", disabled: true, selected: true }, "Choose…"),
      opts.privacy_options.map(p => h("option", { value: p }, TIKTOK_PRIVACY[p] || p)));
    const [comment, commentLabel] = check(opts.comment_disabled ? "Allow comments (off in your TikTok settings)" : "Allow comments", { disabled: opts.comment_disabled });
    const [duet, duetLabel] = check(opts.duet_disabled ? "Allow Duet (off in your TikTok settings)" : "Allow Duet", { disabled: opts.duet_disabled });
    const [stitch, stitchLabel] = check(opts.stitch_disabled ? "Allow Stitch (off in your TikTok settings)" : "Allow Stitch", { disabled: opts.stitch_disabled });
    const [commercial, commercialLabel] = check("This post promotes a brand, product or service");
    const [yours, yoursLabel] = check("Your brand: you're promoting yourself or your own business");
    const [branded, brandedLabel] = check("Branded content: you're promoting another brand or a third party");
    const label = h("div", { class: "small muted" });
    const consent = h("div", { class: "small" });
    const brandBox = h("div", { class: "indent", hidden: true }, yoursLabel, brandedLabel, label);
    const tooLong = opts.max_duration && duration > opts.max_duration;
    const update = () => {
      brandBox.hidden = !commercial.checked;
      const isBranded = commercial.checked && branded.checked;
      const selfOnly = [...privacy.options].find(o => o.value === "SELF_ONLY");
      if (selfOnly) {
        selfOnly.disabled = isBranded;
        selfOnly.title = isBranded ? "Branded content can't be private" : "";
        if (isBranded && privacy.value === "SELF_ONLY") privacy.value = "";
      }
      label.textContent = !commercial.checked ? "" : isBranded ? "Your video will be labeled “Paid partnership”."
        : yours.checked ? "Your video will be labeled “Promotional content”." : "Say whether you're promoting yourself, a third party, or both.";
      const link = (text, href) => h("a", { href, target: "_blank", rel: "noopener" }, text);
      consent.replaceChildren("By posting, you agree to TikTok's ",
        ...(isBranded ? [link("Branded Content Policy", "https://www.tiktok.com/legal/page/global/bc-policy/en"), " and "] : []),
        link("Music Usage Confirmation", "https://www.tiktok.com/legal/page/global/music-usage-confirmation/en"), ".");
      button.disabled = !valid();
      button.title = commercial.checked && !(yours.checked || branded.checked)
        ? "Say whether the post promotes yourself, a third party, or both" : "";
    };
    valid = () => !tooLong && privacy.value && (!commercial.checked || yours.checked || branded.checked);
    [privacy, commercial, yours, branded].forEach(el => el.addEventListener("change", update));
    parts.push(
      field("Caption", title),
      field("Who can see it", privacy, opts.privacy_options.length === 1 && opts.privacy_options[0] === "SELF_ONLY"
        ? "Until TikTok audits your app, posts are visible only to you, and your account must be set to private." : null),
      h("div", { class: "checks" }, commentLabel, duetLabel, stitchLabel),
      commercialLabel, brandBox,
      tooLong ? h("div", { class: "error" }, `This clip is ${Math.round(duration)}s; your account can post up to ${opts.max_duration}s.`) : null,
      consent,
      h("div", { class: "small muted" }, "After you post, TikTok can take a few minutes to process it before it shows on your profile."));
    collect = () => ({ title: title.value, privacy: privacy.value, allow_comment: comment.checked, allow_duet: duet.checked,
      allow_stitch: stitch.checked, commercial: commercial.checked, your_brand: yours.checked,
      branded_content: branded.checked, consent: true });
    setTimeout(update, 0);
  }

  const form = h("form", { onsubmit: async e => {
    e.preventDefault();
    if (!valid()) return;
    button.disabled = true;
    try {
      await api.post(`/api/videos/${v.id}/publish/${c.rank}`, { platform: acct.name, form: collect(), again });
      box.hidden = true;
      toast(`Posting to ${acct.label}… you can keep working`);
      pollJobs();
    } catch (err) { toast(err.message); button.disabled = false; }
  } }, ...parts, h("div", { class: "row", style: "gap:8px; margin-top:8px" }, button,
    h("button", { class: "btn ghost", type: "button", onclick: () => { box.hidden = true; } }, "Cancel")));
  area.replaceChildren(form);
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

async function showPhone() {
  const body = $("#phone-body");
  const info = await api.get("/api/phone").catch(() => ({}));
  if (info.url) {
    const qr = h("div", { class: "qr" });
    if (info.svg) qr.innerHTML = info.svg;  // generated by the server from our own URL
    body.replaceChildren(
      h("p", { class: "muted" }, "Scan with the phone's camera (same Wi-Fi as this computer), or open this link:"),
      info.svg ? qr : null,
      h("div", { class: "link" }, info.url),
      h("p", { class: "small muted" }, "Anyone with this link can use clipper on this computer. Tip: add it to the home screen."));
  } else {
    body.replaceChildren(
      h("p", {}, "Right now clipper only answers this computer. To use it from your phone, stop it and start it again with:"),
      h("p", {}, h("code", {}, "clipper ui --phone")),
      h("p", { class: "small muted" }, "It then shows a QR code to scan. The phone needs to be on the same Wi-Fi."));
  }
  $("#phone-dialog").showModal();
}

(async function init() {
  $("#phone-btn").addEventListener("click", showPhone);
  $("#accounts-btn").addEventListener("click", showAccounts);
  window.addEventListener("focus", () => { if ($("#accounts-dialog").open) showAccounts(); });  // back from signing in
  S.choices = await api.get("/api/choices");
  window.addEventListener("hashchange", route);
  await route();
  pollJobs();
})();
