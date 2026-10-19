"use strict";

// ---------------------------------------------------------------------------------------------
// The clip editor: Claude's proposed edit is the starting point; everything here changes it.
//
// Times: a clip plays its parts (earlier parts first, then the clip itself) in a row. "Clip time" counts
// seconds along that row before any jump cuts (so the timeline never moves under the mouse); each
// moment maps to a part and a time on the video's own timeline ("video time"), which is what the
// server stores. "Output time" is where a moment ends up in the finished clip, after the cuts.

const E = {
  id: null, rank: null, data: null, edits: null, parts: [], total: 0,
  t: 0, playing: false, tool: null, sel: null, pps: 12, timelineHidden: false, muted: false,
  fixing: false, lines: null, shape: null, job: null, raf: null, poll: null,
};

const LAYOUT_NAMES = { auto: "Auto", follow: "Fill", main: "Fill (main person)", split: "Split", fit: "Fit" };
const MODE_LABEL = { fit: "Fit", track: "Fill", follow: "Fill", split: "Split", "follow+split": "Fill + Split" };
const TOOLS = [
  ["ai", "AI enhance", "M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8zM5 17l.9 2.1L8 20l-2.1.9L5 23l-.9-2.1L2 20l2.1-.9zM19 15l.7 1.6 1.6.7-1.6.7L19 19.6l-.7-1.6-1.6-.7 1.6-.7z"],
  ["caption", "Caption", "M4 5h16v14H4zM7 9h4M7 12h10M7 15h7"],
  ["text", "Text", "M5 5h14M12 5v14M9 19h6"],
  ["upload", "Upload", "M12 16V5M7 9l5-5 5 5M4 17v3h16v-3"],
  ["transitions", "Transitions", "M3 6h7l4 12h7M3 18h7l4-12h7"],
  ["hook", "AI hook", "M12 3v6M12 9a4 4 0 1 0 4 4M9 13l-3 3 3 3"],
  ["broll", "B-Roll", "M4 5h11v9H4zM9 10h11v9H9z"],
  ["music", "Music", "M9 18V6l10-2v12M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0zM19 16a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"],
];

function icon(path, size = 20) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24"); svg.setAttribute("width", size); svg.setAttribute("height", size);
  svg.setAttribute("fill", "none"); svg.setAttribute("stroke", "currentColor"); svg.setAttribute("stroke-width", "1.7");
  svg.setAttribute("stroke-linecap", "round"); svg.setAttribute("stroke-linejoin", "round"); svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", path);
  svg.append(p);
  return svg;
}

function fmtClock(t) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

function clone(x) { return JSON.parse(JSON.stringify(x)); }
function put(el, ...kids) { el.replaceChildren(...kids.flat().filter(k => k !== null && k !== undefined && k !== false)); }
function normTok(w) { return w.toLowerCase().replace(/[^\p{L}\p{N}']/gu, ""); }

// ---------------------------------------------------------------------------------------------
// Opening and loading

async function openEditor(id, rank) {
  document.body.classList.add("editing-clip");
  E.id = id; E.rank = Number(rank); E.data = null; E.sel = null; E.tool = E.tool || "ai"; E.fixing = false; E.shape = null;
  $("#app").replaceChildren(h("div", { class: "ed-loading" }, "Opening the editor…"));
  try {
    E.data = await api.get(`/api/videos/${id}/edit/${rank}`);
  } catch (e) {
    $("#app").replaceChildren(h("div", { class: "panel empty" }, `Couldn't open this clip: ${e.message} `,
      h("a", { href: `#/video/${id}/clips` }, "Back to the clips")));
    return;
  }
  E.edits = clone(E.data.edits);
  E.lines = null;
  buildParts();
  E.t = 0;
  renderEditor();
  clearTimeout(E.poll);
  pollEditorJobs();
}

function closeEditor() {
  document.body.classList.remove("editing-clip");
  cancelAnimationFrame(E.raf);
  clearTimeout(E.poll);
  E.data = null;
}

// The clip's parts laid along the clip time axis.
function buildParts() {
  let t0 = 0;
  E.parts = E.data.parts.map(p => { const part = { ...p, t0, len: p.end - p.start }; t0 += part.len; return part; });
  E.total = t0;
  E.pps = Math.max(4, Math.min(60, ($("#ed-timeline")?.clientWidth || 1000) / Math.max(1, E.total) * 0.96));
}

function locate(ct) {
  ct = Math.min(Math.max(0, ct), E.total);
  let part = E.parts[E.parts.length - 1];
  for (const p of E.parts) if (ct < p.t0 + p.len) { part = p; break; }
  return { part, src: part.start + Math.min(part.len, ct - part.t0) };
}
function clipTime(src) {
  const p = E.parts.find(p => src >= p.start && src <= p.end) || E.parts[0];
  return p.t0 + (src - p.start);
}
function toClip(range) { return [clipTime(range[0]), clipTime(range[1])]; }

// Every stretch that doesn't play: clipper's jump cuts (unless you kept them) and your own cuts, in clip time.
function cutRanges() {
  const out = [];
  const kept = E.edits.cuts_kept || [];
  for (const p of E.parts) for (const [a, b] of p.removed) {
    if (!kept.some(([x, y]) => x <= a + 0.01 && y >= b - 0.01)) out.push({ a: clipTime(a), b: clipTime(b), src: [a, b], kind: "auto" });
  }
  for (const [a, b] of E.edits.cuts_removed || []) out.push({ a: clipTime(a), b: clipTime(b), src: [a, b], kind: "mine" });
  return out.sort((x, y) => x.a - y.a);
}
function outputTime(ct) {
  let cut = 0;
  for (const c of cutRanges()) cut += Math.max(0, Math.min(c.b, ct) - c.a);
  return ct - cut;
}
function inCut(ct) { return cutRanges().find(c => ct >= c.a && ct < c.b); }

// Framing segments along the clip: clipper's shots, split where you chose a framing yourself.
function segments() {
  const edges = new Set([0, E.total]);
  for (const p of E.parts) {
    edges.add(p.t0);
    for (const s of p.plan?.shots || []) edges.add(clipTime(s.start));
  }
  for (const l of E.edits.layouts || []) { edges.add(clipTime(l.start)); edges.add(clipTime(l.end)); }
  const xs = [...edges].filter(x => x >= 0 && x <= E.total).sort((a, b) => a - b);
  const segs = [];
  for (let i = 0; i + 1 < xs.length; i++) {
    if (xs[i + 1] - xs[i] < 0.02) continue;
    const mid = (xs[i] + xs[i + 1]) / 2;
    segs.push({ a: xs[i], b: xs[i + 1], ...framingAt(mid) });
  }
  return segs;
}
function framingAt(ct) {
  const { part, src } = locate(ct);
  const own = (E.edits.layouts || []).find(l => src >= l.start && src < l.end && l.layout !== "auto");
  const shot = (part.plan?.shots || []).find(s => src >= s.start && src < s.end) || null;
  const mode = shot?.mode || null;
  const layout = own ? own.layout : (E.edits.layout && E.edits.layout !== "auto" ? E.edits.layout : null);
  let label = own ? LAYOUT_NAMES[own.layout] : mode ? MODE_LABEL[mode] : E.data.layout === "auto" ? "Auto" : LAYOUT_NAMES[E.data.layout];
  return { part, src, shot, mode, layout, own: !!own, label };
}

function keywordHits(words) {
  const toks = words.map(w => normTok(w.w));
  const hits = new Set();
  for (const kw of E.edits.keywords ?? E.data.clip.keywords ?? []) {
    const parts = kw.split(/\s+/).map(normTok).filter(Boolean);
    if (!parts.length) continue;
    for (let i = 0; i + parts.length <= toks.length; i++) {
      if (parts.every((p, k) => toks[i + k] === p)) for (let k = 0; k < parts.length; k++) hits.add(i + k);
    }
  }
  return hits;
}

// Caption lines the way the render groups them: never across a sentence end or a pause, 2-4 words.
function captionLines(part) {
  const phrases = [], SENT = /[.?!…。？！]["'”’)»]*$/;
  let cur = [];
  for (const w of part.words) {
    if (cur.length && w.s - cur[cur.length - 1].e > 0.6) { phrases.push(cur); cur = []; }
    cur.push(w);
    if (SENT.test(w.w)) { phrases.push(cur); cur = []; }
  }
  if (cur.length) phrases.push(cur);
  const lines = [];
  for (const ph of phrases) {
    const n = Math.ceil(ph.length / 4), size = Math.ceil(ph.length / n);
    for (let i = 0; i < ph.length; i += size) lines.push(ph.slice(i, i + size));
  }
  return lines;
}

function dirty() {
  return JSON.stringify(E.edits) !== JSON.stringify(E.data.edits) || !!E.shape || (E.lines && E.lines.some(l => l.text !== l.orig));
}

// ---------------------------------------------------------------------------------------------
// Layout

function renderEditor() {
  const v = E.data;
  const root = h("div", { class: "ed" },
    h("header", { class: "ed-head" },
      h("a", { class: "btn ghost small", href: `#/video/${E.id}/clips`, onclick: e => { if (dirty() && !confirm("Leave without applying your changes?")) e.preventDefault(); } }, "← Clips"),
      h("div", { class: "ed-title" }, h("strong", {}, `#${v.clip.rank}`), " ",
        h("span", { id: "ed-title-text" }, E.edits.title || v.clip.title),
        v.clip.edited || !isDefault(E.edits) ? h("span", { class: "chip", id: "ed-edited-chip" }, "edited by you") : null),
      h("div", { class: "ed-job", id: "ed-job" }),
      h("div", { class: "row", style: "gap:8px" },
        h("button", { class: "btn ghost small", id: "ed-discard", onclick: discardChanges }, "Discard"),
        h("button", { class: "btn primary", id: "ed-apply", onclick: applyChanges }, "Apply changes"))),
    h("div", { class: "ed-body" },
      h("aside", { class: "ed-script panel", id: "ed-script" }),
      h("section", { class: "ed-center" },
        h("div", { class: "ed-stagebar", id: "ed-stagebar" }),
        h("div", { class: "ed-stagewrap" }, h("div", { class: "ed-stage", id: "ed-stage" })),
      ),
      h("aside", { class: "ed-side" },
        h("div", { class: "ed-panel panel", id: "ed-panel", hidden: !E.tool }),
        h("nav", { class: "ed-rail", id: "ed-rail" }))),
    h("section", { class: "ed-timeline panel", id: "ed-timeline" }));
  $("#app").replaceChildren(root);
  renderRail(); renderPanel(); renderScript(); renderStageBar(); renderStage(); renderTimeline(); renderJobPill();
  document.onkeydown = editorKeys;
}

function isDefault(ed) {
  const empty = { title: null, title_overlay: null, keywords: null, jump_cuts: null, layout: null, cuts_removed: [], cuts_kept: [],
    layouts: [], zoom: "auto", zooms: [], texts: [], broll: [], music: null, transition: "cut", fade_in: false, fade_out: false };
  return JSON.stringify({ ...empty, ...ed }) === JSON.stringify(empty);
}

function changed() {
  renderStageBar(); renderTimeline(); updateStage(); renderScript();
  const chip = $("#ed-edited-chip");
  if (chip) chip.hidden = isDefault(E.edits) && !E.data.clip.edited;
  $("#ed-title-text").textContent = E.edits.title || E.data.clip.title;
}

function editorKeys(e) {
  if (!E.data || e.target.matches("input, textarea, select, [contenteditable]")) return;
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); deleteSelection(); }
  else if (e.key === "s" || e.key === "S") { e.preventDefault(); splitAtPlayhead(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); seek(E.t - (e.shiftKey ? 1 : 0.1)); }
  else if (e.key === "ArrowRight") { e.preventDefault(); seek(E.t + (e.shiftKey ? 1 : 0.1)); }
  else if (e.key === "Escape") { E.sel = null; renderTimeline(); renderStageBar(); }
}

// ---------------------------------------------------------------------------------------------
// Script (left): the clip's sections with their words; keywords in green, cuts as dots

function renderScript() {
  const box = $("#ed-script");
  if (!box) return;
  const clip = E.data.clip;
  const kids = [h("button", { class: "btn ghost small ed-add-section", onclick: addSectionDialog }, "+ Add a section")];
  E.parts.forEach((part, pi) => {
    const hits = keywordHits(part.words);
    const cuts = cutRanges().filter(c => c.src[0] >= part.start - 0.01 && c.src[1] <= part.end + 0.01);
    const isMain = pi === E.parts.length - 1;
    const head = h("div", { class: "ed-section-head" },
      h("span", { class: "small muted" }, isMain ? (E.parts.length > 1 ? "The clip" : "Section") : `Played first · section ${pi + 1}`,
        ` · ${fmt(part.start)}–${fmt(part.end)}`),
      !isMain ? h("button", { class: "btn ghost small danger", title: "Leave this part out", onclick: () => {
        const before = currentShape().before.filter(([a, b]) => !(a === part.si && b === part.ei));
        E.shape = { ...currentShape(), before };
        toast("Section removed; apply to re-render");
        renderScript();
      } }, "×") : null);
    let body;
    if (E.fixing) {
      body = h("div", { class: "ed-fix" }, linesFor(part).map(line => h("textarea", { rows: 1, class: line.text !== line.orig ? "changed" : "",
        oninput: e => { line.text = e.target.value; e.target.classList.toggle("changed", line.text !== line.orig); },
        onfocus: () => seek(clipTime(line.start)) }, line.text)));
    } else {
      const spans = [];
      part.words.forEach((w, i) => {
        const prev = part.words[i - 1];
        const cut = prev && cuts.find(c => c.src[0] >= prev.s - 0.05 && c.src[1] <= w.s + 0.05 && c.src[1] - c.src[0] > 0.1);
        if (cut) spans.push(h("button", { class: "ed-dot" + (cut.kind === "mine" ? " mine" : ""),
          title: cut.kind === "mine" ? "You cut this out · click to select" : "A pause clipper cuts out · click to select",
          onclick: () => { E.sel = { kind: "cut", a: cut.a, b: cut.b, src: cut.src, cutKind: cut.kind }; seek(cut.a); renderTimeline(); renderStageBar(); } }, "•"));
        else if (i) spans.push(" ");
        const inCutW = cutRanges().some(c => (w.s + w.e) / 2 >= c.src[0] && (w.s + w.e) / 2 < c.src[1]);
        spans.push(h("span", { class: "ed-word" + (hits.has(i) ? " key" : "") + (inCutW ? " cut" : ""), "data-s": w.s, "data-e": w.e,
          title: hits.has(i) ? "Highlighted keyword · click to remove" : "Click to highlight (and punch in on) this word",
          onclick: () => toggleKeyword(part, i, hits) }, w.w));
      });
      body = h("p", { class: "ed-text" }, spans);
    }
    kids.push(h("div", { class: "ed-section" + (isMain ? " main" : ""), "data-part": pi }, head, body));
  });
  if (E.fixing) kids.push(h("div", { class: "small muted" }, "Fix misheard words here; the time of each line is spread over your words."));
  else kids.push(h("div", { class: "small muted ed-hint" }, "Click a word to highlight it in the captions (Claude's picks are green). A • marks a pause clipper cuts out."));
  put(box, ...kids);
  highlightScript();
}

function linesFor(part) {
  if (!E.lines) E.lines = {};
  if (!E.lines[part.dir]) {
    E.lines[part.dir] = captionLines(part).map(ws => ({ start: ws[0].s, end: ws[ws.length - 1].e, text: ws.map(w => w.w).join(" "), orig: ws.map(w => w.w).join(" ") }));
  }
  return E.lines[part.dir];
}

function toggleKeyword(part, i, hits) {
  const list = [...(E.edits.keywords ?? E.data.clip.keywords ?? [])];
  const tok = normTok(part.words[i].w);
  if (hits.has(i)) {
    const left = list.filter(kw => !kw.split(/\s+/).map(normTok).includes(tok));
    E.edits.keywords = left;
  } else if (tok) {
    E.edits.keywords = [...list, part.words[i].w.replace(/[^\p{L}\p{N}' -]/gu, "")];
  }
  if (E.edits.keywords && JSON.stringify(E.edits.keywords) === JSON.stringify(E.data.clip.keywords || [])) E.edits.keywords = null;
  changed();
}

function highlightScript() {
  const { src } = locate(E.t);
  for (const el of document.querySelectorAll(".ed-word")) {
    el.classList.toggle("now", Number(el.dataset.s) <= src && Number(el.dataset.e) > src);
  }
}

function currentShape() {
  if (E.shape) return E.shape;
  const main = E.parts[E.parts.length - 1];
  return { si: main.si, ei: main.ei, before: E.parts.slice(0, -1).map(p => [p.si, p.ei]) };
}

function addSectionDialog() {
  const shape = currentShape();
  const dialog = $("#section-dialog"), body = $("#section-body");
  let first = null;
  const covered = i => (i >= shape.si && i <= shape.ei) || shape.before.some(([a, b]) => i >= a && i <= b);
  const list = h("div", { class: "transcript ed-pick" });
  const status = h("div", { class: "small muted", style: "margin:8px 0" }, "Click the first sentence of the part to add, then its last.");
  const add = h("button", { class: "btn primary", disabled: true }, "Add");
  let range = null;
  const draw = () => list.replaceChildren(...E.data.sentences.map(s => h("button", {
    class: "sent" + (covered(s.i) ? " in" : "") + (range && s.i >= range[0] && s.i <= range[1] ? " mark" : "") + (first === s.i ? " mark" : ""),
    disabled: covered(s.i), onclick: () => {
      if (first == null || s.i < first) { first = s.i; range = null; status.textContent = `From ${fmt(s.start)} · now click the last sentence`; }
      else { range = [first, s.i]; status.textContent = `${fmt(E.data.sentences[first].start)}–${fmt(s.end)} · ${Math.round(s.end - E.data.sentences[first].start)}s`; }
      add.disabled = !range; draw();
    } }, h("span", { class: "t" }, fmt(s.start)), s.text)));
  draw();
  add.onclick = () => {
    const [a, b] = range;
    let next = { ...shape, before: shape.before.map(x => [...x]) };
    if (b < shape.si) next.before = [...next.before, [a, b]].sort((x, y) => x[0] - y[0]);
    else if (a > shape.ei) next.ei = b;
    else if (a < shape.si) next.si = a;
    else next.ei = Math.max(next.ei, b);
    E.shape = next;
    dialog.close();
    toast(b < shape.si ? "Added as a part played first; apply to re-render" : "Clip extended; apply to re-render");
    renderScript();
  };
  body.replaceChildren(status, list, h("div", { class: "row", style: "justify-content:flex-end; gap:8px; margin-top:10px" }, add));
  dialog.showModal();
}

// ---------------------------------------------------------------------------------------------
// Stage: the vertical preview with the framing, captions and overlays as they will be rendered

function renderStageBar() {
  const bar = $("#ed-stagebar");
  if (!bar) return;
  const target = E.sel?.kind === "segment" ? E.sel : null;
  const f = framingAt(target ? (target.a + target.b) / 2 : E.t);
  const layoutSel = h("select", { class: "ed-pill-select", "aria-label": "Framing of this segment", onchange: e => setLayout(e.target.value, target) },
    S.choices.shot_layouts.map(l => h("option", { value: l, selected: (f.own ? f.layout : "auto") === l }, LAYOUT_NAMES[l])));
  const tracking = (E.edits.layout || E.data.layout) !== "fit";
  put(bar,
    h("span", { class: "ed-pill" }, icon("M8 3h8l-2 3H10zM6 6h12v14H6z", 14), E.data.format),
    h("label", { class: "ed-pill" }, icon("M4 4h16v16H4zM4 12h16", 14), "Layout:", layoutSel),
    h("button", { class: "ed-pill", title: "Follow the people in the picture (on) or show the whole frame (off)", onclick: () => {
      E.edits.layout = tracking ? "fit" : null; changed();
    } }, icon("M12 4v4M12 16v4M4 12h4M16 12h4M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0-6 0", 14), `Track: ${tracking ? "ON" : "OFF"}`),
    target ? h("span", { class: "small muted" }, `Segment ${fmt(target.a)}–${fmt(target.b)} selected`)
      : E.sel?.kind === "cut" ? h("button", { class: "btn small", onclick: () => restoreCut(E.sel) }, E.sel.cutKind === "mine" ? "Restore this stretch" : "Keep this pause") : null);
}

function setLayout(layout, seg) {
  if (!seg) seg = segments().find(s => E.t >= s.a && E.t < s.b) || segments()[0];
  const { part, src } = locate(seg.a);
  const end = Math.min(part.end, src + (seg.b - seg.a));
  const layouts = (E.edits.layouts || []).filter(l => l.end <= src + 0.001 || l.start >= end - 0.001);
  // an override that straddles the segment is split around it
  for (const l of E.edits.layouts || []) {
    if (l.end <= src + 0.001 || l.start >= end - 0.001) continue;
    if (l.start < src) layouts.push({ ...l, end: src });
    if (l.end > end) layouts.push({ ...l, start: end });
  }
  layouts.push({ start: +src.toFixed(3), end: +end.toFixed(3), layout });  // "auto" too: it keeps the segment's edges
  E.edits.layouts = layouts.sort((x, y) => x.start - y.start);
  E.sel = { kind: "segment", a: seg.a, b: seg.b };
  changed();
}

function renderStage() {
  const stage = $("#ed-stage");
  const ratio = E.data.format.split(":").map(Number);
  stage.style.aspectRatio = `${ratio[0]} / ${ratio[1]}`;
  const { part } = locate(E.t);
  stage.replaceChildren(
    h("div", { class: "ed-blur", id: "ed-blur" }),
    h("div", { class: "ed-frame", id: "ed-frame" }, part.media
      ? h("video", { id: "ed-video", playsinline: true, preload: "auto", muted: E.muted })
      : h("div", { class: "ed-nomedia" }, "The video for this part isn't on this computer yet. Render the clip once, then edit it.")),
    h("div", { class: "ed-frame half bottom", id: "ed-frame2", hidden: true }, h("video", { id: "ed-video2", playsinline: true, preload: "auto", muted: true })),
    h("div", { class: "ed-broll", id: "ed-broll", hidden: true }),
    h("div", { class: "ed-overlays", id: "ed-overlays" }),
    h("div", { class: "ed-caps", id: "ed-caps" }),
    h("button", { class: "ed-bigplay", id: "ed-bigplay", onclick: togglePlay, "aria-label": "Play" }, icon("M8 5v14l11-7z", 36)));
  const video = $("#ed-video");
  if (video) {
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("ended", () => advancePart());
    video.addEventListener("play", () => { E.playing = true; loop(); $("#ed-bigplay").hidden = true; });
    video.addEventListener("pause", () => { E.playing = false; $("#ed-bigplay").hidden = false; });
    video.addEventListener("click", togglePlay);
    loadPart(part, part.start, false);
  }
  updateStage();
}

function loadPart(part, src, play) {
  const video = $("#ed-video");
  if (!video || !part.media) return;
  E.loadedPart = part.dir;
  video.src = part.media.url;
  const v2 = $("#ed-video2");
  if (v2) v2.src = part.media.url;
  const seekTo = Math.max(0, src - part.media.offset);
  const onMeta = () => { video.currentTime = seekTo; if (play) video.play().catch(() => {}); };
  video.addEventListener("loadedmetadata", onMeta, { once: true });
}

function seek(ct, play) {
  ct = Math.min(Math.max(0, ct), E.total);
  E.t = ct;
  const { part, src } = locate(ct);
  const video = $("#ed-video");
  if (video && part.media) {
    if (E.loadedPart !== part.dir) loadPart(part, src, play ?? E.playing);
    else video.currentTime = Math.max(0, src - part.media.offset);
  }
  updateStage(); updatePlayhead();
}

function togglePlay() {
  const video = $("#ed-video");
  if (!video) return;
  if (video.paused) { if (E.t >= E.total - 0.05) seek(0); video.play().catch(() => {}); } else video.pause();
}

function onTimeUpdate() { syncFromVideo(); }

function syncFromVideo() {
  const video = $("#ed-video");
  const { part } = locate(E.t);
  if (!video || !part.media || E.loadedPart !== part.dir) return;
  const src = video.currentTime + part.media.offset;
  let ct = part.t0 + (src - part.start);
  if (src >= part.end - 0.03) { advancePart(); return; }
  const cut = inCut(ct);
  if (cut && E.playing) { seek(cut.b + 0.001, true); return; }
  E.t = Math.max(part.t0, Math.min(part.t0 + part.len, ct));
}

function advancePart() {
  const { part } = locate(E.t);
  const k = E.parts.indexOf(part);
  if (k + 1 < E.parts.length) seek(E.parts[k + 1].t0, true);
  else { $("#ed-video")?.pause(); E.t = E.total; updateStage(); updatePlayhead(); }
}

function loop() {
  cancelAnimationFrame(E.raf);
  const step = () => {
    if (!E.data) return;
    syncFromVideo(); updateStage(); updatePlayhead();
    if (E.playing) E.raf = requestAnimationFrame(step);
  };
  E.raf = requestAnimationFrame(step);
}

// Puts the video where the finished clip's window will be: the framing plan's x per frame, or your choice.
function updateStage() {
  const stage = $("#ed-stage"), frame = $("#ed-frame"), frame2 = $("#ed-frame2"), video = $("#ed-video"), v2 = $("#ed-video2");
  if (!stage || !video) return;
  const f = framingAt(E.t);
  const plan = f.part.plan;
  const SW = stage.clientWidth, SH = stage.clientHeight;
  const W = plan?.width || video.videoWidth || 16, H = plan?.height || video.videoHeight || 9;
  const i = plan ? Math.max(0, Math.min(plan.x.length - 1, Math.round((f.src - f.part.start) * plan.fps))) : 0;
  const x = plan ? plan.x[i] : null;
  let mode = f.layout || f.mode || "fit";
  if (mode === "track" || mode === "follow" || mode === "main" || mode === "follow+split") mode = Array.isArray(x) ? "split" : "fill";
  if (mode === "split" && !Array.isArray(x)) mode = "fill";
  if (mode === "auto") mode = x == null ? "fit" : Array.isArray(x) ? "split" : "fill";
  stage.dataset.mode = mode;
  frame.classList.toggle("half", mode === "split");
  frame2.hidden = mode !== "split";
  const zoom = zoomAt(outputTime(E.t)) || 1;
  if (mode === "fit") {
    const s = Math.min(SW / W, SH / H);
    video.style.cssText = `width:${W * s}px;height:${H * s}px;left:${(SW - W * s) / 2}px;top:${(SH - H * s) / 2}px;`;
    $("#ed-blur").style.backgroundImage = "";
  } else if (mode === "split") {
    const [xl, yl, xr, yr, hc] = x;
    const half = SH / 2, s = half / hc * zoom;
    const wc = hc * SW / half;
    video.style.cssText = `width:${W * s}px;height:${H * s}px;left:${-(xl + (wc - SW / s) / 2 * 0) * s}px;top:${-yl * s}px;`;
    v2.style.cssText = `width:${W * s}px;height:${H * s}px;left:${-xr * s}px;top:${-yr * s}px;`;
    if (Math.abs(v2.currentTime - video.currentTime) > 0.12) v2.currentTime = video.currentTime;
    if (E.playing && v2.paused) v2.play().catch(() => {}); else if (!E.playing && !v2.paused) v2.pause();
  } else {
    const cw = plan?.crop_w || Math.round(H * SW / SH);
    const s = SH / H * zoom;
    const cx = typeof x === "number" ? x : Math.max(0, (W - cw) / 2);
    const left = -(cx + cw / 2) * s + SW / 2;
    const top = -(H * (zoom > 1.001 ? 0.42 : 0.5)) * s + SH * (zoom > 1.001 ? 0.42 : 0.5);
    video.style.cssText = `width:${W * s}px;height:${H * s}px;left:${left}px;top:${top}px;`;
  }
  updateCaptions(f);
  updateOverlays();
  highlightScript();
}

function zoomAt(tOut) {
  if (E.edits.zoom === "off" && !(E.edits.zooms || []).length) return 1;
  const windows = [];
  const total = outputTime(E.total);
  if (E.edits.zoom !== "off") {
    windows.push([0, Math.min(1.4, total)]);
    for (const part of E.parts) {
      const hits = keywordHits(part.words);
      part.words.forEach((w, i) => { if (hits.has(i)) windows.push([outputTime(clipTime(w.s)) - 0.05, outputTime(clipTime(w.e)) + 1.0]); });
    }
  }
  for (const [a, b] of E.edits.zooms || []) windows.push([outputTime(clipTime(a)), outputTime(clipTime(b))]);
  const inWin = windows.some(([a, b]) => tOut >= a && tOut <= b);
  return inWin ? 1.12 : 1;
}

function updateCaptions(f) {
  const caps = $("#ed-caps");
  if (!caps) return;
  const opts = options();
  caps.className = `ed-caps style-${opts.caption_style || "bold"} pos-${opts.caption_position || "lower"}` + (opts.uppercase ? " upper" : "");
  const part = f.part, src = f.src;
  const hits = keywordHits(part.words);
  let line = null;
  for (const ws of captionLines(part)) if (src >= ws[0].s && src < ws[ws.length - 1].e + 0.3) { line = ws; break; }
  const title = E.edits.title || E.data.clip.title;
  const showTitle = (E.edits.title_overlay ?? (E.data.options.title_overlay ?? true)) && outputTime(E.t) < 2.6;
  put(caps,
    showTitle ? h("div", { class: "ed-cap-title" }, title) : null,
    line ? h("div", { class: "ed-cap-line" }, line.map(w => h("span", {
      class: (hits.has(part.words.indexOf(w)) ? "key" : "") + (src >= w.s && src < w.e ? " now" : "") }, w.w.replace(/[,.;:]+$/, "")))) : null);
}

function updateOverlays() {
  const box = $("#ed-overlays"), broll = $("#ed-broll");
  if (!box) return;
  const { src } = locate(E.t);
  box.replaceChildren(...(E.edits.texts || []).filter(t => src >= t.start && src < t.end).map(t =>
    h("div", { class: `ed-textcard pos-${t.position}` }, t.text)));
  const b = (E.edits.broll || []).find(b => src >= b.start && src < b.end);
  broll.hidden = !b;
  if (b) {
    const asset = E.data.assets.find(a => a.path === b.file);
    if (broll.dataset.file !== b.file) {
      broll.dataset.file = b.file;
      put(broll, !asset ? h("div", { class: "ed-nomedia" }, "B-roll file missing")
        : asset.kind === "image" ? h("img", { src: asset.url, alt: "" }) : h("video", { src: asset.url, muted: true, loop: true, autoplay: true, playsinline: true }));
    }
  } else broll.dataset.file = "";
}

// ---------------------------------------------------------------------------------------------
// Timeline

function renderTimeline() {
  const box = $("#ed-timeline");
  if (!box) return;
  const cuts = cutRanges();
  const w = E.total * E.pps;
  const px = t => `${(t * E.pps).toFixed(1)}px`;
  const tick = E.pps >= 40 ? 1 : E.pps >= 20 ? 2 : E.pps >= 9 ? 5 : E.pps >= 5 ? 10 : 15;
  const ruler = h("div", { class: "tl-ruler", style: `width:${w}px` });
  for (let t = 0; t <= E.total + 1e-6; t += tick) ruler.append(h("span", { class: "tl-tick", style: `left:${px(t)}` }, String(Math.round(t))));
  ruler.addEventListener("pointerdown", e => scrub(e));

  // top track: your text cards, b-roll and punch-ins
  const over = h("div", { class: "tl-track tl-over", style: `width:${w}px` },
    h("button", { class: "tl-add", onclick: () => { E.tool = "broll"; renderRail(); renderPanel(); } }, "+ B-Roll"));
  const blocks = [
    ...(E.edits.texts || []).map((t, i) => ({ kind: "text", i, a: clipTime(t.start), b: clipTime(t.end), label: t.text })),
    ...(E.edits.broll || []).map((b, i) => ({ kind: "broll", i, a: clipTime(b.start), b: clipTime(b.end), label: E.data.assets.find(a => a.path === b.file)?.name || "B-roll" })),
    ...(E.edits.zooms || []).map((z, i) => ({ kind: "zoom", i, a: clipTime(z[0]), b: clipTime(z[1]), label: "Punch in" })),
  ];
  for (const b of blocks) over.append(blockEl(b, px));
  over.addEventListener("pointerdown", e => { if (e.target === over) scrub(e); });

  // video track: frames, framing segments, cuts, part boundaries
  const video = h("div", { class: "tl-track tl-video", style: `width:${w}px` });
  const every = Math.max(1, Math.round(72 / E.pps));
  for (const part of E.parts) {
    for (let t = 0; t < part.len; t += every) {
      video.append(h("img", { class: "tl-thumb", alt: "", loading: "lazy", style: `left:${px(part.t0 + t)};width:${px(Math.min(every, part.len - t))}`,
        src: `/api/videos/${E.id}/frame?t=${(part.start + t + 0.5).toFixed(1)}&clip=${part.dir}` }));
    }
    if (part.t0 > 0) video.append(h("div", { class: "tl-part", style: `left:${px(part.t0)}` }));
  }
  for (const s of segments()) {
    const el = h("div", { class: "tl-seg" + (E.sel?.kind === "segment" && Math.abs(E.sel.a - s.a) < 1e-6 ? " sel" : "") + (s.own ? " own" : ""),
      style: `left:${px(s.a)};width:${px(s.b - s.a)}`, title: `${s.label}: click to select, then choose the layout above`,
      onpointerdown: e => { e.stopPropagation(); E.sel = { kind: "segment", a: s.a, b: s.b }; seek(Math.max(s.a, Math.min(s.b - 0.01, E.t >= s.a && E.t < s.b ? E.t : s.a))); renderTimeline(); renderStageBar(); } },
      h("span", { class: "tl-seglabel" }, s.label));
    video.append(el);
  }
  for (const c of cuts) {
    video.append(h("div", { class: "tl-cut " + c.kind + (E.sel?.kind === "cut" && Math.abs(E.sel.a - c.a) < 1e-6 ? " sel" : ""),
      style: `left:${px(c.a)};width:${px(c.b - c.a)}`, title: c.kind === "mine" ? "Cut out by you" : "Pause cut out by clipper · click, then Keep",
      onpointerdown: e => { e.stopPropagation(); E.sel = { kind: "cut", a: c.a, b: c.b, src: c.src, cutKind: c.kind }; renderTimeline(); renderStageBar(); } }));
  }
  video.addEventListener("pointerdown", e => scrub(e));

  // audio track: loudness
  const audio = h("div", { class: "tl-track tl-audio", style: `width:${w}px` });
  const canvas = h("canvas", { height: 36, width: Math.max(1, Math.round(w)) });
  audio.append(canvas);
  audio.addEventListener("pointerdown", e => scrub(e));
  for (const c of cuts) audio.append(h("div", { class: "tl-cut " + c.kind, style: `left:${px(c.a)};width:${px(c.b - c.a)}` }));

  const playhead = h("div", { class: "tl-playhead", id: "tl-playhead", style: `left:${px(E.t)}` }, h("i", {}));
  const tracks = h("div", { class: "tl-tracks", id: "tl-tracks" }, ruler, over, video, audio, playhead);
  const outTotal = outputTime(E.total);
  const bar = h("div", { class: "tl-bar" },
    h("div", { class: "row", style: "gap:4px" },
      h("button", { class: "btn ghost small", onclick: () => { E.timelineHidden = !E.timelineHidden; renderTimeline(); } },
        icon("M4 6h16v12H4zM8 10h8M8 14h5", 16), E.timelineHidden ? "Show timeline" : "Hide timeline"),
      h("button", { class: "btn ghost icon", title: "Split the segment at the playhead (S)", onclick: splitAtPlayhead, "aria-label": "Split" }, icon("M12 3v18M7 8l-3 4 3 4M17 8l3 4-3 4", 18)),
      h("button", { class: "btn ghost icon", title: "Cut out the selected segment, or remove the selected item (Delete)", onclick: deleteSelection, "aria-label": "Delete" },
        icon("M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13", 18)),
      h("button", { class: "btn ghost icon", title: E.muted ? "Unmute" : "Mute", onclick: () => { E.muted = !E.muted; const v = $("#ed-video"); if (v) v.muted = E.muted; renderTimeline(); }, "aria-label": "Mute" },
        icon(E.muted ? "M4 9v6h4l5 4V5L8 9zM19 9l-4 6M15 9l4 6" : "M4 9v6h4l5 4V5L8 9zM16 8a5 5 0 0 1 0 8M18.5 5.5a8.5 8.5 0 0 1 0 13", 18))),
    h("div", { class: "tl-transport" },
      h("button", { class: "btn ghost icon", onclick: () => seek(0), "aria-label": "Start" }, icon("M6 5v14M19 5l-11 7 11 7z", 18)),
      h("button", { class: "btn ghost icon tl-play", onclick: togglePlay, "aria-label": "Play or pause" }, icon(E.playing ? "M7 5h4v14H7zM13 5h4v14h-4z" : "M8 5v14l11-7z", 20)),
      h("button", { class: "btn ghost icon", onclick: () => seek(E.total), "aria-label": "End" }, icon("M18 5v14M5 5l11 7-11 7z", 18)),
      h("span", { class: "tl-time" }, h("b", { id: "tl-now" }, fmtClock(outputTime(E.t))), " / ", h("span", {}, fmtClock(outTotal)))),
    h("div", { class: "row", style: "gap:6px" },
      h("button", { class: "btn ghost icon", "aria-label": "Zoom out", onclick: () => { E.pps = Math.max(2, E.pps / 1.4); renderTimeline(); } }, icon("M10 4a6 6 0 1 0 0 12 6 6 0 0 0 0-12zM14.5 14.5L20 20M7 10h6", 16)),
      h("input", { type: "range", min: 2, max: 80, step: 1, value: E.pps, class: "tl-zoom", "aria-label": "Timeline zoom", oninput: e => { E.pps = Number(e.target.value); renderTimeline(); } }),
      h("button", { class: "btn ghost icon", "aria-label": "Zoom in", onclick: () => { E.pps = Math.min(80, E.pps * 1.4); renderTimeline(); } }, icon("M10 4a6 6 0 1 0 0 12 6 6 0 0 0 0-12zM14.5 14.5L20 20M7 10h6M10 7v6", 16))));
  box.classList.toggle("collapsed", E.timelineHidden);
  put(box, bar, E.timelineHidden ? null : tracks);
  if (!E.timelineHidden) {
    drawWave(canvas);
    const now = (E.t * E.pps);
    if (now < tracks.scrollLeft || now > tracks.scrollLeft + tracks.clientWidth) tracks.scrollLeft = Math.max(0, now - tracks.clientWidth / 3);
  }
}

function drawWave(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(255,255,255,.55)";
  const H = canvas.height;
  for (const part of E.parts) {
    const step = part.wave_step || 0.05;
    part.wave.forEach((v, k) => {
      const x = (part.t0 + k * step) * E.pps, w = Math.max(1, step * E.pps - 0.5);
      const hh = Math.max(1, v * (H - 4));
      ctx.fillRect(x, (H - hh) / 2, w, hh);
    });
  }
}

function blockEl(b, px) {
  const el = h("div", { class: `tl-block ${b.kind}` + (E.sel?.kind === b.kind && E.sel.i === b.i ? " sel" : ""), style: `left:${px(b.a)};width:${px(b.b - b.a)}`,
    title: `${b.label} · drag to move, drag the edge to resize` },
    h("i", { class: "tl-handle l" }), h("span", {}, b.label), h("i", { class: "tl-handle r" }));
  el.addEventListener("pointerdown", e => {
    e.stopPropagation(); e.preventDefault();
    E.sel = { kind: b.kind, i: b.i, a: b.a, b: b.b };
    const mode = e.target.classList.contains("l") ? "l" : e.target.classList.contains("r") ? "r" : "move";
    const x0 = e.clientX, a0 = b.a, b0 = b.b, len = b.b - b.a;
    const move = ev => {
      const d = (ev.clientX - x0) / E.pps;
      let a = a0, bb = b0;
      if (mode === "move") { a = Math.max(0, Math.min(E.total - len, a0 + d)); bb = a + len; }
      else if (mode === "l") a = Math.max(0, Math.min(b0 - 0.3, a0 + d));
      else bb = Math.min(E.total, Math.max(a0 + 0.3, b0 + d));
      setBlockRange(b.kind, b.i, a, bb);
      el.style.left = `${(a * E.pps).toFixed(1)}px`; el.style.width = `${((bb - a) * E.pps).toFixed(1)}px`;
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); changed(); renderPanel(); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    renderStageBar();
  });
  return el;
}

function setBlockRange(kind, i, a, b) {
  const sa = locate(a).src, sb = locate(b).src;
  if (kind === "text") Object.assign(E.edits.texts[i], { start: +sa.toFixed(3), end: +sb.toFixed(3) });
  else if (kind === "broll") Object.assign(E.edits.broll[i], { start: +sa.toFixed(3), end: +sb.toFixed(3) });
  else if (kind === "zoom") E.edits.zooms[i] = [+sa.toFixed(3), +sb.toFixed(3)];
}

function scrub(e) {
  const tracks = $("#tl-tracks");
  const rect = tracks.getBoundingClientRect();
  const at = ev => Math.max(0, Math.min(E.total, (ev.clientX - rect.left + tracks.scrollLeft) / E.pps));
  seek(at(e));
  E.sel = null; renderStageBar();
  document.querySelectorAll(".tl-seg.sel, .tl-cut.sel, .tl-block.sel").forEach(el => el.classList.remove("sel"));
  const move = ev => seek(at(ev));
  const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
  window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
}

function updatePlayhead() {
  const ph = $("#tl-playhead"), now = $("#tl-now");
  if (ph) ph.style.left = `${(E.t * E.pps).toFixed(1)}px`;
  if (now) now.textContent = fmtClock(outputTime(E.t));
  const tracks = $("#tl-tracks");
  if (tracks && E.playing) {
    const x = E.t * E.pps;
    if (x > tracks.scrollLeft + tracks.clientWidth - 40) tracks.scrollLeft = x - 40;
  }
}

function splitAtPlayhead() {
  const seg = segments().find(s => E.t > s.a + 0.05 && E.t < s.b - 0.05);
  if (!seg) return toast("Move the playhead inside a segment to split it");
  const { part, src } = locate(E.t);
  const a = locate(seg.a).src, b = Math.min(part.end, a + (seg.b - seg.a));
  const current = seg.own ? seg.layout : "auto";
  const layouts = (E.edits.layouts || []).filter(l => !(l.start < b && l.end > a));
  layouts.push({ start: +a.toFixed(3), end: +src.toFixed(3), layout: current }, { start: +src.toFixed(3), end: +b.toFixed(3), layout: current });
  E.edits.layouts = layouts.sort((x, y) => x.start - y.start);
  E.sel = { kind: "segment", a: E.t, b: seg.b };
  changed();
}

function deleteSelection() {
  const s = E.sel;
  if (!s) return toast("Select a segment, a text, a B-roll or a punch-in first");
  if (s.kind === "segment") {
    const a = locate(s.a).src, b = locate(s.b - 0.001).src + 0.001;
    E.edits.cuts_removed = [...(E.edits.cuts_removed || []), [+a.toFixed(3), +Math.min(locate(s.a).part.end, b).toFixed(3)]].sort((x, y) => x[0] - y[0]);
    toast("Cut out; apply to re-render");
  } else if (s.kind === "cut") { restoreCut(s); return; }
  else if (s.kind === "text") E.edits.texts.splice(s.i, 1);
  else if (s.kind === "broll") E.edits.broll.splice(s.i, 1);
  else if (s.kind === "zoom") E.edits.zooms.splice(s.i, 1);
  E.sel = null;
  changed(); renderPanel();
}

function restoreCut(s) {
  if (s.cutKind === "mine") E.edits.cuts_removed = (E.edits.cuts_removed || []).filter(([a, b]) => !(Math.abs(a - s.src[0]) < 0.01 && Math.abs(b - s.src[1]) < 0.01));
  else E.edits.cuts_kept = [...(E.edits.cuts_kept || []), s.src].sort((x, y) => x[0] - y[0]);
  E.sel = null;
  changed();
}

// ---------------------------------------------------------------------------------------------
// Tool rail and panels (right)

function renderRail() {
  const rail = $("#ed-rail");
  if (!rail) return;
  rail.replaceChildren(...TOOLS.map(([key, label, path]) => h("button", { class: "ed-tool" + (E.tool === key ? " active" : ""), onclick: () => {
    E.tool = E.tool === key ? null : key; renderRail(); renderPanel();
  } }, icon(path), h("span", {}, label))));
}

function renderPanel() {
  const panel = $("#ed-panel");
  if (!panel) return;
  panel.hidden = !E.tool;
  if (!E.tool) return;
  const title = TOOLS.find(t => t[0] === E.tool)[1];
  const builders = { ai: aiPanel, caption: captionPanel, text: textPanel, upload: uploadPanel, transitions: transitionsPanel, hook: hookPanel, broll: brollPanel, music: musicPanel };
  put(panel, h("div", { class: "ed-panel-head" }, h("h2", {}, title),
    h("button", { class: "btn ghost icon", "aria-label": "Close", onclick: () => { E.tool = null; renderRail(); renderPanel(); } }, "×")),
    ...builders[E.tool]());
}

const field = (label, input) => h("label", { class: "field ed-field" }, h("span", {}, label), input);
const check = (label, checked, onchange) => h("label", { class: "check" }, h("input", { type: "checkbox", checked, onchange: e => onchange(e.target.checked) }), label);
const select = (values, names, value, onchange) => h("select", { onchange: e => onchange(e.target.value) },
  values.map((v, i) => h("option", { value: v, selected: String(value) === String(v) }, names ? names[i] : v)));

function aiPanel() {
  const c = E.data.clip;
  const subs = c.scores ? h("div", { class: "subs" }, [["hook", "Hook"], ["payoff", "Payoff"], ["reaction", "Reaction"], ["pace", "Pace"]].map(([k, label]) =>
    h("div", { class: "sub" }, `${label} ${c.scores[k] ?? "–"}`, h("div", { class: "bar" }, h("i", { style: `width:${((c.scores[k] ?? 0) / 10) * 100}%` }))))) : null;
  const kws = c.keywords || [];
  const mine = E.edits.keywords ?? kws;
  const autoCuts = E.parts.reduce((n, p) => n + p.removed.length, 0);
  return [
    h("div", { class: "small muted" }, "What Claude proposed for this clip. Everything below changes it; Reset brings the proposal back."),
    h("div", { class: "ed-proposal" },
      h("div", { class: "meta" }, c.score != null ? h("span", { class: "score" }, c.score) : null, c.type ? h("span", { class: "chip" }, c.type) : null,
        h("span", {}, `${Math.round(c.final_duration || c.duration)}s`)),
      subs,
      c.hook ? h("div", { class: "hook" }, `“${c.hook}”`) : null,
      h("div", { class: "small muted", style: "margin-top:6px" }, "Keywords (highlighted, punched in on):"),
      h("div", { class: "chips" }, ...kws.map(k => h("button", { class: "chip key" + (mine.includes(k) ? "" : " off"), title: "Toggle", onclick: () => {
        const next = mine.includes(k) ? mine.filter(x => x !== k) : [...mine, k];
        E.edits.keywords = JSON.stringify(next) === JSON.stringify(kws) ? null : next; changed(); renderPanel();
      } }, k)), ...mine.filter(k => !kws.includes(k)).map(k => h("span", { class: "chip key" }, k, " (yours)")),
        !kws.length && !mine.length ? h("span", { class: "small muted" }, "none; click words on the left") : null),
      h("div", { class: "small muted", style: "margin-top:6px" }, `Jump cuts: ${autoCuts} pause${autoCuts === 1 ? "" : "s"} or filler words cut (the • marks on the left).`)),
    field("Jump cuts", select(S.choices.jump_cuts, ["off", "pauses only", "pauses + filler words"], E.edits.jump_cuts || E.data.jump_cuts, async v => {
      E.edits.jump_cuts = v === E.data.options.jump_cuts ? null : v; await reloadCuts(v);
    })),
    field("Framing of the whole clip", select(S.choices.layouts, ["auto", "follow the speaker", "split screen", "main person only", "whole frame (no cropping)"],
      E.edits.layout || E.data.layout, v => { E.edits.layout = v === E.data.options.layout ? null : v; changed(); })),
    check("Punch in on the hook and keywords", E.edits.zoom !== "off", on => { E.edits.zoom = on ? "auto" : "off"; changed(); }),
    check("Show the title at the start", E.edits.title_overlay ?? (E.data.options.title_overlay ?? true), on => { E.edits.title_overlay = on; changed(); }),
    h("div", { class: "row", style: "gap:8px; margin-top:8px" },
      h("button", { class: "btn ghost small", onclick: () => { E.edits.zooms = [...(E.edits.zooms || []), rangeAtPlayhead(1.2)]; changed(); renderPanel(); } }, "+ Punch in here"),
      h("button", { class: "btn ghost small danger", onclick: () => { if (confirm("Drop all your changes to this clip and go back to Claude's proposal?")) { E.edits = clone(ClipEditsEmpty()); E.shape = null; E.lines = null; changed(); renderPanel(); } } }, "Reset to Claude's proposal")),
  ];
}

function ClipEditsEmpty() {
  return { title: null, title_overlay: null, keywords: null, jump_cuts: null, layout: null, cuts_removed: [], cuts_kept: [],
    layouts: [], zoom: "auto", zooms: [], texts: [], broll: [], music: null, transition: "cut", fade_in: false, fade_out: false };
}

function rangeAtPlayhead(len) {
  const { part, src } = locate(E.t);
  const a = Math.min(src, part.end - 0.5), b = Math.min(part.end, a + len);
  return [+a.toFixed(3), +b.toFixed(3)];
}

async function reloadCuts(mode) {
  try {
    const fresh = await api.get(`/api/videos/${E.id}/edit/${E.rank}?jump_cuts=${encodeURIComponent(mode)}`);
    E.data.parts.forEach((p, i) => { p.removed = fresh.parts[i].removed; });
    const t = E.t; buildParts(); E.t = t;
    E.edits.cuts_kept = [];
    changed();
  } catch (e) { toast(e.message); }
}

function captionPanel() {
  const opts = options();
  const onOpt = (key, v) => { saveOption(key, v); updateStage(); };
  return [
    h("div", { class: "small muted" }, "How the captions look (for every clip you render), and fixing what Whisper misheard."),
    field("Style", select(S.choices.caption_styles, null, opts.caption_style, v => onOpt("caption_style", v))),
    field("Position", select(["lower", "middle"], null, opts.caption_position, v => onOpt("caption_position", v))),
    check("Capital letters", !!opts.uppercase, on => onOpt("uppercase", on)),
    h("div", { class: "row", style: "gap:8px; margin-top:10px" },
      h("button", { class: "btn small" + (E.fixing ? " primary" : ""), onclick: () => { E.fixing = !E.fixing; renderScript(); renderPanel(); } },
        E.fixing ? "Done fixing words" : "Fix the words"),
      E.fixing ? h("button", { class: "btn ghost small", onclick: () => { E.lines = null; renderScript(); } }, "Undo fixes") : null),
    h("div", { class: "small muted" }, E.fixing ? "The lines on the left are editable now. Click a line to hear it; your fixes are rendered when you apply."
      : "Switches the script on the left to editable lines."),
  ];
}

function textPanel() {
  const list = (E.edits.texts || []).map((t, i) => h("div", { class: "ed-item" + (E.sel?.kind === "text" && E.sel.i === i ? " sel" : "") },
    h("input", { type: "text", value: t.text, maxlength: 120, placeholder: "Text on screen", oninput: e => { t.text = e.target.value; changed(); } }),
    h("div", { class: "row", style: "gap:6px" },
      select(S.choices.text_positions, null, t.position, v => { t.position = v; changed(); }),
      h("span", { class: "small muted grow" }, `${fmt(clipTime(t.start))}–${fmt(clipTime(t.end))}`),
      h("button", { class: "btn ghost small", onclick: () => { E.sel = { kind: "text", i }; seek(clipTime(t.start)); renderTimeline(); } }, "Go"),
      h("button", { class: "btn ghost small danger", onclick: () => { E.edits.texts.splice(i, 1); E.sel = null; changed(); renderPanel(); } }, "×"))));
  return [
    h("div", { class: "small muted" }, "Your own words on the picture, like the title card. Drag them on the timeline to move or stretch them."),
    ...list,
    h("button", { class: "btn small", onclick: () => {
      const [a, b] = rangeAtPlayhead(3);
      E.edits.texts = [...(E.edits.texts || []), { text: "Your text", start: a, end: b, position: "top" }];
      E.sel = { kind: "text", i: E.edits.texts.length - 1 };
      changed(); renderPanel(); panelFocusLast();
    } }, "+ Add text at the playhead"),
  ];
}

function panelFocusLast() { const inputs = $("#ed-panel").querySelectorAll("input[type=text]"); inputs[inputs.length - 1]?.select(); }

function assetCard(a, actions) {
  return h("div", { class: "ed-asset" },
    a.kind === "image" ? h("img", { src: a.url, alt: "" }) : h("div", { class: "ed-asset-icon" }, icon(a.kind === "video" ? "M4 6h12v12H4zM16 10l4-2v8l-4-2" : "M9 18V6l10-2v12M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z", 22)),
    h("div", { class: "ed-asset-name", title: a.path }, a.name),
    actions ? h("div", { class: "row", style: "gap:4px" }, ...actions(a)) : null);
}

function uploadPanel() {
  const pick = h("input", { type: "file", accept: "image/*,video/*,audio/*", hidden: true, onchange: async e => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    status.textContent = `Uploading ${file.name}…`;
    try {
      await putFile(file, f => { status.textContent = `Uploading ${file.name}… ${Math.round(f * 100)}%`; });
      E.data.assets = await api.get("/api/assets");
      status.textContent = "";
      renderPanel();
      toast("Uploaded. Use it as B-roll or music.");
    } catch (err) { status.textContent = "Upload failed"; toast(err.message); }
  } });
  const status = h("div", { class: "small muted" });
  const use = a => a.kind === "audio"
    ? [h("button", { class: "btn ghost small", onclick: () => { E.edits.music = { file: a.path, volume: E.edits.music?.volume ?? 20 }; changed(); E.tool = "music"; renderRail(); renderPanel(); } }, "Use as music")]
    : [h("button", { class: "btn ghost small", onclick: () => { addBroll(a); } }, "B-roll here")];
  return [
    h("div", { class: "small muted" }, "Pictures and videos for B-roll, music, and logos (set the logo under More settings on the home page)."),
    h("div", { class: "ed-assets" }, ...E.data.assets.map(a => assetCard(a, use)),
      !E.data.assets.length ? h("div", { class: "small muted" }, "Nothing uploaded yet.") : null),
    h("button", { class: "btn small", onclick: () => pick.click() }, "Upload a file"), pick, status,
  ];
}

function addBroll(a) {
  const [s, e] = rangeAtPlayhead(a.kind === "image" ? 3 : 4);
  E.edits.broll = [...(E.edits.broll || []), { file: a.path, start: s, end: e }];
  E.sel = { kind: "broll", i: E.edits.broll.length - 1 };
  changed();
  E.tool = "broll"; renderRail(); renderPanel();
}

function transitionsPanel() {
  const many = E.parts.length > 1;
  return [
    h("div", { class: "small muted" }, many ? "This clip plays several parts of the video in a row." : "Between parts of a clip (add one with “Add a section”), and at the ends."),
    field("Between the parts", select(S.choices.transitions, ["hard cut", "dip to black"], E.edits.transition, v => { E.edits.transition = v; changed(); })),
    check("Fade in from black at the start", E.edits.fade_in, on => { E.edits.fade_in = on; changed(); }),
    check("Fade out to black at the end", E.edits.fade_out, on => { E.edits.fade_out = on; changed(); }),
  ];
}

function hookPanel() {
  const c = E.data.clip;
  const input = h("input", { type: "text", maxlength: 80, value: E.edits.title || c.title, oninput: e => {
    const v = e.target.value.trim();
    E.edits.title = v && v !== (c.claude_title || c.title) ? v : null; changed();
  } });
  const running = E.job?.kind === "hooks" && (E.job.state === "running" || E.job.state === "queued");
  const ask = h("button", { class: "btn small", disabled: running, onclick: async () => {
    try {
      E.job = await api.post(`/api/videos/${E.id}/edit/${E.rank}/hooks`, { options: options() });
      toast("Asking Claude for hook ideas…");
      renderPanel(); pollEditorJobs();
    } catch (e) { toast(e.message); }
  } }, running ? "Asking Claude…" : E.data.hooks ? "Ask again" : "Ask Claude for hook ideas");
  return [
    h("div", { class: "small muted" }, "The title shown as the clip starts is its hook. Change it, or let Claude suggest other angles."),
    field("Title / hook", input),
    c.claude_title || E.edits.title ? h("div", { class: "small muted" }, `Claude's: “${c.claude_title || c.title}” `,
      h("button", { class: "btn ghost small", onclick: () => { E.edits.title = null; changed(); renderPanel(); } }, "use it")) : null,
    ask,
    ...(E.data.hooks || []).map(hk => h("button", { class: "ed-hook", onclick: () => { E.edits.title = hk.title === (c.claude_title || c.title) ? null : hk.title; changed(); renderPanel(); } },
      h("b", {}, hk.title), hk.why ? h("span", { class: "small muted" }, hk.why) : null)),
  ];
}

function brollPanel() {
  const list = (E.edits.broll || []).map((b, i) => {
    const a = E.data.assets.find(x => x.path === b.file);
    return h("div", { class: "ed-item" + (E.sel?.kind === "broll" && E.sel.i === i ? " sel" : "") },
      h("div", { class: "row", style: "gap:6px" },
        h("span", { class: "grow", style: "font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap" }, a?.name || b.file),
        h("span", { class: "small muted" }, `${fmt(clipTime(b.start))}–${fmt(clipTime(b.end))}`),
        h("button", { class: "btn ghost small", onclick: () => { E.sel = { kind: "broll", i }; seek(clipTime(b.start)); renderTimeline(); } }, "Go"),
        h("button", { class: "btn ghost small danger", onclick: () => { E.edits.broll.splice(i, 1); E.sel = null; changed(); renderPanel(); } }, "×")));
  });
  const media = E.data.assets.filter(a => a.kind !== "audio");
  return [
    h("div", { class: "small muted" }, "A picture or video shown over the clip for a few seconds (the captions stay on top). Drag it on the timeline to move or stretch it."),
    ...list,
    h("div", { class: "ed-assets" }, ...media.map(a => assetCard(a, a => [h("button", { class: "btn ghost small", onclick: () => addBroll(a) }, "Add here")])),
      !media.length ? h("div", { class: "small muted" }, "Upload a picture or video under Upload first.") : null),
  ];
}

function musicPanel() {
  const audio = E.data.assets.filter(a => a.kind === "audio");
  const m = E.edits.music;
  const current = m && E.data.assets.find(a => a.path === m.file);
  return [
    h("div", { class: "small muted" }, "A music track under the sound of the clip, looped and faded out at the end."),
    m ? h("div", { class: "ed-item sel" },
      h("div", { class: "row", style: "gap:6px" }, h("span", { class: "grow", style: "font-weight:600" }, current?.name || m.file),
        h("button", { class: "btn ghost small danger", onclick: () => { E.edits.music = null; changed(); renderPanel(); } }, "×")),
      field(`Volume: ${m.volume}%`, h("input", { type: "range", min: 0, max: 100, step: 5, value: m.volume, oninput: e => { m.volume = Number(e.target.value); e.target.parentNode.querySelector("span").textContent = `Volume: ${m.volume}%`; changed(); } })))
      : null,
    h("div", { class: "ed-assets" }, ...audio.map(a => assetCard(a, a => [h("button", { class: "btn ghost small", onclick: () => { E.edits.music = { file: a.path, volume: m?.volume ?? 20 }; changed(); renderPanel(); } }, "Use")])),
      !audio.length ? h("div", { class: "small muted" }, "Upload a music file under Upload first.") : null),
  ];
}

// ---------------------------------------------------------------------------------------------
// Applying: save the changes, then render the clip again

async function applyChanges() {
  const btn = $("#ed-apply");
  const linesChanged = E.lines && Object.values(E.lines).flat().some(l => l.text !== l.orig);
  if (!dirty()) return toast("Nothing changed yet");
  btn.disabled = true;
  try {
    const body = { edits: E.edits, options: options(), render: !linesChanged };
    if (E.shape) body.parts = E.shape;
    const reply = await api.post(`/api/videos/${E.id}/edit/${E.rank}`, body);
    let job = reply.job;
    if (linesChanged) {  // the caption fix starts the render itself
      const lines = Object.values(E.lines).flat().map(l => ({ start: l.start, end: l.end, text: l.text }));
      job = await api.post(`/api/videos/${E.id}/captions/${E.rank}`, { lines, options: options() });
    }
    E.job = job;
    E.data.edits = clone(E.edits); E.data.clip = reply.clip; E.shape = null;
    if (E.lines) for (const l of Object.values(E.lines).flat()) l.orig = l.text;
    toast("Rendering the clip again…");
    renderJobPill(); pollEditorJobs();
  } catch (e) { toast(e.message); }
  btn.disabled = false;
}

function discardChanges() {
  if (!dirty()) return toast("No changes to discard");
  if (!confirm("Drop the changes you made since the last apply?")) return;
  E.edits = clone(E.data.edits); E.shape = null; E.lines = null; E.sel = null;
  changed(); renderPanel();
}

function renderJobPill() {
  const box = $("#ed-job");
  if (!box) return;
  const j = E.job;
  if (!j || !["running", "queued"].includes(j.state)) { put(box, j?.state === "failed" ? h("span", { class: "error" }, j.error) : null); return; }
  const p = j.progress;
  box.replaceChildren(h("span", { class: "dot" }), h("span", { class: "small muted" },
    [j.kind === "hooks" ? "Asking Claude" : "Rendering", j.step, p && `${Math.round(p.fraction * 100)}%`, p && fmtEta(p.eta)].filter(Boolean).join(" · ")));
}

async function pollEditorJobs() {
  clearTimeout(E.poll);
  if (!E.data) return;
  try {
    const jobs = await api.get("/api/jobs");
    const mine = jobs.filter(j => j.video === E.id && (["running", "queued"].includes(j.state) || (E.job && j.id === E.job.id)));
    const before = E.job;
    E.job = mine.find(j => before && j.id === before.id) || mine[0] || null;
    renderJobPill();
    if (before && E.job && E.job.id === before.id && before.state !== E.job.state && ["done", "failed", "cancelled"].includes(E.job.state)) {
      if (E.job.state === "done") {
        toast(E.job.kind === "hooks" ? "Claude's hook ideas are in" : "The clip is updated");
        const t = E.t, tool = E.tool;
        E.data = await api.get(`/api/videos/${E.id}/edit/${E.rank}`);
        E.edits = clone(E.data.edits); E.lines = null; E.fixing = false;
        buildParts(); E.t = Math.min(t, E.total); E.tool = tool;
        renderEditor();
      } else if (E.job.state === "failed") toast(`Failed: ${E.job.error}`);
    }
  } catch (e) { /* server restarting */ }
  const busy = E.job && ["running", "queued"].includes(E.job.state);
  E.poll = setTimeout(pollEditorJobs, busy ? 1000 : 5000);
}
