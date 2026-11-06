// SLAM results dashboard: every value shown here comes from the server's /api endpoints, which read the run
// directories under results/runs/ (see dashboard_app/data.py). Missing values are shown as "n/a", never as 0.
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const main = $("#main"), tip = $("#tip");
const enc = encodeURIComponent;
const api = p => fetch(p).then(r => r.json().then(j => { if (!r.ok) throw new Error(j.error || r.statusText); return j; }));
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const isNum = v => typeof v === "number" && isFinite(v);
function f(v, d = 3, unit = "") { return isNum(v) ? `${v.toFixed(d)}${unit}` : `<span class="na" title="not available">n/a</span>`; }
function cm(v, d = 2) { return isNum(v) ? `${(100 * v).toFixed(d)} cm` : f(null); }
function pair2(a, b) { return `${a} <span class="muted">|</span> ${b}`; }
const CLASS = {I: ["PnP inlier", "#0072B2"], O: ["PnP outlier", "#CC3311"], D: ["no depth at the frame-i keypoint (not given to PnP)", "#7f7f7f"], U: ["unclassified (PnP solver failed)", "#000"]};

// ---------- metric definitions (tooltips and the run page) ----------
const DEF = {
  frames: "Frames the evaluator processed (selected frames with RGB, depth and ground truth).",
  stride: "Every stride-th usable frame is used; 'n/a' when the run directory does not record it.",
  span: "Time between the first and last processed frame (s).",
  ok: "Pairs whose pose was accepted / not accepted (solver failure, or a returned pose rejected by the geometric validity check where the run records it); not accepted = pose held.",
  ate_raw: "Absolute trajectory error: RMSE of camera-centre distances to ground truth over all frames, raw (frame 0 at ground truth).",
  ate_al: "ATE after ONE global rigid transform per method (Umeyama, no scale) fitted to all camera centres. Marked when that fit rotates the estimate by more than the caution angle.",
  final: "Distance between the last estimated and ground-truth camera centre / angle between their orientations.",
  rpe1: "Relative pose error over 1 frame (RMSE): (Q_k^-1 Q_k+1)^-1 (P_k^-1 P_k+1); windows spanning a failed pair excluded.",
  rpe1s: "Relative pose error over the frame count closest to 1 s (from the median frame interval); n/a when the run is shorter.",
  inl: "PnP / ICP inliers per successful pair: mean and minimum.",
  matches: "Filtered feature matches per pair: mean and minimum.",
  warn: "Pairs with a diagnostic warning (no ground truth used): low PnP inliers, low inlier ratio, low ICP inliers or PnP-ICP disagreement. Warnings never change the solver status.",
  exp: "Match-level export (tum_trajectory_eval --export-pairs) of this run, or of an equivalent run with byte-identical results.",
};

// ---------- routing ----------
const routes = {runs: viewRuns, run: viewRun, pair: viewPair, scene: viewScene, reg: viewScene, compare: viewCompare};
let current = {run: null, pair: null};
function go(h) { location.hash = h; }
window.addEventListener("hashchange", route);
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  const name = parts[0] || "runs";
  const navName = {reg: "scene", duo: "methods", pairx: "methods", corr: "corr"}[name] || name;
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("on", a.dataset.nav === navName));
  (routes[name] || viewRuns)(...parts.slice(1));
}
function setNav() {
  const r = current.run;
  $("#navRun").href = r ? `#/run/${enc(r)}` : ""; $("#navRun").classList.toggle("off", !r);
  $("#navPair").href = r ? `#/pair/${enc(r)}/${current.pair ?? 0}` : ""; $("#navPair").classList.toggle("off", !r);
  $("#navScene").href = r ? `#/scene/${enc(r)}` : ""; $("#navScene").classList.toggle("off", !r);
  $("#crumb").textContent = r ? r : "";
}
function fail(e) { main.innerHTML = `<div class="missing"><b>Could not load:</b> ${esc(e.message || e)}</div>`; }
function showTip(e, html) { tip.innerHTML = html; tip.hidden = false; tip.style.left = Math.min(e.clientX + 14, innerWidth - 440) + "px"; tip.style.top = (e.clientY + 14) + "px"; }
function hideTip() { tip.hidden = true; }

// ---------- runs overview ----------
let RUNS = null, sortKey = "id", sortDir = 1;
const COLS = [
  ["id", "Run", r => r.meta.id, r => `<a href="#/run/${enc(r.meta.id)}">${esc(r.meta.id)}</a>`, ""],
  ["seq", "Sequence", r => r.meta.sequence, r => esc(r.meta.sequence ?? "n/a"), ""],
  ["frames", "Frames", r => r.metrics.frames, r => `${r.metrics.frames}${r.meta.frames_requested !== null && r.meta.frames_requested !== undefined ? ` <span class="muted">/ ${r.meta.frames_requested === 0 ? "all" : r.meta.frames_requested}</span>` : ""}`, DEF.frames],
  ["stride", "Stride", r => r.meta.stride, r => r.meta.stride ?? f(null), DEF.stride],
  ["span", "Span s", r => r.metrics.time_span_s, r => f(r.metrics.time_span_s, 1), DEF.span],
  ["pnp_ok", "PnP ok/fail", r => r.metrics.pnp.failed_pairs, r => okfail(r.metrics.pnp) + (r.metrics.pnp.rejected_pairs ? ` <span class="badge bad" title="returned by the solver but rejected by the geometric validity check (counted in fail)">${r.metrics.pnp.rejected_pairs} rejected</span>` : ""), DEF.ok],
  ["icp_ok", "ICP ok/fail", r => r.metrics.icp.failed_pairs, r => okfail(r.metrics.icp), DEF.ok],
  ["ate", "ATE raw m PnP|ICP", r => r.metrics.pnp.ate_raw_rmse_m, r => pair2(f(r.metrics.pnp.ate_raw_rmse_m), f(r.metrics.icp.ate_raw_rmse_m)), DEF.ate_raw],
  ["ate_al", "ATE aligned m PnP|ICP", r => r.metrics.pnp.ate_aligned_rmse_m, r => pair2(alcell(r.metrics.pnp), alcell(r.metrics.icp)), DEF.ate_al],
  ["fin", "Final pos m PnP|ICP", r => r.metrics.pnp.final_position_error_m, r => pair2(f(r.metrics.pnp.final_position_error_m), f(r.metrics.icp.final_position_error_m)), DEF.final],
  ["finr", "Final rot deg PnP|ICP", r => r.metrics.pnp.final_orientation_error_deg, r => pair2(f(r.metrics.pnp.final_orientation_error_deg, 1), f(r.metrics.icp.final_orientation_error_deg, 1)), DEF.final],
  ["rpe", "RPE 1 frame cm|deg PnP / ICP", r => r.metrics.pnp.rpe_1frame?.trans_rmse_m, r => rpecell(r, "rpe_1frame"), DEF.rpe1],
  ["rpe1s", "RPE ~1 s cm|deg PnP / ICP", r => r.metrics.pnp.rpe_1s?.trans_rmse_m, r => rpecell(r, "rpe_1s"), DEF.rpe1s],
  ["inl", "Inliers mean/min PnP|ICP", r => r.metrics.pnp.inliers_mean_successful, r => pair2(`${f(r.metrics.pnp.inliers_mean_successful, 0)}/${r.metrics.pnp.inliers_min_successful ?? f(null)}`, `${f(r.metrics.icp.inliers_mean_successful, 0)}/${r.metrics.icp.inliers_min_successful ?? f(null)}`), DEF.inl],
  ["matches", "Matches mean/min", r => r.metrics.matches_mean, r => `${f(r.metrics.matches_mean, 0)}/${r.metrics.matches_min ?? f(null)}`, DEF.matches],
  ["warn", "Warnings", r => r.warnings, r => r.warnings ? `<span class="badge warn">${r.warnings}</span>` + (r.gt_flags ? ` <span class="badge bad" title="pairs with a ground-truth error flag (evaluation only)">${r.gt_flags} GT</span>` : "") : (r.gt_flags ? `<span class="badge bad">${r.gt_flags} GT</span>` : `<span class="muted">0</span>`), DEF.warn],
  ["exp", "Match export", r => r.meta.export_source ? 1 : 0, r => r.meta.export_source ? (r.meta.export_source === r.meta.id ? `<span class="badge ok">own: ${r.meta.exports.length} pairs</span>` : `<span class="badge ok" title="${esc(r.meta.export_source)}">equivalent run</span>`) : `<span class="muted">none</span>`, DEF.exp],
];
function okfail(m) { return `${m.successful_pairs}/<span class="${m.failed_pairs ? "bad" : "muted"}">${m.failed_pairs}</span>`; }
function alcell(m) { return m.aligned_ate_caution ? `<span class="warn" title="${esc(m.aligned_ate_caution)}">${f(m.ate_aligned_rmse_m)}&#9888;</span>` : f(m.ate_aligned_rmse_m); }
function rpecell(r, k) { const a = r.metrics.pnp[k], b = r.metrics.icp[k];
  if (!a) return `<span class="na" title="${esc((r.metrics.notes || []).join("; ") || "not available")}">n/a</span>`;
  return `${f(100 * a.trans_rmse_m, 2)}|${f(a.rot_rmse_deg, 2)} <span class="muted">/</span> ${f(100 * b.trans_rmse_m, 2)}|${f(b.rot_rmse_deg, 2)}`; }

async function viewRuns() {
  current.pair = null; setNav();
  main.innerHTML = `<h1>All evaluation runs</h1><p class="muted">Loading runs from results/runs/&hellip;</p>`;
  try { RUNS = await api("/api/runs"); } catch (e) { return fail(e); }
  const runs = RUNS.runs, uniq = k => [...new Set(runs.map(k))].filter(x => x !== null && x !== undefined).sort((a, b) => a > b ? 1 : -1);
  const fp = {}; runs.forEach(r => (fp[r.meta.fingerprint] = fp[r.meta.fingerprint] || []).push(r.meta.id));
  main.innerHTML = `<h1>All evaluation runs <span class="muted">(${runs.length} run directories under results/runs/)</span></h1>
  <div class="row">
    <input id="q" placeholder="filter by run name" size="28">
    <label>sequence <select id="fseq"><option value="">all</option>${uniq(r => r.meta.sequence).map(s => `<option>${esc(s)}</option>`).join("")}</select></label>
    <label>session <select id="fses"><option value="">all</option>${uniq(r => r.meta.session).map(s => `<option>${esc(s)}</option>`).join("")}</select></label>
    <label>frames <select id="ffr"><option value="">all</option>${uniq(r => r.metrics.frames).map(s => `<option>${s}</option>`).join("")}</select></label>
    <label>stride <select id="fst"><option value="">all</option>${uniq(r => r.meta.stride).map(s => `<option>${s}</option>`).join("")}<option value="na">unknown</option></select></label>
    <label><input type="checkbox" id="fdup"> hide runs whose results duplicate an earlier row</label>
  </div>
  <div class="tablewrap"><table id="runs"></table></div>
  <p class="muted">PnP|ICP side by side. &#9888; = aligned ATE to interpret with care (the global fit rotates the estimate by more than ${RUNS.align_caution_deg}&deg;; hover). Solver success is the evaluator's definition (a pose was returned); the evaluator has no separate tracking-validity check, so none is shown. Click a run for details, its timeline and its visual evidence.</p>
  <h2>Flagged pairs</h2><div id="flagged" class="muted">Loading&hellip;</div>
  ${RUNS.incomplete.length || RUNS.errors.length ? `<h2>Not loaded</h2><ul class="tight">${RUNS.incomplete.map(i => `<li>${esc(i.id)}: missing ${esc(i.missing.join(", "))}</li>`).concat(RUNS.errors.map(e => `<li class="bad">${esc(e.id)}: ${esc(e.error)}</li>`)).join("")}</ul>` : ""}`;
  const render = () => {
    const q = $("#q").value.toLowerCase(), s1 = $("#fseq").value, s2 = $("#fses").value, s3 = $("#ffr").value, s4 = $("#fst").value, dup = $("#fdup").checked;
    let rows = runs.filter(r => (!q || r.meta.id.toLowerCase().includes(q)) && (!s1 || r.meta.sequence === s1) && (!s2 || r.meta.session === s2)
      && (!s3 || String(r.metrics.frames) === s3) && (!s4 || (s4 === "na" ? r.meta.stride === null : String(r.meta.stride) === s4)));
    const col = COLS.find(c => c[0] === sortKey);
    rows.sort((a, b) => { const x = col[2](a), y = col[2](b); if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1; return (x > y ? 1 : -1) * sortDir; });
    if (dup) { const seen = new Set(); rows = rows.filter(r => !seen.has(r.meta.fingerprint) && seen.add(r.meta.fingerprint)); }
    $("#runs").innerHTML = `<tr>${COLS.map(c => `<th class="sort ${c[0] === "id" || c[0] === "seq" ? "" : "num"}" data-k="${c[0]}" title="${esc(c[4] || "")}">${c[1]}${c[0] === sortKey ? (sortDir > 0 ? " &#9650;" : " &#9660;") : ""}</th>`).join("")}</tr>` +
      rows.map(r => `<tr class="click" data-id="${esc(r.meta.id)}">${COLS.map(c => `<td class="${c[0] === "id" || c[0] === "seq" ? "" : "num"}">${c[3](r)}${c[0] === "id" && fp[r.meta.fingerprint].length > 1 ? ` <span class="badge" title="byte-identical results (pairs.csv, trajectory.csv) in: ${esc(fp[r.meta.fingerprint].join(", "))}">&equiv;${fp[r.meta.fingerprint].length}</span>` : ""}</td>`).join("")}</tr>`).join("");
    document.querySelectorAll("#runs th.sort").forEach(th => th.onclick = () => { sortDir = th.dataset.k === sortKey ? -sortDir : 1; sortKey = th.dataset.k; render(); });
    document.querySelectorAll("#runs tr.click").forEach(tr => tr.onclick = e => { if (e.target.tagName !== "A") go(`#/run/${enc(tr.dataset.id)}`); });
  };
  ["#q", "#fseq", "#fses", "#ffr", "#fst", "#fdup"].forEach(s => $(s).oninput = render);
  render();
  // flagged pairs across runs (deduplicated by identical results)
  const seen = new Set(), items = [];
  for (const r of runs) { if (!r.warnings && !r.gt_flags) continue; if (seen.has(r.meta.fingerprint)) continue; seen.add(r.meta.fingerprint);
    try { const tl = await api(`/api/timeline?id=${enc(r.meta.id)}`);
      tl.pairs.filter(p => p.warnings.length || p.gt_flags.length).forEach(p => items.push({r, p, also: fp[r.meta.fingerprint].length - 1})); } catch (e) { /* listed in the run page */ } }
  items.sort((a, b) => (b.p.pnp_rot_err_deg ?? 0) - (a.p.pnp_rot_err_deg ?? 0));
  $("#flagged").innerHTML = items.length ? `<table><tr><th>run</th><th>pair</th><th>warnings (no GT)</th><th>GT flags</th><th class="num">PnP inliers</th><th class="num">PnP err deg / m</th><th class="num">ICP err deg / m</th><th>evidence</th></tr>${items.map(({r, p, also}) =>
    `<tr><td>${esc(r.meta.id)}${also ? ` <span class="muted">(+${also} identical)</span>` : ""}</td><td>${p.frame_i}&rarr;${p.frame_j}</td><td>${esc(p.warnings.join(", "))}</td><td class="bad">${esc(p.gt_flags.join(", "))}</td>
    <td class="num">${p.pnp_inliers}/${p.pnp_correspondences}</td><td class="num">${f(p.pnp_rot_err_deg, 2)} / ${f(p.pnp_trans_err_m, 3)}</td><td class="num">${f(p.icp_rot_err_deg, 2)} / ${f(p.icp_trans_err_m, 3)}</td>
    <td><a href="#/pair/${enc(r.meta.id)}/${p.pair}">${p.exported ? "open pair" : "open (no match export: shows the command)"}</a></td></tr>`).join("")}</table>` : `<span class="muted">No pair carries a warning or ground-truth flag.</span>`;
}

// ---------- one run ----------
async function viewRun(id) {
  current.run = id; setNav();
  main.innerHTML = `<h1>${esc(id)}</h1><p class="muted">Loading&hellip;</p>`;
  let s, tl; try { [s, tl] = await Promise.all([api(`/api/run?id=${enc(id)}`), api(`/api/timeline?id=${enc(id)}`)]); } catch (e) { return fail(e); }
  const m = s.meta, M = s.metrics;
  const rows = [
    ["Pairs: solver returned a pose / failed", x => okfail(x), DEF.ok],
    ["Held poses (after a failed pair)", x => x.held_poses, "Frames whose pose was copied from the previous frame because the pair failed."],
    ["ATE raw (m)", x => f(x.ate_raw_rmse_m, 4), DEF.ate_raw],
    ["ATE after one global SE(3) (m)", x => alcell(x) + (x.aligned_ate_caution ? `<div class="def warn">${esc(x.aligned_ate_caution)}</div>` : ""), DEF.ate_al],
    ["  global fit: rotation (deg) / translation (m)", x => `${f(x.alignment_rotation_deg, 1)} / ${f(x.alignment_translation_m, 3)}`, "The single rigid transform used for the aligned ATE."],
    ["Final position error (m)", x => f(x.final_position_error_m, 4), DEF.final],
    ["Final orientation error (deg)", x => f(x.final_orientation_error_deg, 2), DEF.final],
    ["RPE 1 frame: translation (cm) / rotation (deg) RMSE", x => x.rpe_1frame ? `${f(100 * x.rpe_1frame.trans_rmse_m, 2)} / ${f(x.rpe_1frame.rot_rmse_deg, 3)}` : f(null), DEF.rpe1],
    [`RPE ~1 s (${M.rpe_interval_frames_for_seconds} frames): translation (cm) / rotation (deg)`, x => x.rpe_1s ? `${f(100 * x.rpe_1s.trans_rmse_m, 2)} / ${f(x.rpe_1s.rot_rmse_deg, 3)}` : f(null), DEF.rpe1s],
    ["Per-pair error mean: rotation (deg) / translation (m)", x => `${f(x.pair_rot_err_mean_deg, 3)} / ${f(x.pair_trans_err_mean_m, 4)}`, "Mean over successful pairs of the relative-pose error vs ground truth (pairs.csv)."],
    ["Correspondences per pair (mean)", x => f(x.correspondences_mean, 1), "PnP: matches with depth in frame i. ICP: matches with depth in both frames."],
    ["Inliers per successful pair: mean / min", x => `${f(x.inliers_mean_successful, 1)} / ${x.inliers_min_successful ?? f(null)}`, DEF.inl],
  ];
  main.innerHTML = `<h1>${esc(id)}</h1>
  <div class="row"><button onclick="go('#/scene/${enc(id)}')">3D scene</button><button onclick="go('#/pair/${enc(id)}/${firstFlagged(tl)}')">Pair explorer</button><button onclick="go('#/compare/${enc(id)}')">Compare with another run</button></div>
  <div class="grid2"><div>
  <h2>Configuration</h2><dl class="meta">
   <dt>sequence</dt><dd>${esc(m.dataset ?? "n/a")} ${m.dataset_available ? "" : `<span class="bad">(not found: images unavailable)</span>`}</dd>
   <dt>frames</dt><dd>${M.frames} processed${m.frames_requested !== null ? `, requested ${m.frames_requested === 0 ? "all" : m.frames_requested}` : ""}; ${M.attempted_pairs} pairs; span ${f(M.time_span_s, 2)} s; median frame interval ${f(s.median_frame_interval_s, 4)} s</dd>
   <dt>stride</dt><dd>${m.stride ?? "n/a (not recorded)"}</dd>
   <dt>command</dt><dd><code>${esc(m.command ?? "n/a")}</code></dd>
   <dt>configuration from</dt><dd>${esc(m.config_source)}</dd>
   <dt>code state</dt><dd>${m.git_commit ? `${esc(m.git_commit.slice(0, 10))}${m.git_dirty ? " + uncommitted changes" : ""}` : "n/a"}${m.started ? `, started ${esc(m.started)}` : ""}</dd>
   <dt>runner status</dt><dd>${m.status ? `${esc(m.status)} (exit ${m.exit_code}, ${f(m.wall_time_s, 1)} s)` : "n/a (not a stress-runner case)"}${m.runner_problems && m.runner_problems.length ? `<div class="bad">${esc(m.runner_problems.join("; "))}</div>` : ""}</dd>
   <dt>consistency</dt><dd>${s.consistency_problems.length ? `<span class="bad">${esc(s.consistency_problems.join("; "))}</span>` : "CSV errors match the poses; frame 0 at ground truth"}</dd>
   <dt>tracking validity</dt><dd>${M.pnp.solver_returned_pairs !== null ? `PnP: solver returned ${M.pnp.solver_returned_pairs}, accepted ${M.pnp.successful_pairs}, <span class="${M.pnp.rejected_pairs ? "bad" : ""}">rejected ${M.pnp.rejected_pairs}</span> (geometric validity check; rejected = pose held)` : `<span class="muted">not recorded: run made before the evaluator's PnP validity check (a returned pose counted as success)</span>`}</dd>
   <dt>match export</dt><dd>${m.export_source ? (m.export_source === id ? `own: pairs ${esc(m.exports.join(", "))}` : `from equivalent run ${esc(m.export_source)} (byte-identical results)`) : "none (pair views show the command to create one)"}</dd>
   <dt>frame gaps</dt><dd>${s.frame_gaps.length ? s.frame_gaps.map(g => `pair ${g.pair}: ${g.dt_s.toFixed(3)} s`).join(", ") + ` <span class="muted">(&gt; 1.5 &times; median interval)</span>` : "none"}</dd>
   ${M.notes.length ? `<dt>notes</dt><dd>${esc(M.notes.join("; "))}</dd>` : ""}
  </dl></div><div>
  <h2>PnP vs ICP</h2><table><tr><th>metric</th><th class="num" style="color:var(--pnp)">PnP</th><th class="num" style="color:var(--icp)">ICP</th></tr>
  ${rows.map(([lab, fn, d]) => `<tr title="${esc(d)}"><td>${lab}</td><td class="num">${fn(M.pnp)}</td><td class="num">${fn(M.icp)}</td></tr>`).join("")}
  <tr title="${esc(DEF.matches)}"><td>Filtered matches per pair: mean / min</td><td class="num" colspan="2">${f(M.matches_mean, 1)} / ${M.matches_min ?? f(null)}</td></tr>
  <tr><td>Max RGB-depth time offset</td><td class="num" colspan="2">${f(M.rgb_depth_dt_ms_max_abs, 1, " ms")}</td></tr>
  </table><p class="def">Hover a row for its definition. Errors are raw unless marked aligned. Metrics are recomputed from this run's CSVs by tum_metrics.compute_metrics.</p></div></div>
  <h2>Timeline <span class="muted">(click a pair to open its evidence)</span></h2>
  <canvas id="timeline"></canvas>
  <p class="def">Lanes: counts (log) &middot; PnP inlier ratio &middot; rotation (deg, log): PnP and ICP error vs GT, PnP-ICP disagreement (no GT), true motion &middot; translation (m, log) &middot; markers: solver failed (black), PnP pose rejected by the validity check (purple), warning (orange), GT flag (red), frame gap (blue), match export available (green tick). Thresholds: ${esc(JSON.stringify(tl.thresholds))}.</p>`;
  drawTimeline($("#timeline"), tl, id);
}
function firstFlagged(tl) { const p = tl.pairs.find(p => p.warnings.length || p.gt_flags.length); return p ? p.pair : (tl.pairs.find(p => p.exported) || {pair: 0}).pair; }

function drawTimeline(cv, tl, id) {
  const P = tl.pairs, n = P.length, dpr = window.devicePixelRatio || 1;
  const draw = () => {
    const r = cv.getBoundingClientRect(); cv.width = r.width * dpr; cv.height = r.height * dpr;
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height);
    const L = 64, R = 12, W = r.width - L - R, lanes = [[8, 110], [126, 60], [194, 120], [322, 110], [440, 72]];
    const x = k => L + (n > 1 ? k / (n - 1) : 0.5) * W;
    g.font = "11px system-ui"; g.fillStyle = "#5b6168";
    const series = (lane, vals, col, lo, hi, log, dash) => { const [y0, h] = lane; g.strokeStyle = col; g.lineWidth = 1.4; g.setLineDash(dash || []); g.beginPath(); let on = false;
      vals.forEach((v, k) => { if (!isNum(v) || (log && v <= 0)) { on = false; return; } const t = log ? (Math.log(Math.max(v, lo)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo);
        const yy = y0 + h - Math.min(1, Math.max(0, t)) * h; if (on) g.lineTo(x(k), yy); else g.moveTo(x(k), yy); on = true; }); g.stroke(); g.setLineDash([]); };
    const frame = (lane, label, lo, hi) => { const [y0, h] = lane; g.strokeStyle = "#d9dce0"; g.strokeRect(L, y0, W, h); g.fillStyle = "#5b6168";
      g.fillText(label, 4, y0 + 12); if (lo !== undefined) { g.fillText(String(hi), 4, y0 + 26); g.fillText(String(lo), 4, y0 + h - 2); } };
    const mx = Math.max(...P.map(p => p.matches), 10);
    frame(lanes[0], "count", 1, mx); series(lanes[0], P.map(p => p.matches), "#666", 1, mx, true); series(lanes[0], P.map(p => p.pnp_inliers), "#ff7f0e", 1, mx, true); series(lanes[0], P.map(p => p.icp_inliers), "#9467bd", 1, mx, true);
    frame(lanes[1], "PnP ratio", 0, 1); series(lanes[1], P.map(p => p.pnp_inlier_ratio), "#ff7f0e", 0, 1, false);
    const rmax = Math.max(1, ...P.map(p => Math.max(p.pnp_rot_err_deg || 0, p.icp_rot_err_deg || 0, p.pnp_icp_rot_deg || 0)));
    frame(lanes[2], "rot deg", 0.01, +rmax.toPrecision(2)); [["pnp_rot_err_deg", "#ff7f0e"], ["icp_rot_err_deg", "#9467bd"], ["gt_motion_deg", "#2ca02c"]].forEach(([k, c]) => series(lanes[2], P.map(p => p[k]), c, 0.01, rmax, true));
    series(lanes[2], P.map(p => p.pnp_icp_rot_deg), "#000", 0.01, rmax, true, [4, 3]);
    const tmax = Math.max(0.01, ...P.map(p => Math.max(p.pnp_trans_err_m || 0, p.icp_trans_err_m || 0, p.pnp_icp_trans_m || 0)));
    frame(lanes[3], "trans m", 0.0001, +tmax.toPrecision(2)); [["pnp_trans_err_m", "#ff7f0e"], ["icp_trans_err_m", "#9467bd"], ["gt_motion_m", "#2ca02c"]].forEach(([k, c]) => series(lanes[3], P.map(p => p[k]), c, 0.0001, tmax, true));
    series(lanes[3], P.map(p => p.pnp_icp_trans_m), "#000", 0.0001, tmax, true, [4, 3]);
    const [y0, h] = lanes[4]; frame(lanes[4], "status"); const med = median(P.map(p => p.dt_s));
    const bw = Math.max(2, W / Math.max(n, 1));
    P.forEach((p, k) => { const xx = x(k) - bw / 2;
      if (!p.pnp_solver_ok || !p.icp_solver_ok) { g.fillStyle = "#000"; g.fillRect(xx, y0 + 4, bw, 12); }
      if (p.pnp_rejected) { g.fillStyle = "#7b2cbf"; g.fillRect(xx, y0 + 4, bw, 12); }
      if (p.warnings.length) { g.fillStyle = "#e69f00"; g.fillRect(xx, y0 + 18, bw, 12); }
      if (p.gt_flags.length) { g.fillStyle = "#CC3311"; g.fillRect(xx, y0 + 32, bw, 12); }
      if (p.dt_s > 1.5 * med) { g.fillStyle = "#1f77b4"; g.fillRect(xx, y0 + 46, bw, 8); }
      if (p.exported) { g.fillStyle = "#2ca02c"; g.fillRect(x(k) - 1, y0 + 58, 2, 10); } });
    cv._x = k => x(k); cv._L = L; cv._W = W;
  };
  draw(); window.addEventListener("resize", draw, {once: true});
  const pick = e => { const r = cv.getBoundingClientRect(), xx = e.clientX - r.left; if (xx < cv._L) return null; return Math.max(0, Math.min(n - 1, Math.round((xx - cv._L) / cv._W * (n - 1)))); };
  cv.onmousemove = e => { const k = pick(e); if (k === null) return hideTip(); const p = P[k];
    showTip(e, `pair ${p.frame_i}&rarr;${p.frame_j}  t ${p.t_i.toFixed(3)}  dt ${p.dt_s.toFixed(3)} s\nmatches ${p.matches}; PnP ${p.pnp_inliers}/${p.pnp_correspondences}; ICP ${p.icp_inliers}/${p.icp_correspondences}\nPnP err ${f(p.pnp_rot_err_deg, 2)} deg / ${f(p.pnp_trans_err_m, 3)} m; ICP ${f(p.icp_rot_err_deg, 2)} / ${f(p.icp_trans_err_m, 3)}\nPnP-ICP ${f(p.pnp_icp_rot_deg, 2)} deg / ${f(p.pnp_icp_trans_m, 3)} m\nsolver PnP ${p.pnp_solver_ok ? "returned" : "FAILED"}${p.pnp_rejected ? ` but REJECTED (returned pose ${f(p.pnp_returned_rot_err_deg, 1)} deg / ${f(p.pnp_returned_trans_err_m, 2)} m off)` : ""}, ICP ${p.icp_solver_ok ? "returned" : "FAILED"}${p.warnings.length ? "\nwarnings: " + p.warnings.join(", ") : ""}${p.gt_flags.length ? "\nGT flags: " + p.gt_flags.join(", ") : ""}\n${p.exported ? "match export available" : "no match export"}`); };
  cv.onmouseleave = hideTip;
  cv.onclick = e => { const k = pick(e); if (k !== null) { hideTip(); go(`#/pair/${enc(id)}/${P[k].pair}`); } };
}
function median(a) { const b = a.filter(isNum).sort((x, y) => x - y); return b.length ? b[b.length >> 1] : 0; }

// ---------- pair explorer ----------
const PS = {I: true, O: true, U: true, kp: true, lines: true, depth: true, res: "none", maxl: 80};
async function viewPair(id, k, res) {
  k = +k || 0; if (res === "pnp" || res === "gt") PS.res = res;  // #/pair/<run>/<k>/pnp preselects residuals current.run = id; current.pair = k; setNav();
  main.innerHTML = `<h1>${esc(id)}: pair ${k}&rarr;${k + 1}</h1><p class="muted">Loading&hellip;</p>`;
  let pr, tl; try { [pr, tl] = await Promise.all([api(`/api/pair?id=${enc(id)}&k=${k}`), api(`/api/timeline?id=${enc(id)}`)]); } catch (e) { return fail(e); }
  const n = tl.pairs.length, tp = tl.pairs[k];
  const nav = `<div class="row"><button ${k <= 0 ? "disabled" : ""} onclick="go('#/pair/${enc(id)}/${k - 1}')">&larr; previous</button>
    <label>pair <input type="number" id="pk" min="0" max="${n - 1}" value="${k}"></label> <span class="muted">of 0&ndash;${n - 1}</span>
    <button ${k >= n - 1 ? "disabled" : ""} onclick="go('#/pair/${enc(id)}/${k + 1}')">next &rarr;</button>
    <button onclick="nextFlag(${k})" title="next pair with a warning or GT flag">next flagged</button>
    <a href="#/run/${enc(id)}">back to run</a></div>`;
  window.nextFlag = from => { const p = tl.pairs.find(p => p.pair > from && (p.warnings.length || p.gt_flags.length)) || tl.pairs.find(p => p.warnings.length || p.gt_flags.length); if (p) go(`#/pair/${enc(id)}/${p.pair}`); };
  const summary = tp ? `<p>Frame ${tp.frame_i} (t = ${tp.t_i.toFixed(6)} s) &rarr; frame ${tp.frame_j}; matches ${tp.matches}, PnP ${tp.pnp_inliers}/${tp.pnp_correspondences}, ICP ${tp.icp_inliers}/${tp.icp_correspondences};
    error vs GT: PnP ${f(tp.pnp_rot_err_deg, 2)} deg / ${f(tp.pnp_trans_err_m, 3)} m, ICP ${f(tp.icp_rot_err_deg, 2)} deg / ${f(tp.icp_trans_err_m, 3)} m (pairs.csv).
    ${tp.warnings.length ? `<span class="badge warn">${esc(tp.warnings.join(", "))}</span>` : ""} ${tp.gt_flags.length ? `<span class="badge bad">${esc(tp.gt_flags.join(", "))}</span>` : ""}</p>` : "";
  if (!pr.export.available) {
    const ex = pr.export;
    main.innerHTML = `<h1>${esc(id)}: pair ${k}&rarr;${k + 1}</h1>${nav}${summary}
      <div class="missing"><b>No match-level data for this pair.</b> ${esc(ex.reason)}.<br>Smallest opt-in export (run from the repository root):<pre>${esc(ex.command)}</pre>${esc(ex.note)}</div>
      <p><a href="#/scene/${enc(id)}/${k + 1}">Open frame ${k + 1} in the 3D scene</a> (poses and depth are available without an export).</p>`;
    $("#pk").onchange = e => go(`#/pair/${enc(id)}/${+e.target.value}`);
    return;
  }
  const D = pr.data, c = D.counts, I = pr.indicators, st = pr.solver_status;
  const ratio = c.pnp_correspondences ? (100 * c.pnp_inliers / c.pnp_correspondences).toFixed(1) : "n/a";
  main.innerHTML = `<h1>${esc(id)}: pair ${D.i}&rarr;${D.j}</h1>${nav}
  ${pr.export.equivalent ? `<p class="muted">Match data from the equivalent run <code>${esc(pr.export.source_run)}</code> (same configuration, byte-identical pairs.csv and trajectory.csv).</p>` : ""}
  ${pr.export.problems.length ? `<div class="missing bad">Export inconsistencies: ${esc(pr.export.problems.join("; "))}</div>` : ""}
  <div class="row">
    <label><input type="checkbox" data-t="I"> <span class="sw" style="background:#0072B2"></span>PnP inliers</label>
    <label><input type="checkbox" data-t="O"> <span class="sw" style="background:#CC3311"></span>PnP outliers</label>
    <label><input type="checkbox" data-t="U"> <span class="sw" style="background:#7f7f7f"></span>unclassified (no depth / solver failed)</label>
    <label><input type="checkbox" data-t="kp"> <span class="sw" style="border:2px solid #ffd400"></span>all keypoints</label>
    <label><input type="checkbox" data-t="lines"> correspondence lines</label>
    <label><input type="checkbox" data-t="depth"> depth</label>
    <label>reprojection residuals <select id="res"><option value="none">off</option><option value="pnp">under the returned PnP pose</option><option value="gt">under the ground-truth pose</option></select></label>
    <label>max lines per class <input type="number" id="maxl" min="1"></label>
    <span id="drawn" class="muted"></span>
  </div>
  <canvas id="pairCanvas"></canvas>
  <div id="depth"></div>
  <div class="grid2"><div>
  <h2>Counts and status</h2><table>
   <tr><td>frames / timestamps</td><td>${D.i} (t = ${D.t_i.toFixed(6)} s) &rarr; ${D.j} (t = ${D.t_j.toFixed(6)} s)</td></tr>
   <tr><td>keypoints i / j</td><td>${c.keypoints_i} / ${c.keypoints_j} <span class="muted">(detector cap 500)</span></td></tr>
   <tr><td>raw matches</td><td>${f(pr.raw_matches)} <span class="muted">${esc(pr.raw_matches_note)}</span></td></tr>
   <tr><td>filtered matches</td><td>${c.matches}</td></tr>
   <tr><td>matches with depth: frame i / both frames</td><td>${c.depth_valid_i} / ${c.icp_correspondences}</td></tr>
   <tr><td>PnP correspondences / inliers / ratio</td><td>${c.pnp_correspondences} / ${c.pnp_inliers} / ${ratio} %</td></tr>
   <tr><td>distinct keypoints among PnP inliers (i / j)</td><td>${I.pnp_inliers_distinct_kp_i} / ${I.pnp_inliers_distinct_kp_j}</td></tr>
   <tr><td>matches sharing a frame-j keypoint</td><td>${I.matches_sharing_a_kp_j} of ${c.matches} (at most ${I.max_matches_on_one_kp_j} on one)</td></tr>
   <tr><td>ICP correspondences / inliers</td><td>${c.icp_correspondences} / ${c.icp_inliers} <span class="muted">(threshold ${D.icp_threshold_m * 100} cm)</span></td></tr>
   <tr><td>solver status</td><td>PnP solver returned: <b>${st.pnp_solver_returned}</b> (counted successful: ${st.pnp_counted_successful}); ICP RANSAC ok: <b>${st.icp_ransac_ok}</b> (counted successful: ${st.icp_counted_successful})</td></tr>
   <tr><td>tracking validity</td><td>${pr.tracking_validity ? `PnP accepted: <b>${pr.tracking_validity.pnp_accepted}</b>${pr.tracking_validity.reason ? ` <span class="bad">(${esc(pr.tracking_validity.reason)})</span>` : ""}<div class="def">${esc(pr.tracking_validity_note)}</div>` : `<span class="muted">${esc(pr.tracking_validity_note)}</span>`}</td></tr>
   <tr><td>error vs GT (from the exported transforms)</td><td>PnP ${f(I.pose_errors_from_exported_transforms.pnp?.rot_err_deg, 2)} deg / ${f(I.pose_errors_from_exported_transforms.pnp?.trans_err_m, 4)} m; ICP ${f(I.pose_errors_from_exported_transforms.icp?.rot_err_deg, 2)} deg / ${f(I.pose_errors_from_exported_transforms.icp?.trans_err_m, 4)} m; true motion ${f(I.pose_errors_from_exported_transforms.gt.rotation_deg, 2)} deg / ${f(I.pose_errors_from_exported_transforms.gt.translation_m, 4)} m</td></tr>
  </table>
  <div class="row"><button onclick="go('#/reg/${enc(id)}/${D.pair}/icp')">ICP registration and camera poses in 3D</button>${pr.page ? `<a href="/output/${pr.page}" target="_blank">static diagnostic page</a>` : ""}</div>
  </div><div>
  <h2>Evidence</h2><h3>Measured (from the export)</h3><ul class="tight">${pr.facts.map(x => `<li>${esc(x)}</li>`).join("")}</ul>
  <h3>Hypotheses (rule-based; not established)</h3><ul class="tight">${pr.hypotheses.map(x => `<li>${esc(x)}</li>`).join("")}</ul>
  ${D.refits.length ? `<h3>Export-only re-solves on the PnP inliers</h3><ul class="tight">${D.refits.map(r => `<li><code>${esc(r.name)}</code>: ${esc(r.status.split("\n")[0].slice(0, 120))}</li>`).join("")}</ul>` : ""}
  </div></div>`;
  document.querySelectorAll("[data-t]").forEach(cb => { cb.checked = PS[cb.dataset.t]; cb.onchange = () => { PS[cb.dataset.t] = cb.checked; drawPair(id, D); }; });
  $("#res").value = PS.res; $("#res").onchange = e => { PS.res = e.target.value; drawPair(id, D); };
  $("#maxl").value = PS.maxl; $("#maxl").onchange = e => { PS.maxl = Math.max(1, +e.target.value); drawPair(id, D); };
  $("#pk").onchange = e => go(`#/pair/${enc(id)}/${+e.target.value}`);
  drawPair(id, D);
}
const IMG = {};
function img(src, cb) { if (!IMG[src]) { IMG[src] = new Image(); IMG[src].src = src; } const im = IMG[src]; if (im.complete && im.naturalWidth) return im; im.addEventListener("load", cb, {once: true}); return null; }
function sampleRows(rows, cap) { if (rows.length <= cap) return rows; const s = Math.ceil(rows.length / cap), o = []; for (let i = 0; i < rows.length; i += s) o.push(rows[i]); return o; }
function drawPair(id, D) {
  const cv = $("#pairCanvas"); if (!cv) return; const W = 640, H = 480, G = 16, top = 22;
  cv.width = 2 * W + G; cv.height = H + top; const g = cv.getContext("2d"); g.fillStyle = "#fff"; g.fillRect(0, 0, cv.width, cv.height);
  const redraw = () => drawPair(id, D), a = img(`/img/rgb?id=${enc(id)}&k=${D.i}`, redraw), b = img(`/img/rgb?id=${enc(id)}&k=${D.j}`, redraw);
  if (a) g.drawImage(a, 0, top); if (b) g.drawImage(b, W + G, top);
  g.font = "bold 14px system-ui"; g.fillStyle = "#111";
  g.fillText(`frame ${D.i}   t = ${D.t_i.toFixed(6)} s   (${D.counts.keypoints_i} keypoints)`, 4, 16);
  g.fillText(`frame ${D.j}   t = ${D.t_j.toFixed(6)} s   (${D.counts.keypoints_j} keypoints)`, W + G + 4, 16);
  if (PS.kp) { g.strokeStyle = "rgba(255,212,0,0.95)"; g.lineWidth = 1;
    for (const [kp, ox] of [[D.kp_i, 0], [D.kp_j, W + G]]) for (let n = 0; n < kp.length; n += 2) { g.beginPath(); g.arc(kp[n] + ox, kp[n + 1] + top, 2.5, 0, 6.3); g.stroke(); } }
  const by = {I: [], O: [], D: [], U: []}; for (let r = 0; r < D.m.cls.length; r++) by[D.m.cls[r]].push(r);
  const want = {I: PS.I, O: PS.O, D: PS.U, U: PS.U}, drawn = [];
  for (const cl of ["D", "U", "O", "I"]) { if (!want[cl] || !by[cl].length) continue; const rows = sampleRows(by[cl], PS.maxl);
    drawn.push(`${CLASS[cl][0].split(" (")[0]} ${rows.length}/${by[cl].length}`); g.strokeStyle = g.fillStyle = CLASS[cl][1];
    g.lineWidth = cl === "I" ? 1.8 : 1; g.globalAlpha = cl === "I" ? 0.95 : 0.6; g.setLineDash(cl === "D" || cl === "U" ? [3, 3] : []);
    for (const r of rows) { const x0 = D.m.ui[r], y0 = D.m.vi[r] + top, x1 = D.m.uj[r] + W + G, y1 = D.m.vj[r] + top;
      if (PS.lines) { g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); }
      g.fillRect(x0 - 2.5, y0 - 2.5, 5, 5); g.fillRect(x1 - 2.5, y1 - 2.5, 5, 5); }
    g.setLineDash([]); g.globalAlpha = 1; }
  // many-to-one: a frame-j keypoint that carries at least max(3, 20 %) of the PnP inliers
  const cnt = {}; by.I.forEach(r => cnt[D.m.kpj[r]] = (cnt[D.m.kpj[r]] || 0) + 1); const ni = by.I.length;
  for (const kj in cnt) if (cnt[kj] >= Math.max(3, 0.2 * ni)) { const x = D.kp_j[2 * kj] + W + G, y = D.kp_j[2 * kj + 1] + top;
    g.strokeStyle = "#0072B2"; g.lineWidth = 3; g.beginPath(); g.arc(x, y, 16, 0, 6.3); g.stroke();
    const txt = `${cnt[kj]} inlier matches end on this one keypoint (#${kj})`; g.font = "bold 13px system-ui"; const tw = g.measureText(txt).width, tx = x > W + G + W * 0.55 ? x - 20 - tw : x + 20;
    g.fillStyle = "#0072B2"; g.fillRect(tx - 4, y - 34, tw + 8, 20); g.fillStyle = "#fff"; g.fillText(txt, tx, y - 19); }
  // reprojection residuals on frame j: observed u_j -> projection of X_i under the chosen pose
  if (PS.res !== "none") { g.save(); g.beginPath(); g.rect(W + G, top, W, H); g.clip();
    const pu = D.m[PS.res + "_u"], pv = D.m[PS.res + "_v"];
    for (const cl of ["O", "I"]) { if (!want[cl]) continue; g.strokeStyle = CLASS[cl][1]; g.lineWidth = cl === "I" ? 2.4 : 1; g.globalAlpha = cl === "I" ? 1 : 0.7;
      for (const r of by[cl]) { if (pu[r] === null) continue; g.beginPath(); g.moveTo(D.m.uj[r] + W + G, D.m.vj[r] + top); g.lineTo(pu[r] + W + G, pv[r] + top); g.stroke(); } }
    g.restore(); g.globalAlpha = 1; drawn.push(`residual vectors under the ${PS.res === "pnp" ? "returned PnP" : "ground-truth"} pose (all PnP correspondences, clipped to frame j)`); }
  $("#drawn").textContent = "drawn (every k-th match row per class): " + drawn.join(", ");
  const dp = $("#depth"); if (!dp) return;
  if (!PS.depth) { dp.innerHTML = ""; return; }
  if (!dp.dataset.pair || dp.dataset.pair !== String(D.pair)) {
    dp.dataset.pair = D.pair;
    dp.innerHTML = `<h2>Depth <span class="muted">(depth in metres = raw / 5000, magenta = no measurement; white circle = matched keypoint with depth, red cross = without)</span></h2><div class="depthgrid">` +
      [[D.i, "i"], [D.j, "j"]].flatMap(([fk, s]) => [["color", `frame ${fk}: depth`], ["mask", `frame ${fk}: valid depth (white)`]].map(([mode, lab]) =>
        `<div><div class="muted">${lab}</div><div class="depthcell"><img src="/img/depth?id=${enc(id)}&k=${fk}&mode=${mode}" alt="${lab}"><canvas width="640" height="480" data-s="${s}"></canvas></div></div>`)).join("") + `</div>`;
    dp.querySelectorAll("canvas").forEach(c => { const s = c.dataset.s, x = D.m["u" + s], y = D.m["v" + s], d = D.m["depth_" + s], g2 = c.getContext("2d");
      for (let r = 0; r < x.length; r++) { if (d[r] > 0) { g2.fillStyle = "#fff"; g2.strokeStyle = "#000"; g2.beginPath(); g2.arc(x[r], y[r], 3.5, 0, 6.3); g2.fill(); g2.stroke(); }
        else { g2.strokeStyle = "#e00"; g2.lineWidth = 2; g2.beginPath(); g2.moveTo(x[r] - 4, y[r] - 4); g2.lineTo(x[r] + 4, y[r] + 4); g2.moveTo(x[r] + 4, y[r] - 4); g2.lineTo(x[r] - 4, y[r] + 4); g2.stroke(); } } });
  }
}

// ---------- 3D: scene and registration ----------
let V3 = null;
async function viewScene(id, a, b, c) {
  const reg = location.hash.startsWith("#/reg/");
  current.run = id; setNav();
  main.innerHTML = `<h1>${esc(id)}: 3D</h1><p class="muted">Loading poses and sampled depth&hellip;</p>`;
  let S; try { S = await api(`/api/scene?id=${enc(id)}&extra=${reg ? `${a},${+a + 1}` : (a || "")}`); } catch (e) { return fail(e); }
  const meta = (await api(`/api/run?id=${enc(id)}`)).meta, src = meta.export_source;
  const pairs = src ? (await api(`/api/run?id=${enc(src)}`)).meta.exports : [];
  const N = S.frames.length;
  main.innerHTML = `<h1>${esc(id)}: 3D</h1>
  <div class="row"><label><input type="radio" name="mode" value="scene"> scene (TUM world frame, z up, metres)</label>
    <label><input type="radio" name="mode" value="registration" ${pairs.length ? "" : "disabled"}> ICP registration of pair <select id="rp">${pairs.map(p => `<option>${p}</option>`).join("")}</select></label>
    ${pairs.length ? "" : `<span class="muted">registration needs a match export of this run</span>`}</div>
  <div class="row" id="rowScene">${["gt", "pnp", "icp"].map(m => `<label><input type="checkbox" data-m="${m}" checked> <span class="sw" style="background:${METHOD_COLOR[m]}"></span>${METHOD_LABEL[m]}</label>`).join("")}
    <label><input type="checkbox" id="o_points" checked> point cloud placed with <select id="o_place"><option value="gt">GT</option><option value="pnp">PnP</option><option value="icp">ICP</option></select> poses</label>
    <label><input type="checkbox" id="o_frustums" checked> frustums at the selected frame</label>
    <label><input type="checkbox" id="o_connectors" checked> GT&rarr;estimate connectors every <input type="number" id="o_every" value="5" min="1" style="width:3.5em"> frames</label>
    <label><select id="o_align"><option value="raw">raw estimates</option><option value="aligned">aligned: one global SE(3) per method</option></select></label></div>
  <div class="row" id="rowReg"><label><input type="radio" name="st" value="before"> before (identity)</label><label><input type="radio" name="st" value="icp"> after ICP</label>
    <label><input type="radio" name="st" value="gt"> after ground truth</label><label><input type="checkbox" id="o_rpts" checked> point clouds</label>
    <label><input type="checkbox" id="o_corr" checked> ICP correspondences</label><label><input type="checkbox" id="o_cams" checked> camera j poses (GT / PnP / ICP)</label></div>
  <div class="wrap3d"><canvas id="c3d"></canvas><div class="side3d">
    <div class="row"><label>frame <input type="range" id="fr" min="0" max="${N - 1}" style="width:190px"></label> <input type="number" id="frn" min="0" max="${N - 1}" style="width:4.5em"></div>
    <img id="prgb" alt="RGB of the selected frame"><img id="pdep" alt="depth of the selected frame"><pre id="pose"></pre><div id="leg" class="def"></div></div></div>
  <p class="def">Scene cloud: depth of frames ${esc(S.cloud_frames.join(", "))} (at most ${S.cloud_frames.length} frames, evenly spaced, plus requested ones) on a ${S.cloud_step_px} px grid, back-projected with the ${esc(S.K ? "official" : "unknown")} intrinsics (depth = raw / 5000 m), stored per camera and placed with the chosen method's pose. ${S.dataset_available ? "" : "Dataset unavailable: no point cloud."}
  Aligned option: ${Object.entries(S.alignment).map(([m, v]) => `${m} ${v.rotation_deg.toFixed(1)} deg / ${v.translation_m.toFixed(3)} m`).join("; ")}. Drag: rotate, shift-drag: pan, wheel: zoom, click a camera: select its frame.</p>`;
  V3 = new View3D($("#c3d"), k => setFrame(k)); V3.setScene(S);
  const o = V3.o;
  const setFrame = k => { o.frame = Math.max(0, Math.min(N - 1, k)); $("#fr").value = $("#frn").value = o.frame; side(); V3.draw(); };
  const side = () => { const k = o.frame, fr = S.frames[k]; $("#prgb").src = `/img/rgb?id=${enc(id)}&k=${k}`; $("#pdep").src = `/img/depth?id=${enc(id)}&k=${k}&mode=color`;
    let t = `frame ${k}   t = ${fr.t.toFixed(6)} s\n`; const g = S.poses.gt.raw[k];
    for (const m of ["gt", "pnp", "icp"]) { const P = V3.pose(m, k); t += `${METHOD_LABEL[m].padEnd(12)} centre (${P[9].toFixed(3)}, ${P[10].toFixed(3)}, ${P[11].toFixed(3)})\n`;
      if (m !== "gt") t += `${"".padEnd(12)} error ${(100 * V3.err(m, k)).toFixed(2)} cm, ${V3.rotErr(m, k).toFixed(2)} deg${S.held[m][k] ? "  (held: failed pair)" : ""}\n`; }
    $("#pose").textContent = t + `(${o.align}; T_wc, X_world = R X_cam + t)`;
    if (V3.mode === "registration" && V3.reg) { const p = V3.reg.p, r = p.reg[o.state];
      $("#leg").innerHTML = `Pair ${p.i}&rarr;${p.j} in camera ${p.i}'s frame (x right, y down, z forward). Target = frame ${p.i} (blue), source = frame ${p.j} (orange). ICP correspondences ${p.counts.icp_correspondences}, inliers ${p.counts.icp_inliers} (threshold ${p.icp_threshold_m * 100} cm); solid = inlier, dashed = every ${p.icp.rej_every}th rejected; colour = residual 0&ndash;5 cm (viridis).${r ? ` This state: inlier median residual ${(100 * r.inlier_median_m).toFixed(2)} cm, ${r.within_threshold} of ${p.counts.icp_correspondences} within threshold.` : ""} Frustums: camera i black, camera j by GT / PnP / ICP.`; }
    else $("#leg").innerHTML = `circle = frame 0, square = last frame; camera axes x red, y cyan, z (optical) blue. Connectors: GT centre &rarr; estimated centre of the same frame, colour = |error| on a log scale from &le;1 cm to ${(100 * V3.errMax).toFixed(1)} cm (viridis).`; };
  $("#fr").oninput = e => setFrame(+e.target.value); $("#frn").onchange = e => setFrame(+e.target.value);
  document.querySelectorAll("[data-m]").forEach(cb => cb.onchange = () => { o.show[cb.dataset.m] = cb.checked; V3.draw(); });
  for (const k of ["points", "frustums", "connectors"]) $("#o_" + k).onchange = e => { o[k] = e.target.checked; V3.draw(); };
  $("#o_rpts").onchange = e => { o.points = e.target.checked; V3.draw(); }; $("#o_corr").onchange = e => { o.corr = e.target.checked; V3.draw(); }; $("#o_cams").onchange = e => { o.cams = e.target.checked; V3.draw(); };
  $("#o_place").onchange = e => { o.place = e.target.value; V3.draw(); }; $("#o_align").onchange = e => { o.align = e.target.value; side(); V3.draw(); };
  $("#o_every").onchange = e => { o.every = Math.max(1, +e.target.value); V3.draw(); };
  document.querySelectorAll("input[name=st]").forEach(r => r.onchange = () => { o.state = r.value; side(); V3.draw(); });
  const setMode = async (mode, k, state) => {
    $("#rowScene").style.display = mode === "scene" ? "" : "none"; $("#rowReg").style.display = mode === "registration" ? "" : "none";
    document.querySelectorAll("input[name=mode]").forEach(r => r.checked = r.value === mode);
    if (mode === "registration") { k = +(k ?? $("#rp").value); $("#rp").value = k; o.state = state || o.state;
      document.querySelectorAll("input[name=st]").forEach(r => r.checked = r.value === o.state);
      const R = await api(`/api/registration?id=${enc(id)}&k=${k}`); if (!R.registration) { $("#leg").textContent = "no registration data"; return; }
      V3.setRegistration(R.data, R.registration);
      if (V3.mode === "registration") V3.v = Object.assign({}, V3.views.registration); else V3.setMode("registration");
      setFrame(R.data.j); current.pair = k; setNav(); }
    else { V3.setMode("scene"); setFrame(o.frame); } };
  document.querySelectorAll("input[name=mode]").forEach(r => r.onchange = () => setMode(r.value));
  $("#rp").onchange = () => setMode("registration");
  window.onresize = () => V3 && V3.draw();
  if (reg) await setMode("registration", +a, b); else { await setMode("scene"); setFrame(a !== undefined ? +a : 0); }
}

// ---------- compare ----------
async function viewCompare(a, b) {
  setNav(); main.innerHTML = `<h1>Compare runs</h1><p class="muted">Loading&hellip;</p>`;
  let G; try { G = await api("/api/groups"); } catch (e) { return fail(e); }
  const all = G.flatMap(g => g.runs.map(r => r.id)).sort();
  a = a || all[0]; b = b || (G.find(g => g.runs.some(r => r.id === a))?.runs.map(r => r.id).find(x => x !== a)) || all[1];
  const opts = sel => all.map(x => `<option ${x === sel ? "selected" : ""}>${esc(x)}</option>`).join("");
  main.innerHTML = `<h1>Compare runs</h1>
  <p><b>Implementations side by side</b> (reference PnP / ICP vs scratch PnP on the same frames): <a href="#/methods">Methods view</a>.</p>
  <p>Baseline vs post-fix: a comparison needs a run made after a change. Runs are grouped by configuration below; a group with one distinct result has no post-fix run yet.</p>
  <table><tr><th>configuration</th><th class="num">runs</th><th class="num">distinct results</th><th>runs (code state)</th></tr>${G.map(g => `<tr><td>${esc(g.configuration)}</td><td class="num">${g.runs.length}</td><td class="num">${g.distinct_results === 1 ? `1 <span class="muted">(no post-fix result)</span>` : `<b>${g.distinct_results}</b>`}</td><td>${g.runs.map(r => `${esc(r.id)} <span class="muted">${r.git_commit ? r.git_commit.slice(0, 8) + (r.git_dirty ? "+dirty" : "") : "code state n/a"} &middot; ${r.fingerprint.slice(0, 8)}</span>`).join("<br>")}</td></tr>`).join("")}</table>
  <h2>Side by side</h2><div class="row"><label>A <select id="ca">${opts(a)}</select></label><label>B <select id="cb">${opts(b)}</select></label></div><div id="cmp"></div>`;
  const run = async () => { const A = $("#ca").value, B = $("#cb").value; let C; try { C = await api(`/api/compare?a=${enc(A)}&b=${enc(B)}`); } catch (e) { $("#cmp").innerHTML = `<div class="missing">${esc(e.message)}</div>`; return; }
    const rows = [["frames", x => x.metrics.frames], ["PnP ok / failed", x => `${x.metrics.pnp.successful_pairs} / ${x.metrics.pnp.failed_pairs}`], ["ICP ok / failed", x => `${x.metrics.icp.successful_pairs} / ${x.metrics.icp.failed_pairs}`]];
    const num = [["PnP ATE raw (m)", x => x.metrics.pnp.ate_raw_rmse_m], ["ICP ATE raw (m)", x => x.metrics.icp.ate_raw_rmse_m], ["PnP ATE aligned (m)", x => x.metrics.pnp.ate_aligned_rmse_m], ["ICP ATE aligned (m)", x => x.metrics.icp.ate_aligned_rmse_m],
      ["PnP final position (m)", x => x.metrics.pnp.final_position_error_m], ["ICP final position (m)", x => x.metrics.icp.final_position_error_m], ["PnP final orientation (deg)", x => x.metrics.pnp.final_orientation_error_deg], ["ICP final orientation (deg)", x => x.metrics.icp.final_orientation_error_deg],
      ["PnP RPE 1 frame trans (m)", x => x.metrics.pnp.rpe_1frame?.trans_rmse_m], ["ICP RPE 1 frame trans (m)", x => x.metrics.icp.rpe_1frame?.trans_rmse_m], ["PnP inliers mean", x => x.metrics.pnp.inliers_mean_successful], ["ICP inliers mean", x => x.metrics.icp.inliers_mean_successful], ["warnings (pairs)", x => x.warnings]];
    $("#cmp").innerHTML = `${C.identical_results ? `<div class="missing">A and B have byte-identical results (pairs.csv, trajectory.csv): no difference to compare.</div>` : ""}${C.same_configuration ? "" : `<div class="missing">A and B differ in configuration (dataset, frames or stride): differences are not a before/after comparison.</div>`}
    <table><tr><th>metric</th><th class="num">A</th><th class="num">B</th><th class="num">B - A</th></tr>${rows.map(([l, fn]) => `<tr><td>${l}</td><td class="num">${fn(C.a)}</td><td class="num">${fn(C.b)}</td><td></td></tr>`).join("")}
    ${num.map(([l, fn]) => { const x = fn(C.a), y = fn(C.b); return `<tr><td>${l}</td><td class="num">${f(x, 4)}</td><td class="num">${f(y, 4)}</td><td class="num">${isNum(x) && isNum(y) ? (y - x).toFixed(4) : f(null)}</td></tr>`; }).join("")}</table>`; };
  $("#ca").onchange = $("#cb").onchange = () => go(`#/compare/${enc($("#ca").value)}/${enc($("#cb").value)}`);
  run();
}

window.addEventListener("DOMContentLoaded", route);
