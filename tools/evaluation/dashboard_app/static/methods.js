// Side-by-side comparison of implementations: reference pipeline (PnP, ICP) vs scratch PnP, on the same frames.
// All values come from /api/sources, /api/source_compare, /api/source_scene and /api/pair_compare (methods.py);
// nothing here computes a metric. Sources are compared only when their frames.csv selections are identical.
"use strict";

const SRC_COLOR = {gt: "#2ca02c", ref_pnp: "#ff7f0e", ref_icp: "#9467bd", scratch_pnp: "#17becf"};
let SRC = null;
async function sources() { if (!SRC) SRC = await api("/api/sources"); return SRC; }
function srcBadge(s) { return `<span class="sw" style="background:${SRC_COLOR[s.method]}"></span><b>${esc(s.label)}</b>`; }
function pickReference(runs) {  // prefer a run that records the PnP validity check (latest evaluator)
  return runs.find(r => r.includes("pnp_validity")) || runs[runs.length - 1];
}

// ---------- #/methods[/<frames_key>/<reference run>/<scratch run>] ----------
async function viewMethods(key, refRun, scrRun) {
  current.pair = null; setNav();
  main.innerHTML = `<h1>Methods side by side</h1><p class="muted">Loading&hellip;</p>`;
  let S; try { S = await sources(); } catch (e) { return fail(e); }
  const cfgs = S.configurations.filter(c => c.families.includes("reference") && c.families.includes("scratch"));
  const lone = S.configurations.filter(c => !(c.families.includes("reference") && c.families.includes("scratch")));
  if (!cfgs.length) { main.innerHTML = `<h1>Methods side by side</h1><div class="missing">No configuration has both a reference run and a scratch run with the same frames.csv. Create a scratch run with:<pre>build/scratch_pipeline --tum-frames &lt;reference run&gt;/frames.csv --tum-sequence data/&lt;sequence&gt; --intrinsics fx,fy,cx,cy --out results/runs/scratch/&lt;sequence&gt;/&lt;name&gt;</pre></div>`; return; }
  const cfg = cfgs.find(c => c.frames_key === key) || cfgs[0];
  refRun = cfg.reference_runs.includes(refRun) ? refRun : pickReference(cfg.reference_runs);
  scrRun = cfg.scratch_runs.includes(scrRun) ? scrRun : cfg.scratch_runs[0];
  const ids = [`${refRun}::ref_pnp`, `${refRun}::ref_icp`, `${scrRun}::scratch_pnp`];
  main.innerHTML = `<h1>Methods side by side</h1>
  <p>Reference pipeline (project FAST + rotated BRIEF features Reference pipeline (project ORB-style FAST + rotated BRIEF features with OpenCV PnP, and robust ICP)mdash; not OpenCV ORB Reference pipeline (project ORB-style FAST + rotated BRIEF features with OpenCV PnP, and robust ICP)mdash; with OpenCV PnP, and robust ICP) vs our scratch PnP (OpenCV-free), on <b>identical frames, timestamps and ground truth</b> (same frames.csv). Ground truth is the comparison target, not a method.</p>
  <div class="row"><label>configuration <select id="mcfg">${cfgs.map(c => `<option value="${c.frames_key}" ${c === cfg ? "selected" : ""}>${esc(c.sequence)}: ${c.frames} frames${c.stride ? `, stride ${c.stride}` : ""} [${c.frames_key.slice(0, 8)}]</option>`).join("")}</select></label>
   <label>reference run <select id="mref">${cfg.reference_runs.map(r => `<option ${r === refRun ? "selected" : ""}>${esc(r)}</option>`).join("")}</select></label>
   <label>scratch run <select id="mscr">${cfg.scratch_runs.map(r => `<option ${r === scrRun ? "selected" : ""}>${esc(r)}</option>`).join("")}</select></label></div>
  <div id="mtable" class="muted">Loading metrics&hellip;</div>
  <div class="row"><a href="#/corr">OpenCV ORB vs Scratch PnP correspondences</a></div>
  <div class="row"><button id="b3d">3D side by side (Reference PnP | Scratch PnP)</button> <button id="bpair">pair comparison</button> <label>pair <input type="number" id="mpk" value="0" min="0" style="width:5em"></label></div>
  <h2>Largest per-pair errors <span class="muted">(vs ground truth; click to compare that pair)</span></h2><div id="mworst" class="muted"></div>
  ${lone.length ? `<h2>Configurations without both implementations</h2><ul class="tight">${lone.map(c => `<li>${esc(c.sequence)}: ${c.frames} frames${c.stride ? `, stride ${c.stride}` : ""} [${c.frames_key.slice(0, 8)}] &mdash; ${c.families.includes("scratch") ? "scratch only" : "reference only (no scratch run with these frames)"}</li>`).join("")}</ul>` : ""}`;
  const nav = () => go(`#/methods/${$("#mcfg").value}/${enc($("#mref").value)}/${enc($("#mscr").value)}`);
  $("#mcfg").onchange = () => go(`#/methods/${$("#mcfg").value}`); $("#mref").onchange = nav; $("#mscr").onchange = nav;
  $("#b3d").onclick = () => go(`#/duo/${enc(ids[0])}/${enc(ids[2])}`);
  $("#bpair").onclick = () => go(`#/pairx/${enc(refRun)}/${enc(scrRun)}/${+$("#mpk").value}`);
  let C; try { C = await api(`/api/source_compare?ids=${ids.map(enc).join(",")}`); } catch (e) { $("#mtable").innerHTML = `<div class="missing">${esc(e.message)}</div>`; return; }
  const cols = C.sources;
  const row = (label, fn, d) => `<tr title="${esc(d || "")}"><td>${label}</td>${cols.map(c => `<td class="num">${fn(c.method, c)}</td>`).join("")}</tr>`;
  const rpe = (k, part) => (m) => m[k] ? (part === "t" ? f(100 * m[k].trans_rmse_m, 2) : f(m[k].rot_rmse_deg, 3)) : `<span class="na" title="undefined for this run length">n/a</span>`;
  $("#mtable").innerHTML = `<table><tr><th>metric</th>${cols.map(c => `<th class="num">${srcBadge(c.source)}<div class="def">${esc(c.source.run)}</div></th>`).join("")}</tr>
   ${row("frames / pairs", (m, c) => `${c.frames} / ${c.attempted_pairs}`)}
   ${row("pairs with a used pose (valid estimates)", m => m.successful_pairs, DEF.ok)}
   ${row("failed or rejected pairs (pose held)", m => `<span class="${m.failed_pairs ? "bad" : ""}">${m.failed_pairs}</span>`)}
   ${row("&nbsp;&nbsp;of which: solver returned but rejected", m => m.rejected_pairs ?? `<span class="na" title="this implementation / run records no separate validity check">n/a</span>`)}
   ${row("ATE raw RMSE (m)", m => f(m.ate_raw_rmse_m, 4), DEF.ate_raw)}
   ${row("ATE after one global SE(3) (m) &mdash; labelled diagnostic", m => alcell(m), DEF.ate_al)}
   ${row("final position error (m)", m => f(m.final_position_error_m, 4), DEF.final)}
   ${row("final orientation error (deg)", m => f(m.final_orientation_error_deg, 2), DEF.final)}
   ${row("RPE 1 frame translation (cm)", rpe("rpe_1frame", "t"), DEF.rpe1)}
   ${row("RPE 1 frame rotation (deg)", rpe("rpe_1frame", "r"), DEF.rpe1)}
   ${row(`RPE ~1 s (${cols[0].rpe_interval_frames_for_seconds} frames) translation (cm)`, rpe("rpe_1s", "t"), DEF.rpe1s)}
   ${row(`RPE ~1 s rotation (deg)`, rpe("rpe_1s", "r"), DEF.rpe1s)}
   ${row("per-pair error mean: rotation (deg)", m => f(m.pair_rot_err_mean_deg, 3))}
   ${row("per-pair error mean: translation (m)", m => f(m.pair_trans_err_mean_m, 4))}
   ${row("correspondences per pair (mean)", m => f(m.correspondences_mean, 1), "PnP: matches with frame-i depth; ICP: matches with depth in both frames")}
   ${row("inliers per used pair: mean / min", m => `${f(m.inliers_mean_successful, 1)} / ${m.inliers_min_successful ?? f(null)}`)}
   ${row("filtered matches per pair: mean / min", (m, c) => `${f(c.matches_mean, 1)} / ${c.matches_min}`, "each implementation's own feature matcher")}
   ${row("self-consistency of the source files", (m, c) => c.consistency_problems.length ? `<span class="bad">${esc(c.consistency_problems.join("; "))}</span>` : "ok")}
   <tr><td>source files</td>${cols.map(c => `<td class="def">${c.source.source_files.map(esc).join("<br>")}</td>`).join("")}</tr>
   <tr><td>implementation</td>${cols.map(c => `<td class="def">${esc(c.source.implementation)}</td>`).join("")}</tr>
  </table><p class="def">Same frames (frames.csv hash ${esc(C.frames_key)}), same ground truth, same metric code (tum_metrics.compute_metrics). Raw estimates; the aligned ATE uses one rigid transform per method (no scale), shown only as a labelled diagnostic. Runtime is not recorded by either implementation, so it is not compared.</p>`;
  // worst pairs per method: from the reference timeline and the scratch pairs (via pair_compare on demand would be slow): use the timeline
  try {
    const tl = await api(`/api/timeline?id=${enc(refRun)}`);
    const worst = tl.pairs.map(p => ({k: p.pair, e: Math.max(p.pnp_rot_err_deg ?? p.pnp_returned_rot_err_deg ?? 0, p.pnp_returned_rot_err_deg ?? 0), w: p.warnings.concat(p.gt_flags)}))
      .sort((a, b) => b.e - a.e).slice(0, 6);
    $("#mworst").innerHTML = `<p class="def">Ordered by the reference PnP rotation error (returned pose); open each to see all three methods on that pair.</p>` +
      worst.map(x => `<a href="#/pairx/${enc(refRun)}/${enc(scrRun)}/${x.k}">pair ${x.k}&rarr;${x.k + 1}</a> (${x.e.toFixed(2)} deg${x.w.length ? "; " + esc(x.w.join(", ")) : ""})`).join(" &middot; ");
  } catch (e) { $("#mworst").textContent = e.message; }
}

// ---------- #/duo/<sourceA>/<sourceB>[/<frame>]: two synchronized 3D panels ----------
async function viewDuo(a, b, frame) {
  setNav();
  main.innerHTML = `<h1>3D side by side</h1><p class="muted">Loading&hellip;</p>`;
  let S; try { S = await sources(); } catch (e) { return fail(e); }
  const A = S.sources.find(s => s.id === a), B = S.sources.find(s => s.id === b);
  if (!A || !B) return fail(new Error("unknown method source"));
  if (A.frames_key !== B.frames_key) return fail(new Error("the two sources use different frames (frames.csv differs): no side-by-side comparison"));
  const same = S.sources.filter(s => s.frames_key === A.frames_key);
  main.innerHTML = `<h1>3D side by side <span class="muted">(same frames, same viewpoint; each cloud placed with its own poses)</span></h1>
  <div class="row"><label><input type="checkbox" id="dpts" checked> point clouds</label><label><input type="checkbox" id="dfr" checked> frustums</label>
   <label><input type="checkbox" id="dcon" checked> GT&rarr;estimate connectors</label><label><input type="checkbox" id="dgt" checked> GT trajectory</label>
   <label>frame <input type="range" id="dfrm" min="0" max="0" style="width:220px"></label><input type="number" id="dfrn" style="width:4.5em"></div>
  <div class="grid2" id="duo">${[A, B].map((s, n) => `<div><div class="row"><label>panel ${n ? "B" : "A"} <select data-p="${n}">${same.map(x => `<option value="${esc(x.id)}" ${x.id === s.id ? "selected" : ""}>${esc(x.label)} &mdash; ${esc(x.run)}</option>`).join("")}</select></label></div>
    <div class="panel" id="dlab${n}"></div><canvas id="dcv${n}" style="width:100%;height:56vh;border:1px solid var(--line);touch-action:none;cursor:grab"></canvas></div>`).join("")}</div>
  <div class="row" id="dprev"></div>`;
  document.querySelectorAll("[data-p]").forEach(sel => sel.onchange = () => { const ids = [a, b]; ids[+sel.dataset.p] = sel.value; go(`#/duo/${enc(ids[0])}/${enc(ids[1])}/${views[0] ? views[0].o.frame : 0}`); });
  const views = [];
  let scenes;
  try { scenes = await Promise.all([A, B].map(s => api(`/api/source_scene?id=${enc(s.id)}`))); } catch (e) { return fail(e); }
  const N = scenes[0].frames.length, k0 = Math.min(N - 1, +frame || 0);
  let syncing = false;
  scenes.forEach((sc, n) => {
    const v = new View3D($(`#dcv${n}`), k => setFrame(k));
    v.o.place = sc.method;  // each panel: the cloud is placed with its own method's poses
    v.setScene(sc); v.views.scene = Object.assign({}, scenes[0].view); v.v = Object.assign({}, scenes[0].view);
    v.onView = st => { if (syncing) return; syncing = true; views.forEach(o => { if (o !== v) { o.v = Object.assign({}, st); o.draw(); } }); syncing = false; };
    views.push(v);
    const held = sc.held[sc.method].map((h, k) => h ? k : null).filter(x => x !== null);
    $(`#dlab${n}`).innerHTML = `${srcBadge(sc.source)} <span class="muted">${esc(sc.source.run)}</span><br>
      Point cloud placed with <b>${esc(sc.source.label)}</b> poses (raw). <span class="sw" style="background:${SRC_COLOR[sc.method]}"></span>line and frustum: its trajectory; green: ground truth.
      ${held.length ? `<span class="bad">Held poses excluded from the cloud: frames ${held.join(", ")}</span>` : "No held poses."}
      <div class="def">cloud frames ${esc(sc.cloud_frames.join(", "))}; ${sc.cloud_step_px} px grid; same for both panels</div>`;
  });
  const fr = $("#dfrm"), frn = $("#dfrn"); fr.max = frn.max = N - 1;
  const setFrame = k => { k = Math.max(0, Math.min(N - 1, k)); fr.value = frn.value = k;
    views.forEach(v => { v.o.frame = k; v.draw(); });
    const t = scenes[0].frames[k].t, lines = scenes.map(sc => { const m = sc.method, P = sc.poses[m].raw[k], G = sc.poses.gt.raw[k];
      const e = Math.hypot(P[9] - G[9], P[10] - G[10], P[11] - G[11]); return `${sc.source.label}: ${(100 * e).toFixed(2)} cm, ${rotDeg(G, P).toFixed(2)} deg${sc.held[m][k] ? " (HELD: not an estimate)" : ""}`; });
    const rid = scenes[0].source.family === "reference" ? scenes[0].source.run : (scenes[1].source.family === "reference" ? scenes[1].source.run : null);
    $("#dprev").innerHTML = (rid ? `<img src="/img/rgb?id=${enc(rid)}&k=${k}" style="width:300px"><img src="/img/depth?id=${enc(rid)}&k=${k}&mode=color" style="width:300px">` : "") +
      `<pre>frame ${k}   t = ${t.toFixed(6)} s\n${lines.join("\n")}\n(error of the camera centre / orientation vs ground truth, raw)</pre>`; };
  fr.oninput = () => setFrame(+fr.value); frn.onchange = () => setFrame(+frn.value);
  const opt = (id, key) => $(id).onchange = e => views.forEach(v => { v.o[key] = e.target.checked; v.draw(); });
  opt("#dpts", "points"); opt("#dfr", "frustums"); opt("#dcon", "connectors");
  $("#dgt").onchange = e => views.forEach(v => { v.o.show.gt = e.target.checked; v.draw(); });
  window.onresize = () => views.forEach(v => v.draw());
  setFrame(k0);
}

// ---------- #/pairx/<reference run>/<scratch run>/<k>: one pair across implementations ----------
async function viewPairX(refRun, scrRun, k) {
  k = +k || 0; setNav();
  main.innerHTML = `<h1>Pair ${k}&rarr;${k + 1} across implementations</h1><p class="muted">Loading&hellip;</p>`;
  let P; try { P = await api(`/api/pair_compare?ref=${enc(refRun)}&scratch=${enc(scrRun)}&k=${k}`); } catch (e) { return fail(e); }
  const R = P.reference, X = P.scratch;
  const nav = `<div class="row"><button ${k <= 0 ? "disabled" : ""} onclick="go('#/pairx/${enc(refRun)}/${enc(scrRun)}/${k - 1}')">&larr; previous</button>
    <button onclick="go('#/pairx/${enc(refRun)}/${enc(scrRun)}/${k + 1}')">next &rarr;</button> <a href="#/methods">back to methods</a></div>`;
  const pnpErr = R.pnp_rot_err_deg !== null ? `${f(R.pnp_rot_err_deg, 2)} deg / ${f(R.pnp_trans_err_m, 4)} m` :
    (R.pnp_returned_rot_err_deg !== null ? `not used (rejected: ${esc(R.pnp_reject_reason || "")}); returned pose ${f(R.pnp_returned_rot_err_deg, 2)} deg / ${f(R.pnp_returned_trans_err_m, 3)} m` : "solver failed");
  const tbl = `<table class="wrap"><colgroup><col style="width:22%"><col style="width:30%"><col style="width:18%"><col style="width:30%"></colgroup><tr><th></th><th class="num">${srcBadge({method: "ref_pnp", label: "Reference PnP"})}</th><th class="num">${srcBadge({method: "ref_icp", label: "Reference ICP"})}</th><th class="num">${srcBadge({method: "scratch_pnp", label: "Scratch PnP"})}</th></tr>
   <tr><td>keypoints i / j</td><td class="num" colspan="2">${R.keypoints ? R.keypoints.join(" / ") : f(null)} <span class="def">(${R.keypoints ? "from the match export" : "not exported for this pair"})</span></td><td class="num">${X.available ? `${X.counts.keypoints_i} / ${X.counts.keypoints_j}` : f(null)}</td></tr>
   <tr><td>raw matches</td><td class="num" colspan="2">${f(null)} <span class="def">(not exported)</span></td><td class="num">${X.available ? X.counts.raw_matches : f(null)}</td></tr>
   <tr><td>filtered matches</td><td class="num" colspan="2">${R.counts.matches}</td><td class="num">${X.available ? X.counts.matches : f(null)}</td></tr>
   <tr><td>correspondences</td><td class="num">${R.counts.pnp_correspondences}</td><td class="num">${R.counts.icp_correspondences}</td><td class="num">${X.available ? X.counts.pnp_correspondences : f(null)}</td></tr>
   <tr><td>inliers / ratio</td><td class="num">${R.counts.pnp_inliers} / ${(100 * R.counts.pnp_inliers / Math.max(1, R.counts.pnp_correspondences)).toFixed(1)} %</td><td class="num">${R.counts.icp_inliers} / ${(100 * R.counts.icp_inliers / Math.max(1, R.counts.icp_correspondences)).toFixed(1)} %</td><td class="num">${X.available ? `${X.counts.pnp_inliers} / ${(100 * X.counts.pnp_inliers / Math.max(1, X.counts.pnp_correspondences)).toFixed(1)} %` : f(null)}</td></tr>
   <tr><td>distinct frame-j keypoints among PnP inliers</td><td class="num">${R.distinct_kp_j_among_inliers ?? f(null)}</td><td class="num">&ndash;</td><td class="num">${X.available ? X.counts.distinct_kp_j_among_inliers : f(null)}</td></tr>
   <tr><td>status</td><td class="num">${R.pnp_solver_returned === undefined || R.pnp_solver_returned === null ? "returned" : (R.pnp_solver_returned === "1" ? "solver returned" : "solver FAILED")}${R.pnp_accepted === "0" ? ", <span class=bad>REJECTED</span>" : ""}</td><td class="num">${R.icp_rot_err_deg !== null ? "solved" : "failed"}</td><td class="num">${X.available ? (X.solver_ok ? "solved" : "<span class=bad>FAILED</span> (pose held)") : f(null)}</td></tr>
   <tr><td>error of T_j&lt;-i vs ground truth</td><td class="num">${pnpErr}</td><td class="num">${R.icp_rot_err_deg !== null ? `${f(R.icp_rot_err_deg, 2)} deg / ${f(R.icp_trans_err_m, 4)} m` : "failed"}</td><td class="num">${X.available && X.solver_ok ? `${f(X.rot_err_deg, 2)} deg / ${f(X.trans_err_m, 4)} m` : f(null)}</td></tr>
   <tr><td>true motion of the pair</td><td class="num" colspan="3">${f(R.gt_motion_deg, 2)} deg, ${f(100 * R.gt_motion_m, 2)} cm (ground truth)</td></tr>
   <tr><td>source file</td><td class="def" colspan="2">${esc(R.source_file)}${R.available ? "; matches: " + esc(R.export.dir) : ""}</td><td class="def">${X.available ? esc(X.source_file) : esc(X.reason || "")}</td></tr></table>`;
  main.innerHTML = `<h1>Pair ${k}&rarr;${k + 1} across implementations <span class="muted">(${esc(refRun)} | ${esc(scrRun)})</span></h1>${nav}${tbl}
   <div class="row"><label><input type="checkbox" id="xl" checked> correspondence lines</label><label><input type="checkbox" id="xk" checked> keypoints</label><label>max lines per class <input type="number" id="xm" value="80" min="1"></label></div>
   <div class="grid2"><div><h3>${srcBadge({method: "ref_pnp", label: "Reference PnP"})} matches and the solver's classes</h3>${R.available ? `<canvas id="xc0" style="width:100%"></canvas>` : `<div class="missing">No match-level export of the reference run for this pair. Opt-in export:<pre>${esc(R.export.command || "")}</pre></div>`}</div>
   <div><h3>${srcBadge({method: "scratch_pnp", label: "Scratch PnP"})} matches and the solver's classes</h3>${X.available ? `<canvas id="xc1" style="width:100%"></canvas>` : `<div class="missing">${esc(X.reason)}</div>`}</div></div>
   <p class="def">Blue = PnP inlier, red = outlier, grey dashed = no frame-i depth (not a PnP input), black dashed = unclassified (solver failed). Each implementation's own matches and inlier flags; a circle marks a frame-j keypoint carrying at least max(3, 20 %) of the inliers. ${esc(X.classification || "")}</p>`;
  const draw = () => [[R, "xc0"], [X, "xc1"]].forEach(([d, id]) => { if (d.available && $("#" + id)) drawMatches($("#" + id), refRun, d, {lines: $("#xl").checked, kp: $("#xk").checked, maxl: +$("#xm").value}); });
  ["#xl", "#xk", "#xm"].forEach(s => $(s).onchange = draw);
  draw();
}

function drawMatches(cv, imgRun, D, o) {
  const W = 640, H = 480, G = 12, top = 20; cv.width = 2 * W + G; cv.height = H + top;
  const g = cv.getContext("2d"); g.fillStyle = "#fff"; g.fillRect(0, 0, cv.width, cv.height);
  const redraw = () => drawMatches(cv, imgRun, D, o);
  const a = img(`/img/rgb?id=${enc(imgRun)}&k=${D.i}`, redraw), b = img(`/img/rgb?id=${enc(imgRun)}&k=${D.j}`, redraw);
  if (a) g.drawImage(a, 0, top); if (b) g.drawImage(b, W + G, top);
  g.font = "bold 13px system-ui"; g.fillStyle = "#111";
  g.fillText(`frame ${D.i}  t = ${D.t_i.toFixed(6)}`, 4, 15); g.fillText(`frame ${D.j}  t = ${D.t_j.toFixed(6)}`, W + G + 4, 15);
  if (o.kp) { g.strokeStyle = "rgba(255,212,0,0.9)"; g.lineWidth = 1; for (const [kp, ox] of [[D.kp_i, 0], [D.kp_j, W + G]]) for (let n = 0; n < kp.length; n += 2) { g.beginPath(); g.arc(kp[n] + ox, kp[n + 1] + top, 2.3, 0, 6.3); g.stroke(); } }
  const by = {I: [], O: [], D: [], U: []}; for (let r = 0; r < D.m.cls.length; r++) by[D.m.cls[r]].push(r);
  for (const c of ["D", "U", "O", "I"]) { const rows = sampleRows(by[c], o.maxl); g.strokeStyle = g.fillStyle = CLASS[c][1];
    g.lineWidth = c === "I" ? 1.8 : 1; g.globalAlpha = c === "I" ? 0.95 : 0.55; g.setLineDash(c === "D" || c === "U" ? [3, 3] : []);
    for (const r of rows) { const x0 = D.m.ui[r], y0 = D.m.vi[r] + top, x1 = D.m.uj[r] + W + G, y1 = D.m.vj[r] + top;
      if (o.lines) { g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); } g.fillRect(x0 - 2, y0 - 2, 4, 4); g.fillRect(x1 - 2, y1 - 2, 4, 4); }
    g.setLineDash([]); g.globalAlpha = 1; }
  const cnt = {}; by.I.forEach(r => cnt[D.m.kpj[r]] = (cnt[D.m.kpj[r]] || 0) + 1);
  for (const kj in cnt) if (cnt[kj] >= Math.max(3, 0.2 * by.I.length) && D.kp_j.length > 2 * kj) {
    const x = D.kp_j[2 * kj] + W + G, y = D.kp_j[2 * kj + 1] + top; g.strokeStyle = "#0072B2"; g.lineWidth = 3; g.beginPath(); g.arc(x, y, 15, 0, 6.3); g.stroke();
    g.font = "bold 12px system-ui"; g.fillStyle = "#0072B2"; g.fillText(`${cnt[kj]} inliers -> 1 keypoint`, Math.min(x + 18, cv.width - 170), y - 18); }
  g.fillStyle = "#111"; g.font = "12px system-ui";
  g.fillText(`drawn per class (every k-th): inliers ${sampleRows(by.I, o.maxl).length}/${by.I.length}, outliers ${sampleRows(by.O, o.maxl).length}/${by.O.length}, no depth ${by.D.length}, unclassified ${by.U.length}`, W + G + 4, cv.height - 4);
}

// ---------- #/corr[/<orb run>/<scratch run>/<k>]: OpenCV ORB vs scratch PnP correspondences ----------
async function viewCorr(orbRun, scrRun, k) {
  setNav();
  main.innerHTML = `<h1>Correspondences: OpenCV ORB vs Scratch PnP</h1><p class="muted">Loading&hellip;</p>`;
  let I; try { I = await api("/api/corr_index"); } catch (e) { return fail(e); }
  if (!I.length) { main.innerHTML = `<h1>Correspondences: OpenCV ORB vs Scratch PnP</h1><div class="missing">No OpenCV ORB correspondence run shares its frames with a scratch run. Create one with:<pre>build/tum_orb_correspondences data/&lt;sequence&gt; --frames-csv &lt;run&gt;/frames.csv --out results/runs/orb_tum/&lt;sequence&gt;/&lt;name&gt;</pre></div>`; return; }
  const cur = I.find(x => x.orb_run === orbRun && x.scratch_run === scrRun) || I.find(x => x.orb_run.endsWith("n100_stride4")) || I[0];
  orbRun = cur.orb_run; scrRun = cur.scratch_run; k = k === undefined ? (+Object.keys(cur.notable)[0] || 0) : +k;
  let C; try { C = await api(`/api/corr?orb=${enc(orbRun)}&scratch=${enc(scrRun)}&k=${k}`); } catch (e) { return fail(e); }
  const F = C.frames, go2 = kk => go(`#/corr/${enc(orbRun)}/${enc(scrRun)}/${kk}`);
  const stat = (x) => x.available ? `<table><tr><td>frames</td><td>${x.i} &rarr; ${x.j}</td></tr><tr><td>timestamps</td><td>${x.t_i.toFixed(6)} &rarr; ${x.t_j.toFixed(6)} s</td></tr>
    <tr><td>keypoints i / j</td><td>${x.counts.keypoints_i} / ${x.counts.keypoints_j}</td></tr>
    <tr><td>raw matches</td><td>${x.counts.raw_matches ?? f(null)}</td></tr>
    <tr><td>total (filtered) matches</td><td>${x.drawn_counts.matches}</td></tr>
    <tr><td>PnP inliers</td><td>${x.drawn_counts.inliers}</td></tr><tr><td>PnP outliers</td><td>${x.drawn_counts.outliers}</td></tr>
    <tr><td>not classified</td><td>${x.drawn_counts.no_depth} without frame-i depth${x.drawn_counts.unclassified ? `, ${x.drawn_counts.unclassified} (solver failed)` : ""}</td></tr>
    <tr><td>PnP T_j&lt;-i error vs GT</td><td>${x.solver_ok ? `${f(x.rot_err_deg, 2)} deg / ${f(x.trans_err_m, 4)} m` : `<span class="bad">solver failed</span>`}</td></tr>
    <tr><td>source</td><td class="def">${esc(x.source_file)}</td></tr></table>` : `<div class="missing">Unavailable: ${esc(x.reason || "")}${x.command ? `<pre>${esc(x.command)}</pre>` : ""}</div>`;
  const notable = Object.entries(cur.notable);
  main.innerHTML = `<h1>Correspondences: OpenCV ORB vs Scratch PnP</h1>
  <p>Same frame pair in both panels: <b>frame ${F.frame_i} (t = ${F.t_rgb_i.toFixed(6)} s, <code>${esc(F.rgb_i)}</code>) &rarr; frame ${F.frame_j} (t = ${F.t_rgb_j.toFixed(6)} s, <code>${esc(F.rgb_j)}</code>)</b> of ${esc(I.length > 1 ? "" : "")}<code>${esc(orbRun)}</code> / <code>${esc(scrRun)}</code> (identical frames.csv ${esc(C.frames_key)}).</p>
  <div class="row"><button ${k <= 0 ? "disabled" : ""} id="cprev">&larr; previous</button><label>pair <input type="number" id="cpk" min="0" max="${C.n_pairs - 1}" value="${k}" style="width:5em"></label> <span class="muted">of 0&ndash;${C.n_pairs - 1}</span><button ${k >= C.n_pairs - 1 ? "disabled" : ""} id="cnext">next &rarr;</button>
   <label>run <select id="crun">${I.map(x => `<option value="${esc(x.orb_run)}|${esc(x.scratch_run)}" ${x === cur ? "selected" : ""}>${esc(x.orb_run)} | ${esc(x.scratch_run)}</option>`).join("")}</select></label></div>
  <div class="row"><b>Important pairs:</b> ${notable.map(([kk, why]) => `<a href="#/corr/${enc(orbRun)}/${enc(scrRun)}/${kk}" title="${esc(why.join("; "))}">${kk}&rarr;${+kk + 1}</a> <span class="muted">(${esc(why.join("; "))})</span>`).join(" &middot; ")}</div>
  <div class="row"><label><input type="checkbox" id="cl" checked> correspondence lines</label><label><input type="checkbox" id="ck" checked> keypoints</label><label>max lines per class <input type="number" id="cm" value="80" min="1"></label></div>
  <div class="grid2">
   <div><h2><span class="sw" style="background:#e15759"></span>OpenCV ORB correspondences</h2><p class="def">${esc(C.labels.orb[1])}</p>${C.orb.available ? `<canvas id="cc0" style="width:100%"></canvas>` : ""}${stat(C.orb)}</div>
   <div><h2><span class="sw" style="background:#17becf"></span>Scratch PnP correspondences</h2><p class="def">${esc(C.labels.scratch[1])}. ${esc(C.scratch.classification || "")}</p>${C.scratch.available ? `<canvas id="cc1" style="width:100%"></canvas>` : ""}${stat(C.scratch)}</div>
  </div>
  <p class="def">Blue = PnP inlier, red = PnP outlier, grey dashed = no frame-i depth (not a PnP input), black dashed = unclassified (solver failed). Lines are every k-th match per class; counts above are over all matches. Each method's own keypoints, matches and inlier flags; the reference pipeline (FAST + rotated BRIEF, not ORB) is compared on the Methods page.</p>`;
  $("#cprev").onclick = () => go2(k - 1); $("#cnext").onclick = () => go2(k + 1); $("#cpk").onchange = e => go2(+e.target.value);
  $("#crun").onchange = e => { const [a, b] = e.target.value.split("|"); go(`#/corr/${enc(a)}/${enc(b)}/${k}`); };
  const imgRun = C.image_run;
  const draw = () => [[C.orb, "cc0"], [C.scratch, "cc1"]].forEach(([d, id]) => { if (d.available && $("#" + id)) drawMatches($("#" + id), imgRun, d, {lines: $("#cl").checked, kp: $("#ck").checked, maxl: +$("#cm").value}); });
  ["#cl", "#ck", "#cm"].forEach(s => $(s).onchange = draw);
  draw();
}

routes.corr = viewCorr;
routes.methods = viewMethods;
routes.duo = viewDuo;
routes.pairx = viewPairX;
