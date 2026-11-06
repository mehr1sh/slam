"""The interactive viewer page (offline HTML + one data file; no external libraries).

write(out_dir, data) writes out_dir/index.html and out_dir/data.js (window.SLAM_DATA = {...}); RGB frames
are expected in out_dir/frames/ (written by build_viewer.py). Data schema: see build_viewer.py.
The page has two views:
  correspondences  frame i | frame j of the selected exported pair, keypoints and matches coloured by
                   the PnP class the solver returned (deterministic sampling: every k-th per class)
  3D               "scene": sampled depth points placed with GT / PnP / ICP poses (TUM world frame, z up),
                   trajectories, frustums, GT->estimate connectors; click a camera to inspect it.
                   "registration": the selected pair in camera i's frame (x right, y down, z forward):
                   target = frame i, source = frame j under identity / ICP / GT, actual ICP correspondences.
URL fragment keeps the state, e.g. #pair=73&mode=registration&state=icp&frame=74.
"""

import base64
import json
from pathlib import Path

import numpy as np

COLORS = {"gt": "#2ca02c", "pnp": "#ff7f0e", "icp": "#9467bd"}
LABELS = {"gt": "ground truth", "pnp": "PnP", "icp": "ICP"}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def pack_cloud(xyz, rgb):
    mm = np.clip(np.round(xyz * 1000), -32767, 32767).astype("<i2")
    return {"n": int(len(mm)), "xyz": b64(mm), "rgb": b64(np.asarray(rgb, np.uint8))}


def pose_list(R, p):
    return [[round(float(x), 7) for x in list(np.asarray(R[k]).reshape(9)) + list(np.asarray(p[k]))]
            for k in range(len(p))]


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLAM viewer</title>
<style>
:root{--fg:#111;--muted:#555;--line:#d0d0d0;--bg:#fff;--panel:#f5f5f5}
body{margin:0;font:13px system-ui,sans-serif;color:var(--fg);background:var(--bg)}
header{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding:8px 16px;border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0}
section{padding:8px 16px}
h2{font-size:14px;margin:4px 0 6px}
.row{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;margin-bottom:6px}
.muted{color:var(--muted)}
canvas{display:block;max-width:100%}
#c2d{width:100%;height:auto;background:#fff;border:1px solid var(--line)}
#wrap3d{display:flex;flex-wrap:wrap;gap:12px}
#c3d{flex:1 1 640px;min-width:0;height:68vh;min-height:380px;border:1px solid var(--line);touch-action:none;cursor:grab}
#side{flex:0 0 340px;max-width:100%}
#side img{width:100%;border:1px solid var(--line)}
pre{font:12px ui-monospace,monospace;white-space:pre-wrap;margin:4px 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:3px}
label{white-space:nowrap}
a{color:#0b62b5}
</style></head><body>
<header><h1 id="title"></h1>
<label>pair <select id="pair"></select></label> <span id="detail"></span> <a id="home" href="../../../index.html">all sequences</a></header>
<section>
<h2>Correspondences <span class="muted" id="pairinfo"></span></h2>
<div class="row">
<label><input type="checkbox" id="sh_inlier" checked> <span class="sw" style="background:#0072B2"></span>PnP inliers</label>
<label><input type="checkbox" id="sh_outlier" checked> <span class="sw" style="background:#CC3311"></span>PnP outliers</label>
<label><input type="checkbox" id="sh_nodepth" checked> <span class="sw" style="background:#7f7f7f"></span>no depth (not given to PnP)</label>
<label><input type="checkbox" id="sh_kp" checked> <span class="sw" style="background:#ffd400"></span>all keypoints</label>
<label>max lines per class <input type="number" id="maxl" value="80" min="1" style="width:4.5em"></label>
<span class="muted" id="drawn"></span></div>
<canvas id="c2d"></canvas>
</section>
<section>
<h2>3D <span class="muted" id="info3d"></span></h2>
<div class="row">
<label><input type="radio" name="mode" value="scene" checked> scene (TUM world, z up)</label>
<label><input type="radio" name="mode" value="registration"> registration of the selected pair (camera i frame)</label>
</div>
<div class="row" id="rowScene">
<span id="tog"></span>
<label><input type="checkbox" id="pts" checked> point cloud placed with <select id="place"><option value="gt">GT</option><option value="pnp">PnP</option><option value="icp">ICP</option></select> poses</label>
<label><input type="checkbox" id="conn" checked> GT&rarr;estimate connectors every <input type="number" id="every" value="5" min="1" style="width:3em"> frames</label>
<label><select id="align"><option value="raw">raw estimates</option><option value="aligned">aligned (one global SE(3) per method)</option></select></label>
</div>
<div class="row" id="rowReg" style="display:none">
<label><input type="radio" name="state" value="before" checked> before (identity)</label>
<label><input type="radio" name="state" value="icp"> after ICP</label>
<label><input type="radio" name="state" value="gt"> after ground truth</label>
<label><input type="checkbox" id="rpts" checked> point clouds</label>
<label><input type="checkbox" id="rcorr" checked> ICP correspondences</label>
<label><input type="checkbox" id="rcams" checked> camera j poses (GT / PnP / ICP)</label>
</div>
<div id="wrap3d"><canvas id="c3d"></canvas>
<div id="side"><div class="row"><label>frame <input type="range" id="frame" min="0" max="0" value="0" style="width:200px"></label></div>
<img id="thumb" alt="RGB frame of the selected camera"><pre id="pose"></pre><div id="legend3d" class="muted"></div></div></div>
<p class="muted" id="notes"></p>
</section>
<script src="data.js"></script>
<script>
"use strict";
const D = window.SLAM_DATA, COL = __COLORS__, LAB = __LABELS__;
const N = D.frames.length, METH = ["gt", "pnp", "icp"];
const CLS = {I: ["inlier", "#0072B2"], O: ["outlier", "#CC3311"], D: ["no_depth_i", "#7f7f7f"], U: ["unclassified", "#000000"]};
const S = {pair: D.defaults.pair, frame: D.defaults.frame, mode: "scene", state: "before", place: "gt", align: "raw",
  pts: true, conn: true, every: 5, rpts: true, rcorr: true, rcams: true, show: {gt: true, pnp: true, icp: true},
  yaw: -60, pitch: 28, dist: 0, tx: 0, ty: 0, tz: 0, maxl: 80, sh: {inlier: true, outlier: true, nodepth: true, kp: true}};
const VIEWKEYS = ["yaw", "pitch", "dist", "tx", "ty", "tz"];
for (const kv of location.hash.slice(1).split("&")) { const [k, v] = kv.split("="); if (!k || v === undefined) continue;
  if (k.startsWith("show_")) S.show[k.slice(5)] = v === "1"; else if (["pts", "conn", "rpts", "rcorr", "rcams"].includes(k)) S[k] = v === "1";
  else if (typeof S[k] === "number") S[k] = parseFloat(v); else S[k] = decodeURIComponent(v); }
const $ = id => document.getElementById(id);
function dec(b, T) { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); }
function cloud(c) { return {xyz: dec(c.xyz, Int16Array), rgb: dec(c.rgb, Uint8Array), n: c.n}; }
const sceneClouds = D.scene_clouds.map(c => Object.assign(cloud(c), {k: c.k}));
const pairClouds = {}; for (const k in D.pair_clouds) pairClouds[k] = cloud(D.pair_clouds[k]);
const pairs = {}; D.pairs.forEach(p => pairs[p.pair] = p);
if (!pairs[S.pair]) S.pair = D.pairs.length ? D.pairs[0].pair : null;
if (S.pair !== null && !/(^|[#&])frame=/.test(location.hash)) S.frame = pairs[S.pair].j;  // frame follows the pair
$("title").textContent = D.run.title;
$("notes").textContent = D.notes.join("  ");
// ---------- pair selector and correspondence view ----------
const sel = $("pair");
D.pairs.forEach(p => { const o = document.createElement("option"); o.value = p.pair;
  o.textContent = `${p.i} -> ${p.j}  (PnP inliers ${p.counts.pnp_inliers}/${p.counts.pnp_correspondences})${p.counts.pnp_inliers < 20 ? "  !" : ""}`; sel.appendChild(o); });
sel.value = S.pair; sel.onchange = () => { S.pair = +sel.value; S.frame = pairs[S.pair].j; $("frame").value = S.frame; loadPair(); draw3d(); };
const imgs = {};
function image(k) { if (!imgs[k]) { imgs[k] = new Image(); imgs[k].onload = () => { drawPair(); updateSide(); }; imgs[k].src = D.frames[k].img; } return imgs[k]; }
function loadPair() { const p = pairs[S.pair]; if (!p) return; image(p.i); image(p.j); drawPair();
  $("detail").innerHTML = p.detail ? `<a href="${p.detail}">detailed diagnostic of ${p.i} -> ${p.j}</a>` : "";
  const c = p.counts, r = c.pnp_correspondences ? (100 * c.pnp_inliers / c.pnp_correspondences).toFixed(1) : "-";
  $("pairinfo").textContent = `frame ${p.i} (t = ${p.t_i.toFixed(6)}) -> ${p.j} (t = ${p.t_j.toFixed(6)}): keypoints ${c.keypoints_i} / ${c.keypoints_j}, matches ${c.matches}, PnP inliers ${c.pnp_inliers}/${c.pnp_correspondences} (${r} %), ICP inliers ${c.icp_inliers}/${c.icp_correspondences}; error vs GT: PnP ${fmt(p.err.pnp_rot_err_deg, 2)} deg / ${fmt(p.err.pnp_trans_err_m, 3)} m, ICP ${fmt(p.err.icp_rot_err_deg, 2)} deg / ${fmt(p.err.icp_trans_err_m, 3)} m`; }
function fmt(v, d) { return v === null || v === undefined ? "n/a" : (+v).toFixed(d); }
function sample(rows, cap) { if (rows.length <= cap) return rows; const k = Math.ceil(rows.length / cap), o = []; for (let i = 0; i < rows.length; i += k) o.push(rows[i]); return o; }
function drawPair() {
  const p = pairs[S.pair], cv = $("c2d"); if (!p) return;
  const W = D.K.width, H = D.K.height, G = 16, top = 22; cv.width = 2 * W + G; cv.height = H + top;
  const ctx = cv.getContext("2d"); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, cv.width, cv.height);
  const a = image(p.i), b = image(p.j);
  if (a.complete && a.naturalWidth) ctx.drawImage(a, 0, top); if (b.complete && b.naturalWidth) ctx.drawImage(b, W + G, top);
  ctx.font = "bold 14px sans-serif"; ctx.fillStyle = "#111";
  ctx.fillText(`frame ${p.i}  t = ${p.t_i.toFixed(6)} s  (${p.counts.keypoints_i} keypoints)`, 4, 16);
  ctx.fillText(`frame ${p.j}  t = ${p.t_j.toFixed(6)} s  (${p.counts.keypoints_j} keypoints)`, W + G + 4, 16);
  if (S.sh.kp) { ctx.strokeStyle = "rgba(255,212,0,0.9)"; ctx.lineWidth = 1;
    for (const [kp, ox] of [[p.kp_i, 0], [p.kp_j, W + G]]) for (let n = 0; n < kp.length; n += 2) { ctx.beginPath(); ctx.arc(kp[n] + ox, kp[n + 1] + top, 2.5, 0, 6.3); ctx.stroke(); } }
  const by = {I: [], O: [], D: [], U: []}; for (let r = 0; r < p.m.cls.length; r++) by[p.m.cls[r]].push(r);
  const drawn = []; const want = {I: S.sh.inlier, O: S.sh.outlier, D: S.sh.nodepth, U: S.sh.outlier};
  for (const c of ["D", "U", "O", "I"]) { if (!want[c] || !by[c].length) continue; const rows = sample(by[c], S.maxl);
    drawn.push(`${CLS[c][0]} ${rows.length}/${by[c].length}`);
    ctx.strokeStyle = CLS[c][1]; ctx.fillStyle = CLS[c][1]; ctx.lineWidth = c === "I" ? 1.6 : 1; ctx.globalAlpha = c === "I" ? 0.95 : 0.6;
    ctx.setLineDash(c === "D" ? [3, 3] : []);
    for (const r of rows) { const x0 = p.m.ui[r], y0 = p.m.vi[r] + top, x1 = p.m.uj[r] + W + G, y1 = p.m.vj[r] + top;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.fillRect(x0 - 2, y0 - 2, 4, 4); ctx.fillRect(x1 - 2, y1 - 2, 4, 4); }
    ctx.setLineDash([]); ctx.globalAlpha = 1; }
  $("drawn").textContent = "drawn (every k-th row per class): " + drawn.join(", ");
}
["inlier", "outlier", "nodepth", "kp"].forEach(k => { const e = $("sh_" + k); e.checked = S.sh[k]; e.onchange = () => { S.sh[k] = e.checked; drawPair(); }; });
$("maxl").value = S.maxl; $("maxl").onchange = () => { S.maxl = Math.max(1, +$("maxl").value); drawPair(); };
// ---------- 3D ----------
const cv3 = $("c3d"), g3 = cv3.getContext("2d");
function pose(m, k) { const P = (S.align === "aligned" && D.poses[m].aligned) ? D.poses[m].aligned : D.poses[m].raw; return P[k]; }
function xf(T, x, y, z) { return [T[0]*x + T[1]*y + T[2]*z + T[9], T[3]*x + T[4]*y + T[5]*z + T[10], T[6]*x + T[7]*y + T[8]*z + T[11]]; }
function m4(T) { return [T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2], T[2][0], T[2][1], T[2][2], T[0][3], T[1][3], T[2][3]]; }
function inv(T) { const R = [T[0], T[3], T[6], T[1], T[4], T[7], T[2], T[5], T[8]], t = [T[9], T[10], T[11]];
  return R.concat([-(R[0]*t[0] + R[1]*t[1] + R[2]*t[2]), -(R[3]*t[0] + R[4]*t[1] + R[5]*t[2]), -(R[6]*t[0] + R[7]*t[1] + R[8]*t[2])]); }
function ramp(t) { const s = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]]; t = Math.min(1, Math.max(0, t)) * 4;
  const i = Math.min(3, Math.floor(t)), f = t - i; return s[i].map((v, a) => Math.round(v + f * (s[i + 1][a] - v))); }
let W = 0, H = 0, F = 0, eye, R3, U3, F3, UP = [0, 0, 1];
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function nrm(a) { const n = Math.hypot(...a) || 1; return a.map(x => x / n); }
const views = {};
function defaultView(mode) {
  if (mode === "registration") { const p = pairs[S.pair], c = pairClouds[p.i]; let s = [0, 0, 0], n = 0;
    for (let i = 0; i < c.xyz.length; i += 3) { s[0] += c.xyz[i]; s[1] += c.xyz[i + 1]; s[2] += c.xyz[i + 2]; n++; }
    // look from behind and above camera i at a point between it and the cloud, so camera i's frustum stays in view
    return {yaw: -100, pitch: 22, dist: 1.9, tx: 0.6 * s[0] / n / 1000, ty: 0.6 * s[1] / n / 1000, tz: 0.6 * s[2] / n / 1000}; }
  return Object.assign({}, D.defaults.scene_view); }
function camSetup() {
  const r = cv3.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  W = Math.max(1, Math.round(r.width * dpr)); H = Math.max(1, Math.round(r.height * dpr));
  if (cv3.width !== W || cv3.height !== H) { cv3.width = W; cv3.height = H; }
  F = H / (2 * Math.tan(25 * Math.PI / 180));
  UP = S.mode === "registration" ? [0, -1, 0] : [0, 0, 1];
  const y = S.yaw * Math.PI / 180, p = S.pitch * Math.PI / 180;
  // orbit around the target in a frame whose third axis is UP
  const e1 = S.mode === "registration" ? [1, 0, 0] : [1, 0, 0], e2 = cross(UP, e1);
  const d = [0, 1, 2].map(a => S.dist * (Math.cos(p) * Math.cos(y) * e1[a] + Math.cos(p) * Math.sin(y) * e2[a] + Math.sin(p) * UP[a]));
  eye = [S.tx + d[0], S.ty + d[1], S.tz + d[2]];
  F3 = nrm([S.tx - eye[0], S.ty - eye[1], S.tz - eye[2]]); R3 = nrm(cross(F3, UP)); U3 = cross(R3, F3);
}
function proj(x, y, z) { const a = x - eye[0], b = y - eye[1], c = z - eye[2], zc = a*F3[0] + b*F3[1] + c*F3[2];
  if (zc < 0.02) return null; return [W / 2 + F * (a*R3[0] + b*R3[1] + c*R3[2]) / zc, H / 2 - F * (a*U3[0] + b*U3[1] + c*U3[2]) / zc, zc]; }
let buf = null, zbuf = null;
function beginPoints() { buf = g3.createImageData(W, H); zbuf = new Float32Array(W * H).fill(Infinity);
  const d = buf.data; for (let i = 0; i < d.length; i += 4) { d[i] = d[i + 1] = d[i + 2] = 255; d[i + 3] = 255; } }
function point(x, y, z, r, g, b, size) { const p = proj(x, y, z); if (!p) return;
  const u0 = Math.round(p[0]), v0 = Math.round(p[1]), h = size >> 1;
  for (let v = v0 - h; v <= v0 + h; v++) { if (v < 0 || v >= H) continue; for (let u = u0 - h; u <= u0 + h; u++) { if (u < 0 || u >= W) continue;
    const i = v * W + u; if (p[2] < zbuf[i]) { zbuf[i] = p[2]; const o = 4 * i; buf.data[o] = r; buf.data[o + 1] = g; buf.data[o + 2] = b; } } } }
function line(a, b, col, w, dash) { const p = proj(...a), q = proj(...b); if (!p || !q) return;
  g3.strokeStyle = col; g3.lineWidth = w; g3.setLineDash(dash || []); g3.beginPath(); g3.moveTo(p[0], p[1]); g3.lineTo(q[0], q[1]); g3.stroke(); g3.setLineDash([]); }
function frustum(T, s, col, w) { const K = D.K, cs = [[0,0],[K.width,0],[K.width,K.height],[0,K.height]], c = [T[9], T[10], T[11]];
  const q = cs.map(([u, v]) => xf(T, (u - K.cx) / K.fx * s, (v - K.cy) / K.fy * s, s));
  q.forEach((p, i) => { line(c, p, col, w); line(p, q[(i + 1) % 4], col, w); });
  const ax = [[1,0,0,"#d62728"],[0,1,0,"#17becf"],[0,0,1,"#1f77b4"]]; ax.forEach(([x, y, z, cc]) => line(c, xf(T, x * s * 0.6, y * s * 0.6, z * s * 0.6), cc, w)); }
function marker(P, col, shape) { const p = proj(...P); if (!p) return; const dpr = window.devicePixelRatio || 1;
  g3.fillStyle = col; g3.strokeStyle = "#000"; g3.lineWidth = dpr; g3.beginPath();
  if (shape === "o") g3.arc(p[0], p[1], 5 * dpr, 0, 6.3); else if (shape === "s") g3.rect(p[0] - 5*dpr, p[1] - 5*dpr, 10*dpr, 10*dpr);
  else { g3.moveTo(p[0], p[1] - 7*dpr); g3.lineTo(p[0] + 6*dpr, p[1] + 5*dpr); g3.lineTo(p[0] - 6*dpr, p[1] + 5*dpr); g3.closePath(); }
  g3.fill(); g3.stroke(); }
function errAt(m, k) { const a = pose(m, k), g = D.poses.gt.raw[k]; return Math.hypot(a[9]-g[9], a[10]-g[10], a[11]-g[11]); }
let errMax = 1e-9; ["pnp", "icp"].forEach(m => { for (let k = 0; k < N; k++) errMax = Math.max(errMax, errAt(m, k)); });
function drawScene(dpr) {
  if (S.pts) { beginPoints();
    for (const c of sceneClouds) { if (S.place !== "gt" && D.held[S.place][c.k]) continue;  // held pose: no estimate of its own
      const T = pose(S.place, c.k); for (let i = 0; i < c.xyz.length; i += 3) {
      const w = xf(T, c.xyz[i] / 1000, c.xyz[i + 1] / 1000, c.xyz[i + 2] / 1000); point(w[0], w[1], w[2], c.rgb[i], c.rgb[i + 1], c.rgb[i + 2], 2 * dpr); } }
    g3.putImageData(buf, 0, 0); }
  const k0 = S.frame;
  if (S.conn) for (const m of ["pnp", "icp"]) { if (!S.show[m]) continue;
    for (let k = 0; k < N; k++) { if (k % S.every && k !== k0) continue; const a = pose(m, k), g = D.poses.gt.raw[k], c = ramp(Math.log(Math.max(errAt(m, k), 0.01) / 0.01) / Math.log(Math.max(errMax, 0.02) / 0.01));
      line([g[9], g[10], g[11]], [a[9], a[10], a[11]], `rgb(${c})`, 1.6 * dpr); } }
  for (const m of METH) { if (!S.show[m]) continue; g3.strokeStyle = COL[m]; g3.lineWidth = (m === "gt" ? 2.6 : 2) * dpr; g3.beginPath(); let started = false;
    for (let k = 0; k < N; k++) { const T = pose(m, k), p = proj(T[9], T[10], T[11]); if (!p) { started = false; continue; }
      if (started) g3.lineTo(p[0], p[1]); else g3.moveTo(p[0], p[1]); started = true; } g3.stroke();
    frustum(pose(m, k0), 0.07, COL[m], 1.8 * dpr);
    marker([pose(m, 0)[9], pose(m, 0)[10], pose(m, 0)[11]], COL[m], "o"); marker([pose(m, N - 1)[9], pose(m, N - 1)[10], pose(m, N - 1)[11]], COL[m], "s"); }
  $("legend3d").innerHTML = `circle = frame 0, square = frame ${N - 1}; frustums at the selected frame with camera axes (x red, y cyan, z blue = optical axis). ` +
    `Connectors GT&rarr;estimate (same frame) coloured by |error|, log scale: &le;1 cm <span style="display:inline-block;width:90px;height:9px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)"></span> ${(100 * errMax).toFixed(1)} cm (${S.align}).`;
  $("info3d").textContent = `world = TUM ground-truth frame, metres; ${sceneClouds.length} sampled frames of depth placed with ${LAB[S.place]} poses (${S.align})`;
}
function drawRegistration(dpr) {
  const p = pairs[S.pair]; if (!p) return;
  const T = S.state === "icp" ? (p.T_icp_ij ? m4(p.T_icp_ij) : null) : S.state === "gt" ? inv(m4(p.T_gt_ji)) : [1,0,0,0,1,0,0,0,1,0,0,0];
  if (S.rpts) { beginPoints(); const tg = pairClouds[p.i], sr = pairClouds[p.j];
    for (let i = 0; i < tg.xyz.length; i += 3) { const l = (tg.rgb[i] + tg.rgb[i + 1] + tg.rgb[i + 2]) / 765;
      point(tg.xyz[i] / 1000, tg.xyz[i + 1] / 1000, tg.xyz[i + 2] / 1000, 51 * (0.35 + 0.75 * l), 115 * (0.35 + 0.75 * l), 230 * (0.35 + 0.75 * l), 2 * dpr); }
    if (T) for (let i = 0; i < sr.xyz.length; i += 3) { const l = (sr.rgb[i] + sr.rgb[i + 1] + sr.rgb[i + 2]) / 765, w = xf(T, sr.xyz[i] / 1000, sr.xyz[i + 1] / 1000, sr.xyz[i + 2] / 1000);
      point(w[0], w[1], w[2], Math.min(255, 250 * (0.35 + 0.75 * l)), 140 * (0.35 + 0.75 * l), 26 * (0.35 + 0.75 * l), 2 * dpr); }
    g3.putImageData(buf, 0, 0); }
  if (S.rcorr && T) { const c = p.icp; for (let n = 0; n < c.cls.length; n++) { if (c.cls[n] !== "I" && n % c.rej_every) continue;
    const a = xf(T, c.pj[3*n], c.pj[3*n + 1], c.pj[3*n + 2]), b = [c.pi[3*n], c.pi[3*n + 1], c.pi[3*n + 2]];
    const r = Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]), col = ramp(r / 0.05);
    line(a, b, `rgb(${col})`, (c.cls[n] === "I" ? 2.2 : 1.1) * dpr, c.cls[n] === "I" ? [] : [5, 4]); } }
  if (S.rcams) { frustum([1,0,0,0,1,0,0,0,1,0,0,0], 0.08, "#000", 1.6 * dpr);
    for (const [m, Tji] of [["gt", p.T_gt_ji], ["pnp", p.T_pnp_ji], ["icp", p.T_icp_ji]]) if (Tji) frustum(inv(m4(Tji)), 0.08, COL[m], 2 * dpr); }
  const st = p.reg[S.state];
  $("info3d").textContent = `pair ${p.i} -> ${p.j} in camera ${p.i}'s frame (x right, y down, z forward; metres): target = frame ${p.i} (blue), source = frame ${p.j} (orange) ` +
    (T ? `under ${S.state === "before" ? "the identity" : S.state === "icp" ? "ICP's T_i<-j" : "the ground-truth T_i<-j"}` : "(ICP returned no transform)");
  $("legend3d").innerHTML = `ICP correspondences ${p.counts.icp_correspondences}, inliers ${p.counts.icp_inliers} (EstimateRigidRansac, threshold ${100 * p.icp_threshold_m} cm). ` +
    `Solid = inlier, dashed = every ${p.icp.rej_every}th rejected; colour = residual |p_i - T p_j| 0 <span style="display:inline-block;width:90px;height:9px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)"></span> 5 cm. ` +
    (st ? `This state: inlier median residual ${(100 * st.inlier_median_m).toFixed(2)} cm; ${st.within_threshold} of ${p.counts.icp_correspondences} within threshold.` : "") +
    ` Frustums: camera i (black) and camera j as placed by GT / PnP / ICP.`;
}
function draw3d() { camSetup(); const dpr = window.devicePixelRatio || 1;
  g3.setTransform(1, 0, 0, 1, 0, 0); g3.fillStyle = "#fff"; g3.fillRect(0, 0, W, H);
  if (S.mode === "scene") drawScene(dpr); else drawRegistration(dpr);
  updateSide(); saveHash(); }
function updateSide() { const k = S.frame, f = D.frames[k]; $("thumb").src = f.img;
  let t = `frame ${k}   t = ${f.t.toFixed(6)} s\n`; const g = D.poses.gt.raw[k];
  for (const m of METH) { const a = pose(m, k); t += `${LAB[m].padEnd(12)} centre (${a[9].toFixed(3)}, ${a[10].toFixed(3)}, ${a[11].toFixed(3)})`;
    if (m !== "gt") { const R = [0,1,2].map(r => [0,1,2].map(c => g[3*0 + r] * a[3*0 + c] + g[3 + r] * a[3 + c] + g[6 + r] * a[6 + c]));
      const ang = Math.acos(Math.min(1, Math.max(-1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2))) * 180 / Math.PI;
      t += `\n${"".padEnd(12)} error ${(100 * errAt(m, k)).toFixed(2)} cm, ${ang.toFixed(2)} deg${D.held[m][k] ? "  (pose held: failed pair)" : ""}`; }
    t += "\n"; }
  $("pose").textContent = t + `(${S.align}; T_wc, X_world = R X_cam + t)`; }
function saveHash() { const h = ["pair=" + S.pair, "frame=" + S.frame, "mode=" + S.mode, "state=" + S.state, "place=" + S.place, "align=" + S.align,
  "pts=" + (+S.pts), "conn=" + (+S.conn), "every=" + S.every, "rpts=" + (+S.rpts), "rcorr=" + (+S.rcorr), "rcams=" + (+S.rcams)]
  .concat(VIEWKEYS.map(k => `${k}=${(+S[k]).toFixed(3)}`)).concat(METH.map(m => `show_${m}=${S.show[m] ? 1 : 0}`)).join("&");
  history.replaceState(null, "", "#" + h); }
// controls
const tog = $("tog"); METH.forEach(m => { const l = document.createElement("label");
  l.innerHTML = `<input type="checkbox" ${S.show[m] ? "checked" : ""}> <span class="sw" style="background:${COL[m]}"></span>${LAB[m]}`;
  l.querySelector("input").onchange = e => { S.show[m] = e.target.checked; draw3d(); }; tog.appendChild(l); });
const fr = $("frame"); fr.max = N - 1; fr.value = S.frame; fr.oninput = () => { S.frame = +fr.value; draw3d(); };
for (const [id, key] of [["pts", "pts"], ["conn", "conn"], ["rpts", "rpts"], ["rcorr", "rcorr"], ["rcams", "rcams"]]) { $(id).checked = S[key]; $(id).onchange = () => { S[key] = $(id).checked; draw3d(); }; }
$("place").value = S.place; $("place").onchange = () => { S.place = $("place").value; draw3d(); };
$("align").value = S.align; $("align").onchange = () => { S.align = $("align").value; draw3d(); };
$("every").value = S.every; $("every").onchange = () => { S.every = Math.max(1, +$("every").value); draw3d(); };
document.querySelectorAll("input[name=state]").forEach(r => { r.checked = r.value === S.state; r.onchange = () => { S.state = r.value; draw3d(); }; });
function setMode(m, keepView) { if (!keepView) { views[S.mode] = Object.fromEntries(VIEWKEYS.map(k => [k, S[k]])); Object.assign(S, views[m] || defaultView(m)); }
  S.mode = m; $("rowScene").style.display = m === "scene" ? "" : "none"; $("rowReg").style.display = m === "registration" ? "" : "none"; draw3d(); }
document.querySelectorAll("input[name=mode]").forEach(r => { r.checked = r.value === S.mode; r.onchange = () => setMode(r.value); });
let drag = null, moved = false;
cv3.addEventListener("pointerdown", e => { drag = {x: e.clientX, y: e.clientY, pan: e.shiftKey}; moved = false; cv3.setPointerCapture(e.pointerId); });
cv3.addEventListener("pointermove", e => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) { const s = S.dist / 700; S.tx -= (R3[0] * dx - U3[0] * dy) * s; S.ty -= (R3[1] * dx - U3[1] * dy) * s; S.tz -= (R3[2] * dx - U3[2] * dy) * s; }
  else { S.yaw -= dx * 0.4; S.pitch = Math.max(-89, Math.min(89, S.pitch + dy * 0.4)); } draw3d(); });
cv3.addEventListener("pointerup", e => { const was = drag; drag = null; if (!was || moved || S.mode !== "scene") return;
  const r = cv3.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, x = (e.clientX - r.left) * dpr, y = (e.clientY - r.top) * dpr;
  let best = null, bd = 18 * dpr; for (const m of METH) { if (!S.show[m]) continue; for (let k = 0; k < N; k++) { const T = pose(m, k), p = proj(T[9], T[10], T[11]);
    if (p) { const d = Math.hypot(p[0] - x, p[1] - y); if (d < bd) { bd = d; best = k; } } } }
  if (best !== null) { S.frame = best; fr.value = best; draw3d(); } });
cv3.addEventListener("wheel", e => { e.preventDefault(); S.dist *= Math.exp(e.deltaY * 0.001); draw3d(); }, {passive: false});
window.addEventListener("resize", draw3d);
if (!S.dist) Object.assign(S, defaultView(S.mode));
setMode(S.mode, true); loadPair();
</script></body></html>
"""


def write(out_dir, data):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text("window.SLAM_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    (out_dir / "index.html").write_text(PAGE.replace("__COLORS__", json.dumps(COLORS)).replace("__LABELS__", json.dumps(LABELS)))
