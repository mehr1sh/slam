// 3D view of the dashboard (the project's single 3D viewer). Canvas 2D + a z-buffered point buffer.
// scene mode: world = TUM ground-truth frame (z up), poses T_wc (X_world = R X_cam + t), points stored in each
//             camera's frame (mm) and placed with one method's pose of that frame.
// registration mode: camera i's frame (x right, y down, z forward); target = frame i, source = frame j
//             moved by identity / ICP's T_i<-j / ground-truth T_i<-j; ICP correspondences as exported.
"use strict";
const METHOD_COLOR = {gt: "#2ca02c", pnp: "#ff7f0e", icp: "#9467bd", ref_pnp: "#ff7f0e", ref_icp: "#9467bd", scratch_pnp: "#17becf"};
const METHOD_LABEL = {gt: "ground truth", pnp: "PnP", icp: "ICP", ref_pnp: "Reference PnP", ref_icp: "Reference ICP", scratch_pnp: "Scratch PnP"};

function b64dec(b, T) { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); }
function unpackCloud(c) { return {xyz: b64dec(c.xyz, Int16Array), rgb: b64dec(c.rgb, Uint8Array), n: c.n, k: c.k}; }
function xf(T, x, y, z) { return [T[0]*x + T[1]*y + T[2]*z + T[9], T[3]*x + T[4]*y + T[5]*z + T[10], T[6]*x + T[7]*y + T[8]*z + T[11]]; }
function m4(T) { return [T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2], T[2][0], T[2][1], T[2][2], T[0][3], T[1][3], T[2][3]]; }
function invT(T) { const R = [T[0], T[3], T[6], T[1], T[4], T[7], T[2], T[5], T[8]], t = [T[9], T[10], T[11]];
  return R.concat([-(R[0]*t[0] + R[1]*t[1] + R[2]*t[2]), -(R[3]*t[0] + R[4]*t[1] + R[5]*t[2]), -(R[6]*t[0] + R[7]*t[1] + R[8]*t[2])]); }
function ramp(t) { const s = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]]; t = Math.min(1, Math.max(0, t)) * 4;
  const i = Math.min(3, Math.floor(t)), f = t - i; return s[i].map((v, a) => Math.round(v + f * (s[i + 1][a] - v))); }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function nrm(a) { const n = Math.hypot(...a) || 1; return a.map(x => x / n); }
function rotDeg(A, B) { // angle of A^T B, both row-major 3x3 in the first 9 entries
  let tr = 0; for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) tr += A[3*k + r] * B[3*k + r];
  return Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2))) * 180 / Math.PI; }

class View3D {
  constructor(canvas, onPick) {
    this.cv = canvas; this.g = canvas.getContext("2d"); this.onPick = onPick;
    this.mode = "scene"; this.scene = null; this.reg = null; this.views = {};
    this.o = {show: {gt: true, pnp: true, icp: true}, points: true, frustums: true, connectors: true, every: 5,
              align: "raw", place: "gt", state: "icp", corr: true, cams: true, frame: 0};
    this.v = {yaw: -60, pitch: 25, dist: 3, tx: 0, ty: 0, tz: 0};
    this._controls();
  }
  methods() { return Object.keys(this.scene.poses); }           // "gt" first, then the estimated methods present
  estimated() { return this.methods().filter(m => m !== "gt"); }
  setScene(s) { this.scene = s; this.sceneClouds = s.clouds.map(unpackCloud);
    for (const m of this.methods()) if (this.o.show[m] === undefined) this.o.show[m] = true;
    if (!this.scene.poses[this.o.place]) this.o.place = "gt";
    this.errMax = 1e-9; for (const m of this.estimated()) for (let k = 0; k < s.frames.length; k++) this.errMax = Math.max(this.errMax, this.err(m, k));
    if (!this.views.scene) this.views.scene = Object.assign({}, s.view);
    if (this.mode === "scene") this.v = Object.assign({}, this.views.scene); }
  setRegistration(p, reg) { this.reg = {p, ti: unpackCloud(reg.clouds.i), sj: unpackCloud(reg.clouds.j)};
    const c = this.reg.ti; let s = [0, 0, 0], n = 0; for (let i = 0; i < c.xyz.length; i += 3) { s[0] += c.xyz[i]; s[1] += c.xyz[i+1]; s[2] += c.xyz[i+2]; n++; }
    this.views.registration = {yaw: -100, pitch: 22, dist: 1.9, tx: 0.6 * s[0] / n / 1000, ty: 0.6 * s[1] / n / 1000, tz: 0.6 * s[2] / n / 1000}; }
  setMode(m) { if (m !== this.mode) { this.views[this.mode] = Object.assign({}, this.v); this.mode = m; if (this.views[m]) this.v = Object.assign({}, this.views[m]); } this.draw(); }
  pose(m, k) { const P = (this.o.align === "aligned" && this.scene.poses[m].aligned) ? this.scene.poses[m].aligned : this.scene.poses[m].raw; return P[k]; }
  err(m, k) { const a = this.pose(m, k), g = this.scene.poses.gt.raw[k]; return Math.hypot(a[9]-g[9], a[10]-g[10], a[11]-g[11]); }
  rotErr(m, k) { return rotDeg(this.scene.poses.gt.raw[k], this.pose(m, k)); }
  _cam() {
    const r = this.cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    this.W = Math.max(1, Math.round(r.width * dpr)); this.H = Math.max(1, Math.round(r.height * dpr));
    if (this.cv.width !== this.W || this.cv.height !== this.H) { this.cv.width = this.W; this.cv.height = this.H; }
    this.F = this.H / (2 * Math.tan(25 * Math.PI / 180));
    const UP = this.mode === "registration" ? [0, -1, 0] : [0, 0, 1], e1 = [1, 0, 0], e2 = cross(UP, e1);
    const y = this.v.yaw * Math.PI / 180, p = this.v.pitch * Math.PI / 180;
    const d = [0, 1, 2].map(a => this.v.dist * (Math.cos(p) * Math.cos(y) * e1[a] + Math.cos(p) * Math.sin(y) * e2[a] + Math.sin(p) * UP[a]));
    this.eye = [this.v.tx + d[0], this.v.ty + d[1], this.v.tz + d[2]];
    this.F3 = nrm([this.v.tx - this.eye[0], this.v.ty - this.eye[1], this.v.tz - this.eye[2]]);
    this.R3 = nrm(cross(this.F3, UP)); this.U3 = cross(this.R3, this.F3);
  }
  proj(x, y, z) { const e = this.eye, a = x - e[0], b = y - e[1], c = z - e[2], F = this.F3, zc = a*F[0] + b*F[1] + c*F[2];
    if (zc < 0.02) return null; const R = this.R3, U = this.U3;
    return [this.W / 2 + this.F * (a*R[0] + b*R[1] + c*R[2]) / zc, this.H / 2 - this.F * (a*U[0] + b*U[1] + c*U[2]) / zc, zc]; }
  _begin() { this.buf = this.g.createImageData(this.W, this.H); this.zb = new Float32Array(this.W * this.H).fill(Infinity); this.buf.data.fill(255); }
  _pt(x, y, z, r, g, b, size) { const p = this.proj(x, y, z); if (!p) return; const u0 = Math.round(p[0]), v0 = Math.round(p[1]), h = size >> 1, W = this.W, H = this.H, d = this.buf.data;
    for (let v = v0 - h; v <= v0 + h; v++) { if (v < 0 || v >= H) continue; for (let u = u0 - h; u <= u0 + h; u++) { if (u < 0 || u >= W) continue;
      const i = v * W + u; if (p[2] < this.zb[i]) { this.zb[i] = p[2]; const o = 4 * i; d[o] = r; d[o + 1] = g; d[o + 2] = b; } } } }
  _line(a, b, col, w, dash) { const p = this.proj(...a), q = this.proj(...b); if (!p || !q) return; const g = this.g;
    g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); g.setLineDash([]); }
  _frustum(T, s, col, w, K) { K = K || {width: 640, height: 480, fx: 517.3, fy: 516.5, cx: 318.6, cy: 255.3};
    const cs = [[0,0],[K.width,0],[K.width,K.height],[0,K.height]], c = [T[9], T[10], T[11]];
    const q = cs.map(([u, v]) => xf(T, (u - K.cx) / K.fx * s, (v - K.cy) / K.fy * s, s));
    q.forEach((p, i) => { this._line(c, p, col, w); this._line(p, q[(i + 1) % 4], col, w); });
    [[1,0,0,"#d62728"],[0,1,0,"#17becf"],[0,0,1,"#1f77b4"]].forEach(([x, y, z, cc]) => this._line(c, xf(T, x*s*0.6, y*s*0.6, z*s*0.6), cc, w)); }
  _marker(P, col, shape) { const p = this.proj(...P); if (!p) return; const d = window.devicePixelRatio || 1, g = this.g;
    g.fillStyle = col; g.strokeStyle = "#000"; g.lineWidth = d; g.beginPath();
    if (shape === "o") g.arc(p[0], p[1], 5 * d, 0, 6.3); else g.rect(p[0] - 5*d, p[1] - 5*d, 10*d, 10*d); g.fill(); g.stroke(); }
  draw() {
    this._cam(); const g = this.g, dpr = window.devicePixelRatio || 1;
    g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = "#fff"; g.fillRect(0, 0, this.W, this.H);
    if (this.mode === "scene" && this.scene) this._drawScene(dpr); else if (this.mode === "registration" && this.reg) this._drawReg(dpr);
  }
  _drawScene(dpr) {
    const S = this.scene, o = this.o, N = S.frames.length, K = S.K;
    if (o.points && this.sceneClouds.length) { this._begin();
      for (const c of this.sceneClouds) { if (o.place !== "gt" && S.held[o.place][c.k]) continue;  // held pose: no estimate of its own
        const T = this.pose(o.place, c.k); for (let i = 0; i < c.xyz.length; i += 3) {
        const w = xf(T, c.xyz[i] / 1000, c.xyz[i+1] / 1000, c.xyz[i+2] / 1000); this._pt(w[0], w[1], w[2], c.rgb[i], c.rgb[i+1], c.rgb[i+2], 2 * dpr); } }
      this.g.putImageData(this.buf, 0, 0); }
    const k0 = Math.min(N - 1, o.frame), lg = Math.log(Math.max(this.errMax, 0.02) / 0.01);
    if (o.connectors) for (const m of this.estimated()) { if (!o.show[m]) continue;
      for (let k = 0; k < N; k++) { if (k % o.every && k !== k0) continue; const a = this.pose(m, k), gp = S.poses.gt.raw[k];
        const c = ramp(Math.log(Math.max(this.err(m, k), 0.01) / 0.01) / lg); this._line([gp[9], gp[10], gp[11]], [a[9], a[10], a[11]], `rgb(${c})`, 1.6 * dpr); } }
    for (const m of this.methods()) { if (!o.show[m]) continue; const g = this.g;
      g.strokeStyle = METHOD_COLOR[m]; g.lineWidth = (m === "gt" ? 2.6 : 2) * dpr; g.beginPath(); let on = false;
      for (let k = 0; k < N; k++) { const T = this.pose(m, k), p = this.proj(T[9], T[10], T[11]); if (!p) { on = false; continue; } if (on) g.lineTo(p[0], p[1]); else g.moveTo(p[0], p[1]); on = true; }
      g.stroke();
      if (o.frustums) this._frustum(this.pose(m, k0), 0.07, METHOD_COLOR[m], 1.8 * dpr, K);
      const a = this.pose(m, 0), b = this.pose(m, N - 1); this._marker([a[9], a[10], a[11]], METHOD_COLOR[m], "o"); this._marker([b[9], b[10], b[11]], METHOD_COLOR[m], "s"); }
  }
  _drawReg(dpr) {
    const {p, ti, sj} = this.reg, o = this.o;
    const T = o.state === "icp" ? (p.T_icp_ij ? m4(p.T_icp_ij) : null) : o.state === "gt" ? invT(m4(p.T_gt_ji)) : [1,0,0,0,1,0,0,0,1,0,0,0];
    if (o.points) { this._begin(); const f = l => 0.35 + 0.75 * l;
      for (let i = 0; i < ti.xyz.length; i += 3) { const l = (ti.rgb[i] + ti.rgb[i+1] + ti.rgb[i+2]) / 765; this._pt(ti.xyz[i] / 1000, ti.xyz[i+1] / 1000, ti.xyz[i+2] / 1000, 51 * f(l), 115 * f(l), 230 * f(l), 2 * dpr); }
      if (T) for (let i = 0; i < sj.xyz.length; i += 3) { const l = (sj.rgb[i] + sj.rgb[i+1] + sj.rgb[i+2]) / 765, w = xf(T, sj.xyz[i] / 1000, sj.xyz[i+1] / 1000, sj.xyz[i+2] / 1000);
        this._pt(w[0], w[1], w[2], Math.min(255, 250 * f(l)), 140 * f(l), 26 * f(l), 2 * dpr); }
      this.g.putImageData(this.buf, 0, 0); }
    if (o.corr && T) { const c = p.icp; for (let n = 0; n < c.cls.length; n++) { if (c.cls[n] !== "I" && n % c.rej_every) continue;
      const a = xf(T, c.pj[3*n], c.pj[3*n+1], c.pj[3*n+2]), b = [c.pi[3*n], c.pi[3*n+1], c.pi[3*n+2]], r = Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
      this._line(a, b, `rgb(${ramp(r / 0.05)})`, (c.cls[n] === "I" ? 2.2 : 1.1) * dpr, c.cls[n] === "I" ? [] : [5, 4]); } }
    if (o.cams) { this._frustum([1,0,0,0,1,0,0,0,1,0,0,0], 0.08, "#000", 1.6 * dpr);
      for (const [m, Tji] of [["gt", p.T_gt_ji], ["pnp", p.T_pnp_ji], ["icp", p.T_icp_ji]]) if (Tji) this._frustum(invT(m4(Tji)), 0.08, METHOD_COLOR[m], 2 * dpr); }
  }
  _controls() {
    let drag = null, moved = false; const cv = this.cv;
    cv.addEventListener("pointerdown", e => { drag = {x: e.clientX, y: e.clientY, pan: e.shiftKey}; moved = false; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointermove", e => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
      drag.x = e.clientX; drag.y = e.clientY; const v = this.v;
      if (drag.pan) { const s = v.dist / 700; v.tx -= (this.R3[0]*dx - this.U3[0]*dy) * s; v.ty -= (this.R3[1]*dx - this.U3[1]*dy) * s; v.tz -= (this.R3[2]*dx - this.U3[2]*dy) * s; }
      else { v.yaw -= dx * 0.4; v.pitch = Math.max(-89, Math.min(89, v.pitch + dy * 0.4)); } this.draw(); if (this.onView) this.onView(this.v); });
    cv.addEventListener("pointerup", e => { const was = drag; drag = null; if (!was || moved || this.mode !== "scene" || !this.scene) return;
      const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1, x = (e.clientX - r.left) * d, y = (e.clientY - r.top) * d;
      let best = null, bd = 18 * d; for (const m of this.methods()) { if (!this.o.show[m]) continue;
        for (let k = 0; k < this.scene.frames.length; k++) { const T = this.pose(m, k), p = this.proj(T[9], T[10], T[11]); if (p) { const dd = Math.hypot(p[0] - x, p[1] - y); if (dd < bd) { bd = dd; best = k; } } } }
      if (best !== null && this.onPick) this.onPick(best); });
    cv.addEventListener("wheel", e => { e.preventDefault(); this.v.dist *= Math.exp(e.deltaY * 0.001); this.draw(); if (this.onView) this.onView(this.v); }, {passive: false});
  }
}
