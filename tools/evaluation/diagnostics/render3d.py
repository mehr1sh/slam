"""Minimal software renderer for static 3D views (NumPy + Matplotlib, no GPU, no extra dependency).

A virtual pinhole camera (OpenCV axes: x right, y down, z forward) looks at the scene; points are splatted
into an image with a z-buffer (nearest point wins per pixel); line segments (trajectories, frustums,
correspondence vectors) are projected and drawn on top with Matplotlib, without occlusion by points.
World units are metres; nothing here changes any geometry, it only projects it.
"""

import math

import numpy as np


class View:
    def __init__(self, eye, target, up=(0, 0, 1), fov_deg=55.0, size=(1280, 860)):
        self.W, self.H = size
        self.eye = np.asarray(eye, float)
        f = np.asarray(target, float) - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, float))
        if np.linalg.norm(r) < 1e-9:  # looking along `up`: pick any perpendicular
            r = np.cross(f, [1.0, 0, 0])
        r /= np.linalg.norm(r)
        d = np.cross(f, r)  # image "down"
        self.R = np.stack([r, d, f])  # world -> view rotation (rows)
        self.f = self.H / 2 / math.tan(math.radians(fov_deg) / 2)
        self.near = 0.02

    def project(self, P):
        Y = (np.asarray(P, float) - self.eye) @ self.R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.c_[self.f * Y[:, 0] / Y[:, 2] + self.W / 2, self.f * Y[:, 1] / Y[:, 2] + self.H / 2]
        return uv, Y[:, 2]

    def splat(self, P, colors, radius=1, image=None, depth=None):
        """Draw points (N,3) with RGB colours (N,3, 0..1) as (2r+1)^2 squares; z-buffered."""
        img = np.ones((self.H, self.W, 3)) if image is None else image
        zb = np.full((self.H, self.W), np.inf) if depth is None else depth
        uv, z = self.project(P)
        ok = z > self.near
        uv, z, col = uv[ok], z[ok], np.asarray(colors, float)[ok]
        u0, v0 = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                u, v = u0 + du, v0 + dv
                m = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
                if not m.any():
                    continue
                idx = v[m] * self.W + u[m]
                zz, cc = z[m], col[m]
                o = np.lexsort((zz, idx))  # per pixel, nearest first
                idx, zz, cc = idx[o], zz[o], cc[o]
                first = np.r_[True, idx[1:] != idx[:-1]]
                idx, zz, cc = idx[first], zz[first], cc[first]
                flat = zb.reshape(-1)
                win = zz < flat[idx]
                flat[idx[win]] = zz[win]
                img.reshape(-1, 3)[idx[win]] = cc[win]
        return img, zb

    def segments_2d(self, segs):
        """(M,2,3) world segments -> list of ((u0,v0),(u1,v1)) for segments fully in front of the camera."""
        segs = np.asarray(segs, float).reshape(-1, 2, 3)
        uv, z = self.project(segs.reshape(-1, 3))
        uv, z = uv.reshape(-1, 2, 2), z.reshape(-1, 2)
        keep = (z > self.near).all(axis=1)
        return uv[keep], keep

    def scale_at(self, point):
        """Pixels per metre at the depth of `point`."""
        _, z = self.project(np.asarray(point, float)[None])
        return self.f / z[0]


def draw_segments(ax, view, segs, **kw):
    uv, _ = view.segments_2d(segs)
    for (a, b) in uv:
        ax.plot([a[0], b[0]], [a[1], b[1]], **kw)


def draw_polyline(ax, view, P, **kw):
    P = np.asarray(P, float)
    uv, z = view.project(P)
    uv = np.where((z > view.near)[:, None], uv, np.nan)
    ax.plot(uv[:, 0], uv[:, 1], **kw)


def draw_axes_gizmo(ax, view, labels=("x", "y", "z"), at=(70, None), length=45):
    """World axis directions (projected) in a corner; lengths are directional only."""
    u0, v0 = at[0], (view.H - 70) if at[1] is None else at[1]
    for k, (c, lab) in enumerate(zip(("#d62728", "#17becf", "#1f77b4"), labels)):
        d = view.R @ np.eye(3)[k]
        vec = d[:2] / max(np.linalg.norm(d[:2]), 1e-9) * length * np.linalg.norm(d[:2])
        ax.annotate("", (u0 + vec[0], v0 + vec[1]), (u0, v0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
        ax.text(u0 + vec[0] * 1.2, v0 + vec[1] * 1.2, lab, color=c, fontsize=9, ha="center", va="center", weight="bold")


def draw_scale_bar(ax, view, point, metres, at=None):
    px = view.scale_at(point) * metres
    u0, v0 = at if at else (view.W - px - 40, view.H - 40)
    ax.plot([u0, u0 + px], [v0, v0], color="black", lw=3)
    ax.text(u0 + px / 2, v0 - 10, f"{metres:g} m at the scene centre", ha="center", va="bottom", fontsize=8)


def look_at_bounds(points, direction, distance_factor=1.6, fov_deg=55.0, size=(1280, 860), up=(0, 0, 1),
                   percentile=(2, 98)):
    """A view from `direction` (world vector from target to eye) framing the robust extent of `points`."""
    lo, hi = np.percentile(points, percentile[0], axis=0), np.percentile(points, percentile[1], axis=0)
    c = (lo + hi) / 2
    rad = np.linalg.norm(hi - lo) / 2
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    dist = rad / math.tan(math.radians(fov_deg) / 2) * distance_factor * 0.75
    return View(c + d * dist, c, up=up, fov_deg=fov_deg, size=size)
