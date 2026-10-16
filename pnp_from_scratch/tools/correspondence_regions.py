"""Neighbourhood colours for the correspondence figures (visualization only).

Colour = local keypoint neighbourhood of the frame-i end of a match:
  1. the frame-i endpoints are grouped into K neighbourhoods by k-means
     (deterministic farthest-point seeding, fixed number of iterations), so
     nearby keypoints share a colour and the groups follow the object's shape;
  2. each neighbourhood gets a hue from the angle of its centre around the
     centre of all endpoints (a smooth rainbow around the object, kept below
     the magenta used for warnings).

Neighbourhood of the frame-(i+1) end: the labels of the K_VOTE nearest
frame-(i+1) endpoints of OTHER geometrically verified matches (RANSAC inliers);
label_j is the majority. A match is flagged as `neighbourhood_changed` when its
own frame-i label is not among those K_VOTE labels, i.e. it lands where none of
its neighbourhood went (a match near a neighbourhood border is not flagged).
This is a visual warning only. Inlier / outlier status always
comes from the C++ pipeline's RANSAC export:
  inlier    pnp_inlier == 1 (RANSAC's inlier set; the refinement keeps it)
  outlier   pnp_used == 1 and pnp_inlier == 0
  no depth  pnp_used == 0 (no frame-i depth, not a PnP input)
"""

import colorsys

import numpy as np

K = 12        # neighbourhoods per pair
K_VOTE = 5    # frame-(i+1) neighbours that vote for the label of a frame-(i+1) end
ITERATIONS = 20


def neighbourhoods(p, k=K, iterations=ITERATIONS):
    """Deterministic k-means of points p (n x 2). Returns (labels, centres)."""
    p = np.asarray(p, float)
    k = min(k, len(p))
    first = int(np.argmin(p[:, 0] + p[:, 1]))  # top-left-most point
    seeds = [first]
    d = np.linalg.norm(p - p[first], axis=1)
    for _ in range(1, k):  # farthest-point seeding: spread over the object
        seeds.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(p - p[seeds[-1]], axis=1))
    c = p[seeds].copy()
    for _ in range(iterations):
        lab = np.argmin(((p[:, None, :] - c[None, :, :]) ** 2).sum(2), axis=1)
        for j in range(k):
            if (lab == j).any():
                c[j] = p[lab == j].mean(0)
    lab = np.argmin(((p[:, None, :] - c[None, :, :]) ** 2).sum(2), axis=1)
    return lab, c


def neighbourhood_colours(centres):
    """One RGB colour per neighbourhood: hue 0 .. 0.72 (red .. violet, clear of the warning magenta) by angle around the mean centre."""
    m = centres.mean(0)
    ang = np.arctan2(centres[:, 1] - m[1], centres[:, 0] - m[0])
    rank = np.argsort(np.argsort(ang))
    n = max(1, len(centres) - 1)
    return np.array([colorsys.hsv_to_rgb(0.72 * r / n, 0.85, 1.0) for r in rank])


def classify(m, k=K, k_vote=K_VOTE):
    """Visualization data of the matches of one pair.

    m: dict with arrays ui, vi, uj, vj (pixels), used, inlier (bool, from the C++ export).
    Returns per-match arrays: label_i, label_j, colour (of label_i), status
    ('inlier' / 'outlier' / 'nodepth'), neighbourhood_changed (bool).
    """
    used, inl = np.asarray(m["used"], bool), np.asarray(m["inlier"], bool)
    if (inl & ~used).any():
        raise ValueError("an inlier must be a PnP input (pnp_used)")
    pi = np.c_[m["ui"], m["vi"]].astype(float)
    pj = np.c_[m["uj"], m["vj"]].astype(float)
    label_i, centres = neighbourhoods(pi, k)
    colour = neighbourhood_colours(centres)[label_i]
    anchors = np.nonzero(inl)[0] if inl.sum() > k_vote else np.arange(len(pi))
    label_j, changed = label_i.copy(), np.zeros(len(pj), bool)
    for n in range(len(pj)):
        a = anchors[anchors != n]
        near = a[np.argsort(np.linalg.norm(pj[a] - pj[n], axis=1), kind="stable")[:k_vote]]
        votes = np.bincount(label_i[near], minlength=len(centres))
        label_j[n] = int(np.argmax(votes))  # ties -> lowest label
        changed[n] = votes[label_i[n]] == 0
    status = np.where(inl, "inlier", np.where(used, "outlier", "nodepth"))
    return dict(label_i=label_i, label_j=label_j, colour=colour, status=status,
                neighbourhood_changed=changed)
