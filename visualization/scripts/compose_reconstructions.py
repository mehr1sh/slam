#!/usr/bin/env python3
"""Side-by-side image of the three reconstructions rendered by build_tum_scene.py --render-all (same camera,
frame, density and overlays). Captions carry the numbers from comparison_<density>.json; the images are not
modified (no cropping, scaling or colour change).

Usage: python3 visualization/scripts/compose_reconstructions.py --export <export dir> --prefix <render prefix> \
           --density preview|render [--out <png>]
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABEL = {"gt": "placed with GROUND-TRUTH poses (reference)", "pnp": "placed with raw PnP poses",
         "icp": "placed with raw ICP poses"}
COLOR = {"gt": (38, 209, 115), "pnp": (255, 158, 26), "icp": (217, 102, 242)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--density", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    exp = Path(a.export)
    cmp = json.loads((exp / f"comparison_{a.density}.json").read_text())
    imgs = {m: Image.open(f"{a.prefix}_{m}.png").convert("RGB") for m in ("gt", "pnp", "icp")}
    W, H = imgs["gt"].size
    cap = 92
    out = Image.new("RGB", (3 * W + 20, H + cap), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = bold = ImageFont.load_default()
    d = ImageDraw.Draw(out)
    for n, m in enumerate(("gt", "pnp", "icp")):
        x = n * (W + 10)
        out.paste(imgs[m], (x, cap))
        man = json.loads((exp / f"manifest_{a.density}_{m}.json").read_text())
        c = cmp[m]
        d.rectangle([x, 0, x + 10, 22], fill=COLOR[m])
        d.text((x + 16, 2), LABEL[m], fill="black", font=bold)
        d.text((x, 26), f"{man['frames_processed']} frames, {man['points_after_sampling']} points; skipped "
                        f"{[s['frame'] for s in man['frames_skipped']] or 'none'}", fill="black", font=font)
        d.text((x, 46), f"occupied 1 cm voxels per 1000 points: {c['occupied_voxels_per_1000_points']:.0f}",
               fill="black", font=font)
        dv = c.get("displacement_vs_gt_m")
        d.text((x, 66), (f"same pixels vs GT cloud: median {100 * dv['median']:.1f} cm, p90 {100 * dv['p90']:.1f} cm, "
                         f"max {dv['max']:.2f} m" if dv else "reference"), fill="black", font=font)
    path = Path(a.out or f"{a.prefix}_comparison.png")
    out.save(path)
    print(path)


if __name__ == "__main__":
    main()
