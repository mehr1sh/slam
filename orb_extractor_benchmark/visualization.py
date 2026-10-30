"""Clean, spatially thinned display for ORB correspondences and HTML reports.

Visualization-only sampling never changes feature extraction, matching, or metrics.
No PnP or RGB-D pose estimation is performed by this benchmark.
"""
from __future__ import annotations

import html
import math
import re
from pathlib import Path

import cv2
import numpy as np


def safe_name(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(value)).strip('._')[:90] or 'pair'


def _eligible(points, max_points, min_spacing, seed_indices=None):
    """Pick spatially separated points (ONLY FOR DRAWING)."""
    n = len(points)
    if max_points <= 0 or n == 0:
        return []
    order = range(n) if seed_indices is None else seed_indices
    selected, cellset = [], set()
    # Conservative grid-spacing. Independent of actual detector NMS.
    cell = max(1.0, float(min_spacing))
    for idx in order:
        x, y = points[idx]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        if any((gx + dx, gy + dy) in cellset for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)):
            continue
        selected.append(int(idx))
        cellset.add((gx, gy))
        if len(selected) >= max_points:
            break
    return selected


def draw_points(gray, points, max_points=130, spacing=5, radius=2):
    """Return preview with small, non-overlapping feature centres (not rich circles)."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    selected = _eligible(points, max_points, spacing)
    for i in selected:
        p = tuple(np.rint(points[i]).astype(int))
        cv2.circle(rgb, p, radius, (0, 214, 255), -1, cv2.LINE_AA)
    return rgb, len(selected)


def draw_correspondences(im1, im2, features1, features2, matches, class_mask,
                         max_matches=55, spacing=7, marker_radius=2):
    """Render adjacent frames with minimally cluttered, classified match lines.

    class_mask is None for unknown geometry, otherwise a length-M bool array.
    Blue-green means inlier, orange-red outlier; grey means unclassified.
    """
    first = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR) if im1.ndim == 2 else im1.copy()
    second = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR) if im2.ndim == 2 else im2.copy()
    # Both frames retain native image resolution. For unequal heights, pad canvas.
    h = max(first.shape[0], second.shape[0]); w = first.shape[1] + second.shape[1]
    canvas = np.full((h, w, 3), 248, np.uint8)
    canvas[:first.shape[0], :first.shape[1]] = first
    canvas[:second.shape[0], first.shape[1]:] = second
    if not matches or max_matches <= 0:
        return canvas, 0

    # Distance priority, but ensure broad image coverage. No feature is deleted.
    selected = []
    source_cells = set()
    target_cells = set()
    cell = max(float(spacing), 1.)
    for j, m in enumerate(matches):
        p = features1.points[m.queryIdx]; q = features2.points[m.trainIdx]
        if not (np.isfinite(p).all() and np.isfinite(q).all()):
            continue
        c1 = (int(p[0] // cell), int(p[1] // cell))
        c2 = (int(q[0] // cell), int(q[1] // cell))
        if c1 in source_cells or c2 in target_cells:
            continue
        selected.append(j)
        source_cells.add(c1); target_cells.add(c2)
        if len(selected) == max_matches:
            break

    colors = {'inlier': (217, 174, 21), 'outlier': (64, 81, 232),
              'unclassified': (185, 165, 127)}  # OpenCV uses BGR
    # Draw lines first, dots second. Use thin lines instead of large circles.
    for j in selected:
        m = matches[j]
        p = tuple(np.rint(features1.points[m.queryIdx]).astype(int))
        q0 = np.rint(features2.points[m.trainIdx]).astype(int)
        q = (int(q0[0] + first.shape[1]), int(q0[1]))
        kind = ('unclassified' if class_mask is None
                else ('inlier' if bool(class_mask[j]) else 'outlier'))
        cv2.line(canvas, p, q, colors[kind], 1, cv2.LINE_AA)
    for j in selected:
        m = matches[j]
        p = tuple(np.rint(features1.points[m.queryIdx]).astype(int))
        q0 = np.rint(features2.points[m.trainIdx]).astype(int)
        q = (int(q0[0] + first.shape[1]), int(q0[1]))
        kind = ('unclassified' if class_mask is None
                else ('inlier' if bool(class_mask[j]) else 'outlier'))
        # Tiny yellow centre and subtle dark edge keep the original image visible.
        cv2.circle(canvas, p, marker_radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, q, marker_radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, marker_radius, (0, 226, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, q, marker_radius, (0, 226, 255), -1, cv2.LINE_AA)
    return canvas, len(selected)


def _fmt_num(n, digits=3):
    if n is None or n == '':
        return 'N/A'
    try:
        return f'{float(n):.{digits}f}'
    except (TypeError, ValueError):
        return html.escape(str(n))


def _fmt_pct(n):
    return 'N/A' if n is None or n == '' else f'{100 * float(n):.1f}%'


def render_panel(method, row, img_rel, drawn, dataset, sequence, pair_name, args):
    name = 'Scratch ORB correspondences' if method == 'scratch' else 'OpenCV ORB correspondences'
    color = '#16b7be' if method == 'scratch' else '#e5555c'
    if row['geometry'] == 'homography_ground_truth':
        basis = f'Ground-truth homography, reprojection error ≤ {args.epsilon:g}px'
        n_in = row['correct_matches']; n_out = row['matches'] - row['correct_matches']
        label_in, label_out = 'GT correct matches', 'GT incorrect matches'
        legend = 'Blue-green: correct under ground-truth homography · Red: incorrect'
    else:
        basis = 'Estimated fundamental-matrix RANSAC (1.5 px); not GT or PnP'
        n_in = row['fundamental_ransac_inliers']
        n_out = row['matches'] - n_in if n_in is not None else None
        label_in, label_out = 'F-RANSAC inliers (estimated)', 'F-RANSAC outliers (estimated)'
        legend = ('Blue-green: estimated F-inlier · Red: estimated F-outlier'
                  if n_in is not None else 'Grey: matches unclassified; insufficient F-RANSAC geometry')
    tuning = (f'{args.levels} pyramid levels, scale {args.scale_factor:g}, '
              f'FAST threshold {args.fast_threshold}, '
              f'{args.features_per_level} features/level (nominal for scratch)')
    if method == 'opencv':
        config = 'OpenCV ORB detector and BRIEF-style descriptors'
    else:
        config = 'Original C++ FAST/Harris-style pyramid + rotated BRIEF descriptors'
    desc = (f'{config}. {tuning}. Mutual nearest-neighbour Hamming matching'
            + (f', distance ≤ {args.max_hamming_distance}' if args.max_hamming_distance > 0 else ', no distance cutoff')
            + f'. {basis}. Display only: {drawn} spatially separated matches (of {row["matches"]}).')
    entries = [
        ('Dataset / sequence', f'{dataset} / {sequence}'),
        ('Pair', pair_name),
        ('Image files', f'{Path(row["image_1"]).name} → {Path(row["image_2"]).name}'),
        ('Keypoints i / j', f'{row["keypoints_1"]} / {row["keypoints_2"]}'),
        ('Raw mutual Hamming matches', row['raw_matches']),
        ('Filtered matches', row['matches']),
        (label_in, 'N/A' if n_in is None else n_in),
        (label_out, 'N/A' if n_out is None else n_out),
    ]
    if row['geometry'] == 'homography_ground_truth':
        entries += [
            ('Matching precision (GT)', _fmt_pct(row['match_precision'])),
            ('Keypoint repeatability (GT)', _fmt_pct(row['repeatability'])),
        ]
    else:
        entries += [('F-RANSAC inlier ratio (estimated)', _fmt_pct(row['fundamental_ransac_inlier_ratio']))]
    entries += [
        ('Extract image i / j', f'{_fmt_num(row["extract_1_ms"])} / {_fmt_num(row["extract_2_ms"])} ms'),
        ('Descriptor matching time', f'{_fmt_num(row["match_ms"])} ms'),
        ('Drawn matches (visual only)', drawn),
    ]
    lines = '\n'.join('<tr><th>' + html.escape(str(k)) + '</th><td>' + html.escape(str(v)) + '</td></tr>'
                      for k, v in entries)
    return f'''<section class="panel" style="--accent:{color}">
      <h2><span class="swatch"></span>{html.escape(name)}</h2>
      <p class="description">{html.escape(desc)}</p>
      <img class="matches" src="{html.escape(img_rel, quote=True)}" alt="Correspondence lines between two frames for {html.escape(name)}" loading="lazy">
      <div class="legend">{html.escape(legend)}. Yellow dots mark matched feature centres.</div>
      <table>{lines}</table>
    </section>'''


_CSS = '''
:root { font-family: Inter, Arial, Helvetica, sans-serif; color: #242930; background: #f4f6f9; }
* { box-sizing: border-box; }
body { margin: 0; }
header { background: white; padding: 22px 28px; border-bottom: 1px solid #e1e5ec; }
header h1 { margin: 0 0 6px; font-size: 23px; }
header p { color: #535e6d; margin: 0; font-size: 14px; line-height: 1.5; }
main { max-width: 1800px; margin: 18px auto; padding: 0 16px; }
.compare { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 540px),1fr)); gap: 18px; align-items: start; }
.panel { background: white; padding: 18px; border: 1px solid #e1e5ec; border-radius: 8px; min-width: 0; }
h2 { font-size: 19px; margin: 2px 0 12px; display: flex; align-items: center; gap: 9px; }
.swatch { display:inline-block; width:13px; height:13px; border-radius:3px; background:var(--accent); flex: 0 0 auto; }
.description { min-height: 74px; color:#596472; font-size: 14px; line-height:1.55; margin:0 0 10px; }
img.matches { display:block; width:100%; height:auto; background:#f5f5f5; border:1px solid #ddd; }
.legend { color:#617080; font-size:12px; margin:10px 0; }
table { width: 100%; border-collapse: collapse; font-size:14px; }
th,td { border-top:1px solid #e2e6ea; padding:9px 8px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }
th { color:#454e58; font-weight:500; width:49%; } td { font-weight:500; }
.explain { font-size: 14px; line-height: 1.6; background:#fff; border:1px solid #e1e5ec; border-radius:8px; margin-top:16px; padding:15px 20px; }
.explain h3 { margin:0 0 8px; } .explain p { margin:6px 0; }
a { color:#136eb5; text-decoration:none; } a:hover { text-decoration:underline; }
li { margin-bottom:7px; }
'''


def write_pair_dashboard(path, panels, dataset, sequence, pair_name, args):
    page_title = f'ORB comparison — {dataset} / {sequence} / {pair_name}'
    body = '\n'.join(panels)
    caveat = ('''<section class="explain"><h3>How to read these results</h3>
      <p><strong>Keypoints:</strong> counts reported by the extractors, without visualization thinning.
      <strong>Raw mutual matches:</strong> reciprocal nearest-neighbour descriptor matches by Hamming distance.
      <strong>Filtered matches:</strong> after any optional descriptor-distance cutoff (default: none).</p>
      <p><strong>Inliers / outliers:</strong> for Oxford, HPatches, or synthetic pairs, correctness is decided
      from a supplied homography at the chosen pixel threshold. For TUM, KITTI, and arbitrary video,
      an estimated fundamental matrix is used to label geometric consistency;
      it is <strong>not ground truth, a PnP classification, or proof of pose accuracy.</strong></p>
      <p><strong>Visualization:</strong> just a small spatially separated selection of matches is drawn.
      No match or keypoint is discarded from the CSV or numerical metrics due to drawing limits.
      <strong>Extraction time:</strong> algorithm runtime (excludes image loading / subprocess overhead).</p></section>''')
    path.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <title>{html.escape(page_title)}</title><style>{_CSS}</style></head>
      <body><header><p><a href="../../index.html">← All datasets</a> &nbsp; · &nbsp; <a href="index.html">This dataset</a></p><h1>{html.escape(page_title)}</h1><p>OpenCV versus original scratch C++ ORB. Both evaluated with the same mutual Hamming matcher.</p></header>
      <main><div class="compare">{body}</div>{caveat}</main></body></html>''', encoding='utf-8')


def write_gallery(path, items, dataset):
    lis = '\n'.join(f'<li><a href="{html.escape(fname, quote=True)}">{html.escape(title)}</a></li>'
                    for fname, title in items)
    path.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>ORB match dashboards</title><style>{_CSS}</style></head><body>
        <header><h1>{html.escape(dataset)} — ORB correspondence dashboards</h1>
        <p>Each pair compares the original C++ extractor with OpenCV, including metrics and interpretation.</p></header>
        <main><section class="explain"><h3>Pairs</h3><ol>{lis}</ol></section></main>
        </body></html>''', encoding='utf-8')


def write_dashboard_png(path, panel_data, dataset, sequence, pair_name, args):
    """Save a self-contained screenshot-style PNG (alongside the interactive HTML).

    panel_data: list of tuples (method, row, image_path, drawn).
    """
    from PIL import Image, ImageDraw, ImageFont
    def font(size, bold=False):
        name = ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default()
    regular = font(17); small = font(15); bold = font(19, True); heading = font(25, True)
    W, margin, gap, pad = 1700, 20, 20, 22
    col_w = (W - margin * 2 - gap) // 2
    x0 = [margin, margin + col_w + gap]
    background = '#f4f6f9'; fg = '#212934'; secondary = '#526173'
    # Fixed-size canvas accommodates all the original metrics without clipping.
    H = 1250
    im = Image.new('RGB', (W, H), background)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, W, 95), fill='white')
    draw.text((26, 19), f'ORB correspondences | {dataset} / {sequence} / {pair_name}', font=heading, fill=fg)
    draw.text((26, 57), 'Same Hamming matcher for both extractors; all numerical metrics use full match sets',
              font=regular, fill=secondary)

    def wrap(text, fnt, maxwidth):
        words = str(text).split(); lines = []; buf = ''
        for word in words:
            test = (buf + ' ' + word).strip()
            if draw.textbbox((0, 0), test, font=fnt)[2] > maxwidth and buf:
                lines.append(buf); buf = word
            else:
                buf = test
        if buf:
            lines.append(buf)
        return lines

    for n, (method, row, source_img, drawn) in enumerate(panel_data):
        left = x0[n]; right = left + col_w
        draw.rounded_rectangle((left, 112, right, H - 97), radius=9, fill='white', outline='#dfe5ed', width=2)
        accent = '#1eb6c2' if method == 'scratch' else '#e4545c'
        title = 'Scratch ORB correspondences' if method == 'scratch' else 'OpenCV ORB correspondences'
        draw.rounded_rectangle((left + pad, 138, left + pad + 14, 152), radius=3, fill=accent)
        draw.text((left + pad + 26, 132), title, font=bold, fill=fg)
        config = ('Original C++ pyramid FAST/Harris + rotated BRIEF' if method == 'scratch'
                  else 'OpenCV ORB with Harris scoring')
        basis = ('Ground-truth homography classification' if row['geometry'] == 'homography_ground_truth'
                 else 'Estimated F-RANSAC classification; NOT PnP or GT')
        desc = f'{config}. Mutual nearest-neighbour Hamming matching. {basis}. Only {drawn} spatially separated matches drawn.'
        yy = 173
        for line in wrap(desc, small, col_w - 2 * pad):
            draw.text((left + pad, yy), line, font=small, fill=secondary)
            yy += 23
        yy = max(270, yy + 13)
        photo = Image.open(source_img).convert('RGB')
        max_w, max_h = col_w - 2 * pad, 310
        photo.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        im.paste(photo, (left + pad + (max_w-photo.width)//2, yy))
        yy += photo.height + 24
        legend = ('Blue: ground-truth correct, red: incorrect' if row['geometry'] == 'homography_ground_truth'
                  else 'Blue: estimated F-inlier, red: estimated F-outlier')
        draw.text((left + pad, yy), legend + ' | yellow: selected keypoint centres',
                  font=small, fill=secondary)
        yy += 36
        if row['geometry'] == 'homography_ground_truth':
            inlier, outlier = row['correct_matches'], row['matches'] - row['correct_matches']
            geo_rows = [('GT correct / incorrect', f'{inlier} / {outlier}'),
                        ('GT matching precision', _fmt_pct(row['match_precision'])),
                        ('GT repeatability', _fmt_pct(row['repeatability']))]
        else:
            inlier = row['fundamental_ransac_inliers']
            outlier = row['matches'] - inlier if inlier is not None else None
            geo_rows = [('F-RANSAC inliers / outliers (est.)',
                         f'{inlier if inlier is not None else "N/A"} / {outlier if outlier is not None else "N/A"}'),
                        ('F-RANSAC inlier ratio (est.)', _fmt_pct(row['fundamental_ransac_inlier_ratio']))]
        fields = [('Image pair', f'{Path(row["image_1"]).name} -> {Path(row["image_2"]).name}'),
                  ('Keypoints i / j', f'{row["keypoints_1"]} / {row["keypoints_2"]}'),
                  ('Raw mutual descriptor matches', row['raw_matches']),
                  ('Matches after Hamming filter', row['matches']),
                  *geo_rows,
                  ('Extraction time i / j', f'{_fmt_num(row["extract_1_ms"])} / {_fmt_num(row["extract_2_ms"])} ms'),
                  ('Match time', f'{_fmt_num(row["match_ms"])} ms'),
                  ('Displayed matches (visual only)', drawn)]
        for k, v in fields:
            draw.line((left + pad, yy - 5, right - pad, yy - 5), fill='#e0e5ec', width=1)
            draw.text((left + pad, yy + 4), str(k), font=small, fill=secondary)
            wrapped = wrap(str(v), small, (col_w - 2 * pad)//2 - 4)
            for jj, line in enumerate(wrapped[:3]):
                draw.text((left + col_w // 2, yy + 4 + 20 * jj), line, font=small, fill=fg)
            yy += max(36, 12 + 20 * min(len(wrapped), 3))
        if yy > H - 98:
            raise RuntimeError(f'Dashboard height too small for panel: {yy} > {H-98}')
    notes = ("Ground-truth homography metrics apply to synthetic, Oxford, and HPatches pairs. "
             "On TUM/KITTI/folder, F-RANSAC consistency is estimated, not verified pose accuracy. "
             "Shown dots/lines are display-only; counts and CSV include all matches.")
    y = H - 77
    for line in wrap(notes, small, W - 50):
        draw.text((25, y), line, font=small, fill=secondary); y += 22
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, optimize=True)
