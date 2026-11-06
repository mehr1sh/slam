#!/usr/bin/env python3
"""Benchmark original scratch ORB vs OpenCV ORB on five dataset formats.

Copyright for original C++ feature code is preserved in its original files.
This Python script is an evaluation adapter; it does not reimplement scratch ORB.
"""
import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from visualization import (draw_correspondences, render_panel, write_pair_dashboard,
                           write_gallery, safe_name, write_dashboard_png)

cv2.setNumThreads(1)

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.ppm', '.pgm', '.tif', '.tiff'}
FIELDS = [
    'dataset', 'sequence', 'pair', 'extractor', 'image_1', 'image_2',
    'keypoints_1', 'keypoints_2', 'raw_matches', 'matches', 'correct_matches', 'match_precision',
    'repeatable_keypoints', 'eligible_keypoints', 'repeatability',
    'fundamental_ransac_inliers', 'fundamental_ransac_inlier_ratio',
    'extract_1_ms', 'extract_2_ms', 'match_ms', 'geometry',
]


@dataclass
class Features:
    points: np.ndarray              # (N,2), base image coordinates
    descriptors: np.ndarray         # (N,32), uint8
    angles: np.ndarray              # radians
    scales: np.ndarray
    responses: np.ndarray
    levels: np.ndarray
    milliseconds: float


@dataclass
class Pair:
    name: str
    sequence: str
    first: Path
    second: Path
    homography: np.ndarray | None = None
    second_image: np.ndarray | None = None
    mask: np.ndarray | None = None
    # Optional for generated synthetic frames (stored without writing to disk).


def image_paths(folder, recursive=False):
    folder = Path(folder)
    iterator = folder.rglob('*') if recursive else folder.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and not re.search(r'(?:^|_)depth(?:_|\.)', p.name.lower()))


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f'Cannot read image: {path}')
    return image


def scratch_extract(image, args, tmp):
    pgm = tmp / 'current.pgm'
    csvpath = tmp / 'current.csv'
    if not cv2.imwrite(str(pgm), image):
        raise RuntimeError('Failed to create PGM for scratch executable')
    cmd = [str(args.binary), '--input', str(pgm), '--output', str(csvpath),
           '--levels', str(args.levels), '--scale-factor', str(args.scale_factor),
           '--fast-threshold', str(args.fast_threshold),
           '--features-per-level', str(args.features_per_level),
           '--cell-size', str(args.cell_size), '--min-threshold', str(args.min_threshold)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f'Scratch extractor failed:\n{proc.stderr}\nCommand: {cmd}')
    stats = json.loads(proc.stdout)
    coords, desc, angle, scales, scores, levels = [], [], [], [], [], []
    with csvpath.open(newline='') as f:
        for row in csv.DictReader(f):
            coords.append([float(row['x']), float(row['y'])])
            desc.append(bytes.fromhex(row['descriptor_hex']))
            angle.append(float(row['angle_rad']))
            scales.append(float(row['scale']))
            scores.append(float(row['score']))
            levels.append(int(row['level']))
    if desc:
        ds = np.array([np.frombuffer(x, np.uint8) for x in desc], dtype=np.uint8)
    else:
        ds = np.empty((0, 32), dtype=np.uint8)
    return Features(np.asarray(coords, np.float32).reshape(-1, 2), ds,
                    np.asarray(angle, np.float32), np.asarray(scales, np.float32),
                    np.asarray(scores, np.float32), np.asarray(levels, np.int32),
                    float(stats['extract_ms']))


def opencv_extract(image, args):
    # Same nominal total feature budget, pyramid levels, scale and FAST threshold.
    # Different implementation: OpenCV uses a learned ORB pattern and its own
    # per-level budget / default Harris ranking. Not bitwise comparable.
    orb = cv2.ORB_create(nfeatures=args.features_per_level * args.levels,
                         scaleFactor=args.scale_factor, nlevels=args.levels,
                         edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE,
                         patchSize=31, fastThreshold=args.fast_threshold)
    t = time.perf_counter()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    ms = (time.perf_counter() - t) * 1000
    n = len(keypoints)
    return Features(np.asarray([k.pt for k in keypoints], np.float32).reshape(-1, 2),
                    descriptors if descriptors is not None else np.empty((0, 32), np.uint8),
                    np.asarray([math.radians(k.angle) for k in keypoints], np.float32),
                    np.asarray([k.size / 31.0 for k in keypoints], np.float32),
                    np.asarray([k.response for k in keypoints], np.float32),
                    np.asarray([k.octave for k in keypoints], np.int32), ms)


def match_features(a, b):
    """Mutual nearest-neighbour Hamming matches, no geometric filtering."""
    if len(a.descriptors) == 0 or len(b.descriptors) == 0:
        return [], 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    t = time.perf_counter()
    matches = matcher.match(a.descriptors, b.descriptors)
    duration = (time.perf_counter() - t) * 1000
    return sorted(matches, key=lambda m: m.distance), duration


def homography_projection(points, H):
    h = np.column_stack((points, np.ones(len(points), np.float64))) @ H.T
    z = h[:, 2]
    out = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (np.abs(z) > 1e-12)
    out[valid] = h[valid, :2] / z[valid, None]
    return out


def geometric_correct(a, b, matches, H, epsilon):
    if not matches:
        return np.empty((0,), dtype=bool)
    src = np.asarray([a.points[m.queryIdx] for m in matches], np.float64)
    dst = np.asarray([b.points[m.trainIdx] for m in matches], np.float64)
    projected = homography_projection(src, H)
    err = np.linalg.norm(projected - dst, axis=1)
    return np.isfinite(err) & (err <= epsilon)


def repeatable(a, b, H, shape, epsilon, mask=None):
    """Count one-to-one repeated detections among ground-truth-visible points."""
    h, w = shape[:2]
    projected = homography_projection(a.points.astype(np.float64), H)
    inside = np.isfinite(projected).all(axis=1)
    # Evaluate centres within the field of view (not entire descriptor support).
    inside &= (projected[:, 0] >= 0) & (projected[:, 0] < w)
    inside &= (projected[:, 1] >= 0) & (projected[:, 1] < h)
    if mask is not None:
        idx = np.flatnonzero(inside)
        xx = np.clip(np.rint(projected[idx, 0]).astype(int), 0, w - 1)
        yy = np.clip(np.rint(projected[idx, 1]).astype(int), 0, h - 1)
        inside[idx] &= mask[yy, xx] > 0
    eligible = projected[inside]
    if len(eligible) == 0 or len(b.points) == 0:
        return 0, len(eligible)
    from scipy.spatial import cKDTree
    tree = cKDTree(b.points)
    possible = tree.query_ball_point(eligible, epsilon)
    candidate = []
    for i, js in enumerate(possible):
        for j in js:
            err2 = float(np.sum((eligible[i] - b.points[j]) ** 2))
            candidate.append((err2, i, j))
    candidate.sort()
    used_a, used_b = set(), set()
    for _, i, j in candidate:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return len(used_a), len(eligible)


def fundamental_inlier_mask(a, b, matches, epsilon):
    """Estimated epipolar-consistency labels; not ground truth or PnP."""
    if len(matches) < 8:
        return None
    p = np.float32([a.points[m.queryIdx] for m in matches])
    q = np.float32([b.points[m.trainIdx] for m in matches])
    try:
        F, mask = cv2.findFundamentalMat(p, q, cv2.FM_RANSAC, epsilon, 0.99)
    except cv2.error:
        return None
    if F is None or mask is None or mask.size != len(matches):
        return None
    return mask.reshape(-1).astype(bool)


def fundamental_ransac(a, b, matches, epsilon):
    """Estimated F-RANSAC diagnostic only; NOT ground-truth precision."""
    mask = fundamental_inlier_mask(a, b, matches, epsilon)
    if mask is None:
        return None, None
    count = int(mask.sum())
    return count, count / len(matches)


def get_sequences(root, dataset):
    root = Path(root).expanduser().resolve()
    if dataset == 'oxford':
        if (root / 'img1.ppm').exists() or (root / 'img1.png').exists():
            return [root]
        return sorted(p for p in root.iterdir() if p.is_dir() and
                      (p / 'H1to2p').exists())
    if dataset == 'hpatches':
        if (root / 'H_1_2').exists():
            return [root]
        return sorted(p for p in root.iterdir() if p.is_dir() and (p / 'H_1_2').exists())
    raise ValueError(dataset)


def affine_pairs(root, dataset):
    for seq in get_sequences(root, dataset):
        if dataset == 'oxford':
            imgs = {int(re.fullmatch(r'img(\d+)', p.stem).group(1)): p for p in image_paths(seq)
                    if re.fullmatch(r'img(\d+)', p.stem)}
            prefix = lambda i: f'H1to{i}p'
        else:
            imgs = {int(p.stem): p for p in image_paths(seq) if p.stem.isdigit()}
            prefix = lambda i: f'H_1_{i}'
        if 1 not in imgs:
            continue
        for i in sorted(imgs):
            if i == 1:
                continue
            Hpath = seq / prefix(i)
            if not Hpath.exists():
                continue
            H = np.loadtxt(Hpath, dtype=np.float64).reshape(3, 3)
            yield Pair(f'1_to_{i}', seq.name, imgs[1], imgs[i], H)


def tum_frames(root):
    root = Path(root)
    rgb_list = root / 'rgb.txt'
    if rgb_list.exists():
        frames = []
        for line in rgb_list.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) >= 2:
                p = root / fields[1]
                if p.exists():
                    frames.append(p)
        return frames
    folder = root / 'rgb'
    return image_paths(folder if folder.exists() else root)


def kitti_frames(root):
    root = Path(root)
    # Supports odometry root, one sequence, or KITTI raw folder.
    dirs = [root / 'image_0', root / 'image_2', root / 'image_00' / 'data', root]
    for d in dirs:
        if d.is_dir():
            frames = image_paths(d)
            if frames:
                return frames
    return []


def temporal_pairs(frames, sequence, step):
    for i in range(0, len(frames) - step, step):
        yield Pair(f'{i:06d}_to_{i + step:06d}', sequence, frames[i], frames[i + step])


def synth_cases(img):
    h, w = img.shape[:2]
    centre = ((w - 1) / 2, (h - 1) / 2)
    identity = np.eye(3, dtype=np.float64)
    def warp(name, H):
        dest = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        support = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (w, h),
                                      flags=cv2.INTER_NEAREST)
        return name, dest, H, support
    for deg in (15, 45, 90):
        H = np.vstack([cv2.getRotationMatrix2D(centre, deg, 1), [0, 0, 1]])
        yield warp(f'rotation_{deg}', H)
    for scale in (0.65, 1.5):
        H = np.vstack([cv2.getRotationMatrix2D(centre, 0, scale), [0, 0, 1]])
        yield warp(f'scale_{str(scale).replace(".", "p")}', H)
    for strength in (0.5, 1.5):
        out = np.clip(img.astype(np.float32) * strength, 0, 255).astype(np.uint8)
        yield f'brightness_{str(strength).replace(".", "p")}', out, identity, None
    for sigma in (1.5, 3.0):
        yield f'blur_{str(sigma).replace(".", "p")}', cv2.GaussianBlur(img, (0, 0), sigma), identity, None
    rng = np.random.default_rng(12345)
    noise = rng.normal(0, 15, img.shape)
    yield 'noise_std15', np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8), identity, None
    good, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 35])
    if good:
        yield 'jpeg_q35', cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE), identity, None
    # Gentle projective distortion, with complete ground-truth homography.
    margin = 0.09
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[w*margin, h*margin], [w*(1-margin), h*.02],
                      [w*(1-2*margin), h*(1-margin)], [w*margin*.5, h*(1-2*margin)]])
    H = cv2.getPerspectiveTransform(src, dst)
    yield warp('perspective', H)


def synthetic_pairs(img_files):
    for src in img_files:
        img = read_gray(src)
        for name, dest, H, mask in synth_cases(img):
            yield Pair(name, src.stem, src, src, H, dest, mask)


def sequences_for(args):
    if args.dataset in ('oxford', 'hpatches'):
        yield from affine_pairs(args.root, args.dataset)
    elif args.dataset == 'tum':
        root = Path(args.root)
        seq_roots = ([root] if (root / 'rgb.txt').exists() or (root / 'rgb').is_dir()
                     else sorted(d for d in root.iterdir() if d.is_dir() and
                                 ((d / 'rgb.txt').exists() or (d / 'rgb').is_dir())))
        for seq in seq_roots:
            yield from temporal_pairs(tum_frames(seq), seq.name, args.step)
    elif args.dataset == 'kitti':
        root = Path(args.root)
        seq_roots = sorted((root / 'sequences').iterdir()) if (root / 'sequences').is_dir() else [root]
        for seq in seq_roots:
            if seq.is_dir():
                yield from temporal_pairs(kitti_frames(seq), seq.name, args.step)
    elif args.dataset == 'folder':
        yield from temporal_pairs(image_paths(args.root), Path(args.root).name, args.step)
    elif args.dataset == 'synthetic':
        files = [args.image] if args.image else image_paths(args.root, recursive=False)
        if args.image is None and args.max_source_images:
            files = files[:args.max_source_images]
        if not files:
            raise ValueError('No source images found: pass --image photo.png or --root image_folder')
        yield from synthetic_pairs(files)
    else:
        raise ValueError(args.dataset)



def plot_outputs(rows, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('WARNING: matplotlib missing, skipping plots')
        return
    if not rows:
        return
    metrics = [('matches', 'Mutual Hamming matches'),
               ('match_precision', 'Matching precision (homography GT)'),
               ('repeatability', 'Keypoint repeatability'),
               ('fundamental_ransac_inlier_ratio', 'Estimated F-RANSAC inlier ratio'),
               ('extract_1_ms', 'Extraction time (ms), image 1')]
    plot_dir = out / 'plots'
    plot_dir.mkdir(exist_ok=True)
    for key, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        has_data = False
        for extr in ('scratch', 'opencv'):
            subset = [r for r in rows if r['extractor'] == extr and r[key] != '']
            if not subset:
                continue
            ax.plot(np.arange(len(subset)), [float(r[key]) for r in subset],
                    marker='.', markersize=5, linewidth=1.2, label=extr)
            has_data = True
        if has_data:
            ax.set_xlabel('Pair index (see pairs.csv for sequence and image IDs)')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f'{key}.png', dpi=150)
        plt.close(fig)


def write_summary(rows, out):
    keys = ['extractor', 'pairs', 'mean_keypoints_1', 'mean_keypoints_2',
            'mean_matches', 'mean_correct_matches', 'mean_match_precision',
            'mean_repeatability', 'mean_fundamental_ransac_inlier_ratio',
            'mean_extract_1_ms', 'mean_extract_2_ms', 'mean_match_ms']
    summaries = []
    for method in sorted(set(r['extractor'] for r in rows)):
        matching = [r for r in rows if r['extractor'] == method]
        d = {'extractor': method, 'pairs': len(matching)}
        for key in keys[2:]:
            col = key.removeprefix('mean_')
            val = [float(r[col]) for r in matching if r[col] != '']
            d[key] = round(float(np.mean(val)), 5) if val else ''
        summaries.append(d)
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('dataset', choices=['synthetic', 'oxford', 'hpatches', 'tum', 'kitti', 'folder'])
    p.add_argument('--root', type=Path, help='Extracted dataset root or image directory')
    p.add_argument('--image', type=Path, help='Source image for synthetic transformations')
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--binary', type=Path, default=(Path(__file__).resolve().parent / 'bin' / 'orb_extract'))
    p.add_argument('--extractor', choices=['both', 'scratch', 'opencv'], default='both')
    p.add_argument('--levels', type=int, default=8)
    p.add_argument('--scale-factor', type=float, default=1.2)
    p.add_argument('--fast-threshold', type=int, default=30)
    p.add_argument('--features-per-level', type=int, default=500)
    p.add_argument('--cell-size', type=int, default=0)
    p.add_argument('--min-threshold', type=int, default=0)
    p.add_argument('--epsilon', type=float, default=3.0, help='Ground truth reprojection tolerance (pixels)')
    p.add_argument('--step', type=int, default=3, help='TUM / KITTI / folder: temporal frame stride')
    p.add_argument('--max-pairs', type=int, default=20, help='Maximum pairs processed, 0 for all')
    p.add_argument('--max-source-images', type=int, default=3, help='Synthetic: maximum source images in --root (0 for all)')
    p.add_argument('--visualizations', type=int, default=3,
                   help='Number of frame pairs with dashboard and match visualizations (-1 for ALL; 0 disables)')
    p.add_argument('--skip-dashboard-pngs', action='store_true',
                   help='Do not create large composite PNGs; HTML and match JPEGs are still saved')
    p.add_argument('--viz-max-matches', type=int, default=55,
                   help='Maximum spatially separated correspondences DRAWN per pair; all matches still evaluated')
    p.add_argument('--viz-min-spacing', type=int, default=7,
                   help='Minimum approximate marker spacing in pixels, VISUALIZATION ONLY')
    p.add_argument('--viz-marker-radius', type=int, default=2,
                   help='Radius of the small displayed keypoint dots, VISUALIZATION ONLY')
    p.add_argument('--max-hamming-distance', type=int, default=0,
                   help='Optional descriptor-distance cutoff, 0 disables filtering')
    args = p.parse_args()
    if args.dataset != 'synthetic' and args.root is None:
        p.error('--root is required for this dataset')
    if args.dataset == 'synthetic' and args.image is None and args.root is None:
        p.error('Synthetic requires --image or --root')
    if args.step < 1 or args.max_pairs < 0 or args.epsilon <= 0:
        p.error('--step >= 1, --max-pairs >= 0 and --epsilon > 0 required')
    args.binary = args.binary.expanduser().resolve()
    if args.extractor != 'opencv' and not args.binary.is_file():
        p.error(f'Executable not found: {args.binary}. Build it with cmake first.')
    if args.viz_max_matches < 0 or args.viz_min_spacing < 0 or args.viz_marker_radius < 1:
        p.error('Visualization options must be non-negative, and marker radius >= 1')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'visualizations').mkdir(exist_ok=True)
    (args.output / 'dashboards').mkdir(exist_ok=True)
    with (args.output / 'settings.json').open('w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    methods = ['opencv', 'scratch'] if args.extractor == 'both' else [args.extractor]
    cache = {}
    all_rows = []
    gallery_items = []
    pairs_iter = sequences_for(args)
    with tempfile.TemporaryDirectory(prefix='orb_benchmark_') as folder:
        tmp = Path(folder)
        for pair_idx, pair in enumerate(pairs_iter):
            if args.max_pairs and pair_idx >= args.max_pairs:
                break
            im1 = read_gray(pair.first)
            im2 = pair.second_image if pair.second_image is not None else read_gray(pair.second)
            print(f'[{pair_idx + 1}] {pair.sequence}: {pair.name}', flush=True)
            pair_panels = []
            png_panels = []
            for method in methods:
                def extract(image, path, second_generated=False):
                    key = (method, str(path)) if not second_generated else None
                    if key is not None and key in cache:
                        return cache[key]
                    item = scratch_extract(image, args, tmp) if method == 'scratch' else opencv_extract(image, args)
                    if key is not None:
                        cache[key] = item
                    return item
                a = extract(im1, pair.first)
                b = extract(im2, pair.second, pair.second_image is not None)
                matches, match_ms = match_features(a, b)
                raw_match_count = len(matches)
                if args.max_hamming_distance > 0:
                    matches = [m for m in matches if m.distance <= args.max_hamming_distance]
                if pair.homography is not None:
                    correct = geometric_correct(a, b, matches, pair.homography, args.epsilon)
                    nrepeat, neligible = repeatable(a, b, pair.homography, im2.shape, args.epsilon, pair.mask)
                    num_correct = int(correct.sum())
                    precision = num_correct / len(matches) if matches else None
                    reprate = nrepeat / neligible if neligible else None
                    f_inliers = f_fraction = None
                    class_mask = correct
                    geom = 'homography_ground_truth'
                else:
                    correct = None
                    nrepeat = neligible = None
                    num_correct = precision = reprate = None
                    class_mask = fundamental_inlier_mask(a, b, matches, 1.5)
                    f_inliers = int(class_mask.sum()) if class_mask is not None else None
                    f_fraction = f_inliers / len(matches) if class_mask is not None else None
                    geom = 'estimated_fundamental_matrix_only'
                row = dict(dataset=args.dataset, sequence=pair.sequence, pair=pair.name,
                           extractor=method, image_1=str(pair.first), image_2=str(pair.second),
                           keypoints_1=len(a.points), keypoints_2=len(b.points),
                           raw_matches=raw_match_count, matches=len(matches), correct_matches=num_correct,
                           match_precision=precision, repeatable_keypoints=nrepeat,
                           eligible_keypoints=neligible, repeatability=reprate,
                           fundamental_ransac_inliers=f_inliers,
                           fundamental_ransac_inlier_ratio=f_fraction,
                           extract_1_ms=a.milliseconds, extract_2_ms=b.milliseconds,
                           match_ms=match_ms, geometry=geom)
                all_rows.append({k: '' if v is None else v for k, v in row.items()})
                print(f'  {method:8} kp={len(a.points)}/{len(b.points)}, matches={len(matches)}, '
                      f'precision={precision if precision is not None else "N/A"}, '
                      f'repeatability={reprate if reprate is not None else "N/A"}', flush=True)
                if args.visualizations < 0 or pair_idx < args.visualizations:
                    img_name = f'{pair_idx:03d}_{method}_{safe_name(pair.name)}.jpg'
                    img_path = args.output / 'visualizations' / img_name
                    canvas, drawn = draw_correspondences(
                        im1, im2, a, b, matches, class_mask,
                        max_matches=args.viz_max_matches, spacing=args.viz_min_spacing,
                        marker_radius=args.viz_marker_radius)
                    if not cv2.imwrite(str(img_path), canvas):
                        raise RuntimeError(f'Could not save {img_path}')
                    png_panels.append((method, row, img_path, drawn))
                    pair_panels.append(render_panel(
                        method, row, f'../visualizations/{img_name}', drawn,
                        args.dataset, pair.sequence, pair.name, args))
            if pair_panels:
                html_name = f'{pair_idx:03d}_{safe_name(pair.sequence)}_{safe_name(pair.name)}.html'
                write_pair_dashboard(args.output / 'dashboards' / html_name, pair_panels,
                                     args.dataset, pair.sequence, pair.name, args)
                png_name = Path(html_name).with_suffix('.png').name
                if not args.skip_dashboard_pngs:
                    write_dashboard_png(args.output / 'dashboards' / png_name, png_panels,
                                        args.dataset, pair.sequence, pair.name, args)
                gallery_items.append((html_name, f'{pair.sequence} / {pair.name}'))
    if not all_rows:
        raise SystemExit('No valid image pairs found. See README for expected directory layout.')
    with (args.output / 'pairs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    write_gallery(args.output / 'dashboards' / 'index.html', gallery_items, args.dataset)
    summaries = write_summary(all_rows, args.output)
    plot_outputs(all_rows, args.output)
    print('\nSummary:')
    for item in summaries:
        print(json.dumps(item, indent=2))
    print(f'\nSaved: {args.output.resolve()} (pairs.csv, summary.csv, plots/, visualizations/, dashboards/index.html, settings.json)')


if __name__ == '__main__':
    main()
