#!/usr/bin/env python3
"""Extract scratch ORB from any image and save keypoints, descriptors and a preview."""
import argparse
import csv
from pathlib import Path
import tempfile
import cv2
from benchmark import read_gray, scratch_extract, opencv_extract
from visualization import draw_points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('--output', type=Path, default=Path('image_features'))
    p.add_argument('--binary', type=Path, default=Path(__file__).resolve().parent / 'bin' / 'orb_extract')
    p.add_argument('--extractor', choices=('scratch', 'opencv', 'both'), default='scratch')
    p.add_argument('--levels', type=int, default=8)
    p.add_argument('--features-per-level', type=int, default=500)
    p.add_argument('--fast-threshold', type=int, default=30)
    p.add_argument('--scale-factor', type=float, default=1.2)
    p.add_argument('--cell-size', type=int, default=0)
    p.add_argument('--min-threshold', type=int, default=0)
    p.add_argument('--viz-max-keypoints', type=int, default=140,
                   help='Number of spatially separated dots DRAWN (all features remain in CSV)')
    p.add_argument('--viz-min-spacing', type=int, default=6)
    p.add_argument('--viz-marker-radius', type=int, default=2)
    args = p.parse_args()
    args.binary = args.binary.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    gray = read_gray(args.image)
    methods = ('scratch', 'opencv') if args.extractor == 'both' else (args.extractor,)
    with tempfile.TemporaryDirectory(prefix='orb_image_') as work:
        for method in methods:
            features = (scratch_extract(gray, args, Path(work)) if method == 'scratch'
                        else opencv_extract(gray, args))
            path = args.output / f'{method}_features.csv'
            with path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'angle_rad', 'scale', 'score', 'level', 'descriptor_hex'])
                for i in range(len(features.points)):
                    writer.writerow([float(features.points[i, 0]), float(features.points[i, 1]),
                                     float(features.angles[i]), float(features.scales[i]),
                                     float(features.responses[i]), int(features.levels[i]),
                                     features.descriptors[i].tobytes().hex()])
            preview, drawn = draw_points(gray, features.points,
                                         max_points=args.viz_max_keypoints,
                                         spacing=args.viz_min_spacing,
                                         radius=args.viz_marker_radius)
            target = args.output / f'{method}_keypoints.jpg'
            cv2.imwrite(str(target), preview)
            details = (f'Extractor: {method}\n'
                       f'Detected keypoints (all): {len(features.points)}\n'
                       f'Displayed dots (spatially separated): {drawn}\n'
                       f'Extraction time: {features.milliseconds:.2f} ms\n'
                       f'Pyramid levels: {args.levels}\n'
                       f'Features per level setting: {args.features_per_level}\n'
                       f'Descriptors: {len(features.descriptors)} x 256 bits\n'
                       f'Note: Display-only thinning does not remove features from CSV.\n'
                       f'For correspondences + inlier/outlier descriptions, run benchmark.py.\n')
            (args.output / f'{method}_description.txt').write_text(details)
            print(f'{method}: {len(features.points)} keypoints, {drawn} visualized, '
                  f'{features.milliseconds:.2f} ms -> {path}, {target}, description.txt')


if __name__ == '__main__':
    main()
