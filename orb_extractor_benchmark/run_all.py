#!/usr/bin/env python3
"""Run installed ORB datasets and create a unified offline HTML report.

Examples:
  python run_all.py --auto-datasets datasets --output results --max-pairs 0 --visualizations -1
  python run_all.py --oxford datasets/oxford --hpatches datasets/hpatches/hpatches-sequences-release --output results --max-pairs 0 --visualizations -1
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

from build_report import assemble


def has_sequence(root, marker):
    if not root.is_dir():return False
    if (root/marker).exists():return True
    return any(p.is_dir() and (p/marker).exists() for p in root.iterdir())


def find_root(root, ds):
    """Find the actual extracted root under datasets/{ds}, allowing one extra folder."""
    if not root.is_dir():return None
    markers={'oxford':'H1to2p','hpatches':'H_1_2','tum':'rgb.txt'}
    if ds in markers:
        marker=markers[ds]
        if has_sequence(root,marker):return root
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if has_sequence(d,marker):return d
    if ds=='kitti':
        for p in (root/'dataset',root):
            if (p/'sequences').is_dir():return p
        # Fall back to single KITTI sequence.
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if (d/'image_0').is_dir() or (d/'image_2').is_dir():return d
    if ds=='folder':
        from benchmark import image_paths
        if image_paths(root):return root
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if image_paths(d):return d
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--auto-datasets',type=Path, help='Automatically find Oxford, HPatches, TUM, KITTI and custom inside this directory')
    p.add_argument('--synthetic-image',type=Path,help='Optional synthetic source image; not required in automatic mode')
    p.add_argument('--oxford',type=Path)
    p.add_argument('--hpatches',type=Path)
    p.add_argument('--tum',type=Path)
    p.add_argument('--kitti',type=Path)
    p.add_argument('--custom',type=Path)
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--levels',type=int,default=8)
    p.add_argument('--features-per-level',type=int,default=150)
    p.add_argument('--fast-threshold',type=int,default=30)
    p.add_argument('--max-pairs',type=int,default=0,help='Per dataset, 0 = ALL pairs')
    p.add_argument('--step',type=int,default=3)
    p.add_argument('--visualizations',type=int,default=-1,help='-1 = EVERY evaluated pair (0 disables)')
    p.add_argument('--viz-max-matches',type=int,default=30)
    p.add_argument('--viz-min-spacing',type=int,default=12)
    p.add_argument('--viz-marker-radius',type=int,default=2)
    p.add_argument('--epsilon',type=float,default=3.0)
    p.add_argument('--max-hamming-distance',type=int,default=0)
    p.add_argument('--skip-dashboard-pngs',action='store_true',default=False,
                   help='Save disk space: HTML & JPEG visualizations only (recommended with --visualizations -1)')
    p.add_argument('--strict',action='store_true',help='Stop at first failing dataset')
    args=p.parse_args()
    selected=[]
    if args.auto_datasets is not None:
        dsroot=args.auto_datasets.expanduser().resolve()
        if not dsroot.is_dir():p.error(f'No datasets directory: {dsroot}')
        for ds,folder in [('oxford','oxford'),('hpatches','hpatches'),('tum','tum'),
                          ('kitti','kitti'),('folder','custom')]:
            found=find_root(dsroot/folder, ds)
            if found:
                selected.append((ds,found))
                print(f'Found {ds:10s}: {found}',flush=True)
            else:
                print(f'Skipping {folder}: no recognized extracted dataset found',flush=True)
    # Explicitly supplied paths take precedence over automatic discovery.
    for ds in ('oxford','hpatches','tum','kitti'):
        path=getattr(args,ds)
        if path is not None:
            selected=[(d,r) for d,r in selected if d!=ds]
            selected.append((ds,path))
    if args.custom is not None:
        selected=[(d,r) for d,r in selected if d!='folder']
        selected.append(('folder',args.custom))
    if args.synthetic_image is not None:
        selected.insert(0,('synthetic',args.synthetic_image))
    if not selected and args.auto_datasets is None:
        # Preserve the old runner's synthetic-only convenience.
        selected=[('synthetic',Path(__file__).resolve().parent/'sample_data'/'bunny.png')]
    if not selected:
        print('No usable datasets detected. Existing reports can still be indexed with build_report.py.')
        sys.exit(1)
    out=args.output.expanduser().resolve()
    out.mkdir(exist_ok=True,parents=True)
    summary=[]; failures=[]
    for ds,root in selected:
        suffix='custom' if ds=='folder' else ds
        print(f'\n===== {suffix.upper()} =====',flush=True)
        cmd=[sys.executable,str(Path(__file__).resolve().parent/'benchmark.py'),ds,
             '--output',str(out/suffix),'--levels',str(args.levels),
             '--features-per-level',str(args.features_per_level),
             '--fast-threshold',str(args.fast_threshold),
             '--max-pairs',str(args.max_pairs),'--step',str(args.step),
             '--visualizations',str(args.visualizations),
             '--viz-max-matches',str(args.viz_max_matches),
             '--viz-min-spacing',str(args.viz_min_spacing),
             '--viz-marker-radius',str(args.viz_marker_radius),
             '--epsilon',str(args.epsilon),
             '--max-hamming-distance',str(args.max_hamming_distance)]
        if args.skip_dashboard_pngs:cmd.append('--skip-dashboard-pngs')
        if ds=='synthetic':cmd+=['--image',str(root)]
        else:cmd+=['--root',str(root)]
        process=subprocess.run(cmd,check=False)
        if process.returncode:
            failures.append(suffix)
            print(f'WARNING: {suffix} failed (status {process.returncode}); continuing.',flush=True)
            if args.strict:break
            continue
        with (out/suffix/'summary.csv').open(newline='') as f:
            for row in csv.DictReader(f):summary.append({'dataset':suffix,**row})
    if summary:
        with (out/'all_datasets_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    assemble(out,out/'index.html')
    print(f'\nOPEN MASTER HTML:\nxdg-open "{out / "index.html"}"',flush=True)
    if failures:print('Failed datasets: '+', '.join(failures),file=sys.stderr)
    if failures:sys.exit(1)


if __name__=='__main__':main()
