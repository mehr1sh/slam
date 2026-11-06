#!/usr/bin/env python3
"""Render C++ benchmark exports. No pose/feature estimation in Python."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'results/reports/figures/orb_benchmark')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/synthetic_bunny')
    args=parser.parse_args();out=args.results
    if not (out/'run_complete.txt').is_file():
        raise SystemExit('Run orb_benchmark first.')
    summary=read(out/'summary.csv');frames=read(out/'frames.csv')
    if len(summary)!=2: raise ValueError('Expected two benchmark methods')
    labels=['Project ORB-style','OpenCV ORB'];colors=['#0077bb','#ee7733']
    specs=[('mean_keypoints','Mean keypoints / frame',1),
           ('mean_matches','Mean mutual ratio matches / pair',1),
           ('mean_ransac_inliers','Mean RANSAC inliers / pair',1),
           ('gt_precision_3d','GT-correct / valid-depth matches (%)',100),
           ('median_extraction_ms','Median extraction time (ms)',1),
           ('ATE_RMSE_m','Trajectory RMSE without loop closure (mm)',1000)]
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,(key,title,scale) in zip(axes.flat,specs):
        vals=[float(r[key])*scale if r[key] else np.nan for r in summary]
        bars=ax.bar(labels,vals,color=colors)
        ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_title(title,fontsize=10)
        ax.set_ylim(0,max([v for v in vals if np.isfinite(v)]+[1])*1.2)
        ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('ORB frontend benchmark: identical matcher and GN ICP backend')
    fig.text(.5,.018,'Requested feature budget = 1400; actual counts differ. GT is used only for evaluation. Timings depend on hardware.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'01_benchmark_summary.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r,label,color in zip(summary,labels,colors):
        rows=[f for f in frames if f['method']==r['method']]
        axes[0].plot([int(f['frame']) for f in rows],[float(f['keypoints']) for f in rows],label=label,color=color)
        axes[1].plot([int(f['frame']) for f in rows],[float(f['extraction_median_ms']) for f in rows],label=label,color=color)
    axes[0].set(xlabel='Frame',ylabel='Keypoints',title='Actual retained features')
    axes[1].set(xlabel='Frame',ylabel='Milliseconds',title='Extraction, median of repeated runs')
    for ax in axes: ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'02_counts_and_timing.png',dpi=160);plt.close(fig)
    notes=['# ORB benchmark results','',
           'OpenCV ORB is compared with the production project ORB-style extractor. Both use the same project Hamming matcher and GN ICP backend; this is not ORB-SLAM.', '',
           '| Method | Solved pairs | Mean features | GT precision | ATE (mm) | Extraction (ms) |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r,label in zip(summary,labels):
        precision=f"{100*float(r['gt_precision_3d']):.2f}%" if r['gt_precision_3d'] else 'N/A'
        ate=f"{1000*float(r['ATE_RMSE_m']):.3f}" if r['ATE_RMSE_m'] else 'N/A'
        notes.append(f"| {label} | {r['solved_pairs']}/{r['pairs']} | {float(r['mean_keypoints']):.1f} | {precision} | {ate} | {float(r['median_extraction_ms']):.3f} |")
    notes+=['','See protocol.txt for all controls, timing definitions, remaining implementation differences, and failure handling.','']
    (out/'RESULTS.md').write_text('\n'.join(notes))
    print(out/'RESULTS.md')
if __name__=='__main__': main()
