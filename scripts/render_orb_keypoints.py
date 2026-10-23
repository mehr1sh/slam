#!/usr/bin/env python3
"""Draw actual exported feature positions. No feature detection is done here.
Spatial overlap is a visualization measure, not a descriptor or geometry test.
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PROJECT='#26b4ff'; ORB='#ff9b26'; SHARED='#72e396'

def read(path):
    with path.open() as f: return list(csv.DictReader(f))

def points(rows):
    return np.array([[float(r['x']),float(r['y'])] for r in rows]).reshape(-1,2)

def compare(a,b,tolerance):
    xy_a,xy_b=points(a),points(b)
    la=np.array([int(r['level']) for r in a]);lb=np.array([int(r['level']) for r in b])
    edges=[]
    for level in range(8):
        ia=np.flatnonzero(la==level);ib=np.flatnonzero(lb==level)
        if not len(ia) or not len(ib):continue
        dist=((xy_a[ia,None,:]-xy_b[None,ib,:])**2).sum(2)
        aa,bb=np.where(dist<=tolerance**2)
        edges.extend((float(dist[x,y]),int(ia[x]),int(ib[y])) for x,y in zip(aa,bb))
    used_a=set();used_b=set();pairs=[]
    # Deterministic greedy pairing by ascending image distance, then array IDs.
    for squared,i,j in sorted(edges):
        if i not in used_a and j not in used_b:
            used_a.add(i);used_b.add(j);pairs.append((i,j,float(np.sqrt(squared))))
    return pairs,sorted(set(range(len(a)))-used_a),sorted(set(range(len(b)))-used_b)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frames',type=int,nargs='+',default=[0,15])
    p.add_argument('--tolerance',type=float,default=2.)
    p.add_argument('--results',type=Path,default=ROOT/'outputs/orb_benchmark')
    args=p.parse_args();dest=args.results/'keypoint_visuals';dest.mkdir(exist_ok=True,parents=True)
    summaries=[]
    selections=[(frame,None) for frame in args.frames]
    if 15 in args.frames: selections.append((15,0))
    for frame,level_filter in selections:
        data={m:read(args.results/'keypoints'/f'frame_{frame:06d}_{m}.csv') for m in ['project_orb_style','opencv_orb']}
        if level_filter is not None:
            data={m:[r for r in rows if int(r['level'])==level_filter] for m,rows in data.items()}
        stem=f'frame_{frame:06d}'+(f'_level{level_filter}' if level_filter is not None else '')
        a,b=data.values();xy_a,xy_b=points(a),points(b)
        pairs,only_a,only_b=compare(a,b,args.tolerance)
        im=np.asarray(Image.open(ROOT/f'data/synthetic_bunny/{frame:06d}.png').convert('RGB'))
        xy=np.concatenate([xy_a,xy_b]);lo=np.floor(xy.min(0)-14).astype(int);hi=np.ceil(xy.max(0)+14).astype(int)
        lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[im.shape[1],im.shape[0]])
        bounds=(int(lo[0]),int(hi[0]),int(lo[1]),int(hi[1]))
        def axis(ax):
            ax.imshow(im,interpolation='nearest')
            ax.set(xlim=bounds[:2],ylim=(bounds[3],bounds[2]));ax.axis('off')
        def scatter(ax,xy,color,marker='o',size=6,alpha=.85):
            if len(xy):ax.scatter(xy[:,0],xy[:,1],s=size,facecolors='none' if marker=='o' else color,
                                  edgecolors=color,linewidths=.55,alpha=alpha,marker=marker)
        def panel(ax,key):
            axis(ax)
            if key=='project':scatter(ax,xy_a,PROJECT)
            if key=='opencv':scatter(ax,xy_b,ORB)
            if key=='difference':
                # One midpoint per close, same-level pair; original points are in CSVs.
                common=np.array([(xy_a[i]+xy_b[j])*.5 for i,j,d in pairs]).reshape(-1,2)
                scatter(ax,common,SHARED,size=5,alpha=.65)
                scatter(ax,xy_a[only_a],PROJECT,size=9)
                scatter(ax,xy_b[only_b],ORB,marker='^',size=10)
        for key in ['project','opencv','difference']:
            fig=plt.figure(figsize=(5.4,5.4));ax=fig.add_axes([0,0,1,1]);panel(ax,key)
            fig.savefig(dest/f'{stem}_{key}.png',dpi=200);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,6),facecolor='white')
        titles=[f'Our extractor: {len(a)} keypoints',f'OpenCV ORB: {len(b)} keypoints',f'Shared: {len(pairs)} pairs; unique: {len(only_a)} / {len(only_b)}']
        for ax,key,title in zip(axes,['project','opencv','difference'],titles):panel(ax,key);ax.set_title(title,fontsize=13)
        fig.suptitle(f'Actual feature locations: frame {frame}'+(f', level {level_filter}' if level_filter is not None else ', all levels'),fontsize=17)
        fig.text(.5,.08,'Blue circles: ours only   Orange triangles: OpenCV only   Green: spatially paired locations',ha='center',fontsize=12)
        fig.text(.5,.035,f'Same crop and scale. Greedy one-to-one pairs within {args.tolerance:g} original-image pixels at the same pyramid level. This tests spatial overlap only.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.02,right=.98,top=.86,bottom=.15,wspace=.04)
        fig.savefig(dest/f'{stem}_comparison.png',dpi=170);plt.close(fig)
        with (dest/f'{stem}_spatial_pairs.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['project_index','opencv_index','distance_pixels','level'])
            for i,j,d in pairs:w.writerow([i,j,d,a[i]['level']])
        with (dest/f'{stem}_unpaired.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['method','index','x','y','level'])
            for method,ids,rows in [('project',only_a,a),('opencv',only_b,b)]:
                for i in ids:w.writerow([method,i,rows[i]['x'],rows[i]['y'],rows[i]['level']])
        summary=dict(frame=frame,level_filter=level_filter,stem=stem,project_count=len(a),opencv_count=len(b),spatial_pairs=len(pairs),
                     project_unpaired=len(only_a),opencv_unpaired=len(only_b),tolerance_pixels=args.tolerance,
                     criterion='greedy one-to-one, same pyramid level, original-image coordinates',crop=list(bounds),
                     project_levels=[sum(int(r['level'])==l for r in a) for l in range(8)],
                     opencv_levels=[sum(int(r['level'])==l for r in b) for l in range(8)])
        assert len(pairs)+len(only_a)==len(a) and len(pairs)+len(only_b)==len(b)
        summaries.append(summary);print(json.dumps(summary))
    (dest/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print('Saved actual C++ feature overlays and spatial comparison evidence:',dest)
if __name__=='__main__':main()
