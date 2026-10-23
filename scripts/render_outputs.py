#!/usr/bin/env python3
"""Plot measurements exported by C++; no feature or pose estimation here.
PNG images are primary outputs; CSVs preserve every candidate and decision.
Ubuntu dependencies: python3-matplotlib python3-pil python3-numpy.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
except ImportError as e:
    raise SystemExit('PNG export needs plotting packages. Run: sudo apt install '
                     'python3-matplotlib python3-pil python3-numpy. '
                     'Use the Python environment containing those packages.') from e

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
STAGES = ['00_fast_all', '01_harris_all', '02_fast_quotas', '03_harris_quotas']
LABELS = ['Before: FAST ranking, no quotas', 'Harris ranking, no quotas',
          'FAST ranking + quotas (control)', 'After: Harris ranking + quotas']
COLORS = ['#0077bb', '#ee7733', '#994499', '#009988', '#cc3311', '#aa8800', '#ee3377', '#558833']

def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))

def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def save(fig, path, note=''):
    if note: fig.text(.5, .018, note, ha='center', fontsize=9)
    handles,labels=fig.axes[0].get_legend_handles_labels() if fig.axes else ([],[])
    level_legend=labels==[f'L{k}' for k in range(8)]
    if level_legend: fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.043),ncol=8,fontsize=8)
    fig.tight_layout(rect=(0, .11 if level_legend else .05 if note else 0, 1, .95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor='white'); plt.close(fig)

def crop(rows, width, height):
    if not rows: return 0, width, 0, height
    xy = np.array([[float(r['x']), float(r['y'])] for r in rows])
    lo = np.maximum(xy.min(0) - 16, 0); hi = np.minimum(xy.max(0) + 16, [width, height])
    return lo[0], hi[0], lo[1], hi[1]

def feature_view(ax, image, rows, bounds, title, label=False):
    ax.imshow(image)
    for level in range(8):
        points = [r for r in rows if int(r['level']) == level]
        ax.scatter([float(r['x']) for r in points], [float(r['y']) for r in points],
                   s=5, color=COLORS[level], alpha=.65, linewidths=0,label=f'L{level}')
    if label:
        for r in rows[:12]:
            ax.annotate(r['index'], (float(r['x']), float(r['y'])), xytext=(3, 3),
                        textcoords='offset points', fontsize=7, color='black',
                        bbox=dict(fc='white', alpha=.8, edgecolor='none', pad=.5))
    x0, x1, y0, y1 = bounds
    ax.set(xlim=(x0, x1), ylim=(y1, y0), title=f'{title}\n{len(rows)} candidates'); ax.set_axis_off()

def frontend(i, j):
    dest = OUT / 'frontend' / f'pair_{i}_{j}'; frames = [i, j]; all_data = {}; mapping = []
    for frame in frames:
        image = np.asarray(Image.open(ROOT / f'data/synthetic_bunny/{frame:06d}.png').convert('RGB'))
        stages = [read(dest / f'frame_{frame:06d}_{s}.csv') for s in STAGES]
        all_data[frame] = (image, stages); bounds = crop(stages[0], image.shape[1], image.shape[0])
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Frame {frame}: feature selection at identical detector settings')
        for ax, rows, title in zip(axes.flat, stages, LABELS):
            feature_view(ax, image, rows, bounds, title, label=True)
        save(fig, dest / f'frame_{frame:06d}_00_feature_stages.png',
             'Colours = levels 0-7. Numbers = first 12 array indices. Harris alone reorders the same points.')
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)); fig.suptitle(f'Frame {frame}: Harris ranking before / after')
        for ax, rows, name in zip(axes, stages[:2], ['FAST order', 'Harris order']):
            feature_view(ax, image, rows, bounds, name, label=True)
        save(fig, dest / f'frame_{frame:06d}_01_harris_ranking_before_after.png',
             'Same pixel positions and descriptors; sorting changes array order. No candidates are removed here.')
        fq = {r['stable_id']: r['index'] for r in stages[2]}; hq = {r['stable_id']: r['index'] for r in stages[3]}
        for r in stages[0]:
            mapping.append(dict(frame=frame, stable_id=r['stable_id'], fast_order=r['fast_order'],
                harris_order=r['harris_order'], fast_quota_index=fq.get(r['stable_id'], ''),
                actual_harris_quota_index=hq.get(r['stable_id'], ''), harris_response=r['harris_response']))
        top_ids = list(dict.fromkeys(r['stable_id'] for r in stages[0][:10] + stages[1][:10])); lookup = {r['stable_id']: r for r in stages[0]}
        cells = [[uid, lookup[uid]['fast_order'], lookup[uid]['harris_order'], hq.get(uid, 'removed')] for uid in top_ids]
        fig, ax = plt.subplots(figsize=(11, 7)); ax.axis('off'); fig.suptitle(f'Frame {frame}: same keypoint ID, changing array index')
        table = ax.table(cellText=cells, colLabels=['Frame-local candidate ID', 'FAST index', 'Harris index', 'Final index'],
                         colWidths=[.49, .16, .16, .19], loc='center', cellLoc='left')
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.5)
        save(fig, dest / f'frame_{frame:06d}_identifier_changes.png',
             'Shown: union of first 10 FAST and Harris candidates. Every candidate is in identifier_mapping.csv.')
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)); fig.suptitle(f'Frame {frame}: effect of pyramid quotas')
        for ax, rows, title in zip(axes, [stages[1], stages[3]], ['Before quotas', 'After quotas']):
            feature_view(ax, image, rows, bounds, title)
        save(fig, dest / f'frame_{frame:06d}_02_quotas_before_after.png',
             'Harris ranking is enabled on both sides. Budget = 1400; unused level quotas are not redistributed.')
        fig, axes = plt.subplots(1, 3, figsize=(16, 5)); fig.suptitle(f'Frame {frame}: isolate ranking under identical quotas')
        for ax, rows, name in zip(axes[:2], stages[2:], ['FAST + quotas', 'Harris + quotas']):
            feature_view(ax, image, rows, bounds, name)
        ax = axes[2]; ax.imshow(image)
        groups = {'Both selections': ('#009988', []), 'FAST only': ('#ee7733', []), 'Harris only': ('#0077bb', [])}
        for r in stages[0]:
            a, b = r['stable_id'] in fq, r['stable_id'] in hq
            if a or b: groups['Both selections' if a and b else 'FAST only' if a else 'Harris only'][1].append(r)
        for name, (color, rows) in groups.items():
            ax.scatter([float(r['x']) for r in rows], [float(r['y']) for r in rows], s=8, c=color, linewidths=0, label=f'{name}: {len(rows)}')
        ax.set(xlim=bounds[:2], ylim=(bounds[3], bounds[2]), title='Membership difference'); ax.set_axis_off(); ax.legend(fontsize=8, loc='lower right')
        save(fig, dest / f'frame_{frame:06d}_03_ranking_same_quota.png',
             'Counts can be identical while selected locations differ. This isolates Harris ranking at fixed quotas.')
    write_csv(dest / 'identifier_mapping.csv', mapping)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.suptitle('Eight-level pyramid: candidates, quotas and actual retained features')
    for ax, frame in zip(axes, frames):
        levels = read(dest / f'frame_{frame:06d}_levels.csv'); x = np.arange(len(levels))
        for offset, key, name, color in [(-.26, 'fast_all', 'Before quotas', '#999999'), (0, 'quota', 'Quota limit', '#ee7733'), (.26, 'harris_quota', 'After quotas', '#0077bb')]:
            ax.bar(x+offset, [int(r[key]) for r in levels], .25, label=name, color=color)
        ax.set(title=f'Frame {frame}', xticks=x, xlabel='Pyramid level', ylabel='Keypoints'); ax.legend(fontsize=8)
    save(fig, dest / '04_pyramid_quotas.png', 'A quota is an upper limit. A level with fewer candidates keeps all of them.')
    matches = read(dest / 'matching_decisions.csv')
    sets = {'raw_nearest': matches, 'forward_ratio': [r for r in matches if r['ratio_forward'] == '1'],
            'mutual_final': [r for r in matches if r['accepted'] == '1'], 'rejected': [r for r in matches if r['accepted'] == '0']}
    def match_view(ax, rows, title):
        crops = []
        for frame in frames:
            im, stages = all_data[frame]; x0,x1,y0,y1 = crop(stages[0],im.shape[1],im.shape[0]); crops.append((int(x0),int(x1)+1,int(y0),int(y1)+1))
        sizes = [(b-a,d-c) for a,b,c,d in crops]; offset = sizes[0][0]+20
        canvas = np.full((max(s[1] for s in sizes),offset+sizes[1][0],3),245,dtype=np.uint8)
        for k,frame in enumerate(frames):
            a,b,c,d=crops[k]; x=0 if k==0 else offset; canvas[:d-c,x:x+b-a]=all_data[frame][0][c:d,a:b]
        ax.imshow(canvas); stride=max(1,int(np.ceil(len(rows)/100)))
        for r in rows[::stride]:
            ax.plot([float(r['u_i'])-crops[0][0],float(r['u_j'])-crops[1][0]+offset],
                    [float(r['v_i'])-crops[0][2],float(r['v_j'])-crops[1][2]],
                    color='#009988' if r['accepted']=='1' else '#cc3311',alpha=.45,lw=.6)
        ax.set_title(f'{title}: {len(rows)} total; {len(rows[::stride])} lines shown'); ax.set_axis_off()
    fig,axes=plt.subplots(2,1,figsize=(12,10)); fig.suptitle(f'Frames {i} to {j}: descriptor matching before / after')
    match_view(axes[0],sets['raw_nearest'],'Before: one nearest descriptor per query')
    match_view(axes[1],sets['mutual_final'],'After: forward + reverse ratio and mutual best')
    save(fig,dest/'05_matching_before_after.png','Green = passes descriptor checks; red = rejected. CSV includes all matches; depth and geometry checks follow later.')
    for key,rows in sets.items():
        fig,ax=plt.subplots(figsize=(12,5)); match_view(ax,rows,key)
        save(fig,dest/f'matches_{key}.png','At most 100 lines shown, deterministically subsampled; CSV includes every match.')
    fig,axes=plt.subplots(1,2,figsize=(12,5)); names=['raw_nearest','forward_ratio','mutual_final']
    bars=axes[0].bar(['Raw nearest','Forward ratio','Final mutual'],[len(sets[n]) for n in names],color=['#999999','#ee7733','#009988'])
    axes[0].bar_label(bars); axes[0].set(ylabel='Proposed correspondences',title='Filtering counts')
    for accept,color,label in [('0','#cc3311','Rejected'),('1','#009988','Final accepted')]:
        rows=[r for r in matches if r['accepted']==accept]
        axes[1].scatter([int(r['d2']) for r in rows],[int(r['d1']) for r in rows],s=8,alpha=.45,c=color,label=label)
    axes[1].plot([0,256],[0,.87*256],c='#0077bb',label='d1 = 0.87 d2')
    axes[1].set(xlabel='Second-best Hamming distance (d2)',ylabel='Best Hamming distance (d1)',title='Forward ratio test'); axes[1].legend(fontsize=8)
    save(fig,dest/'06_matching_filter_counts.png','Below the line passes the forward test; reverse ratio and mutual best are also required.')
    summary=dict(frames=frames,feature_counts={str(f):{s:len(rows) for s,rows in zip(STAGES,all_data[f][1])} for f in frames},
                 matching_counts={k:len(v) for k,v in sets.items()},ratio_threshold=.87,line_limit=100)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(f'PNG frontend visuals: {dest}')

def tum(path):
    return np.array([[float(x) for x in line.split()] for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')])

def icp():
    dest=OUT/'icp'; dest.mkdir(parents=True,exist_ok=True); raw=ROOT/'icp_cpp_results'
    if not (raw/'run_complete.txt').is_file(): raise RuntimeError('Run icp_sequence before exporting ICP plots.')
    gt=tum(ROOT/'data/synthetic_bunny/groundtruth.txt')[:,1:4]
    names=['baseline','robust','point','plane']; selected=(raw/'selected_trajectory.txt').read_text().strip()
    if selected=='pose_graph': names.append('pose_graph')
    elif selected!='plane': raise RuntimeError('Unknown current trajectory selection')
    positions={name:tum(raw/f'{name}_trajectory.txt')[:,1:4] for name in names}
    pnp=read(ROOT/'pnp_from_scratch/results/pipeline/trajectory.csv')
    positions['scratch_pnp']=np.array([[float(r[k]) for k in ('tx','ty','tz')] for r in pnp])
    labels={'baseline':'Feature-pair SVD','robust':'RANSAC initialization','point':'Gauss-Newton point-to-point',
            'plane':'Huber Gauss-Newton point-to-plane','pose_graph':'Verified loop + pose graph','scratch_pnp':'Independent scratch PnP'}
    colors=dict(zip(positions,['#999999','#ee7733','#aa4499','#0077bb','#009988','#cc3311']))
    bounds=np.concatenate([gt]+list(positions.values())); fig,axes=plt.subplots(2,3,figsize=(15,10)); metrics=[]
    for ax,(name,xyz) in zip(axes.flat,positions.items()):
        if xyz.shape!=gt.shape: raise RuntimeError('Pose count mismatch')
        error=np.linalg.norm(xyz-gt,axis=1); ate=np.sqrt(np.mean(error**2))
        metrics.append(dict(method=name,ATE_RMSE_m=ate,final_position_error_m=error[-1]))
        ax.plot(gt[:,0],gt[:,2],c='black',ls='--',label='Ground truth')
        ax.plot(xyz[:,0],xyz[:,2],c=colors[name],label='Estimate'); ax.scatter(xyz[0,0],xyz[0,2],c='black',s=20)
        ax.set(title=f'{labels[name]}\nPosition RMSE: {ate*1000:.2f} mm',xlabel='World X (m)',ylabel='World Z (m)',
               xlim=(bounds[:,0].min()-.04,bounds[:,0].max()+.04),ylim=(bounds[:,2].min()-.04,bounds[:,2].max()+.04))
        ax.set_aspect('equal',adjustable='box'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(positions):]: ax.axis('off')
    fig.suptitle('Camera trajectories: cumulative registration stages and independent PnP')
    save(fig,dest/'01_trajectory_stages.png','Same frame-zero world anchor, no fitted scale/alignment. These cumulative stages are not isolated ablations.')
    write_csv(dest/'trajectory_metrics.csv',metrics)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for name,xyz in positions.items():
        error=np.linalg.norm(xyz-gt,axis=1)*1000
        axes[0].plot(error,label=labels[name],color=colors[name])
        if name in ['plane','pose_graph','scratch_pnp']: axes[1].plot(error,label=labels[name],color=colors[name])
    for ax in axes: ax.set(xlabel='Frame',ylabel='Position error (mm)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[0].set_title('All stages'); axes[1].set_title('Final methods: closer view'); fig.suptitle('Trajectory error against ground truth')
    save(fig,dest/'02_position_errors.png','Ground truth is used for evaluation after ICP inference. PnP has no ICP loop correction.')
    rows=read(raw/'iteration_history.csv'); pair=[r for r in rows if int(r['pair'])==15]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,stage in enumerate(['point','plane']):
        h=[r for r in pair if r['stage']==stage]
        if not h: raise RuntimeError('Missing pair 15->16 iteration history')
        x=[int(r['iteration']) for r in h]
        for key,label,style in [('nn_rmse_before_m','Before update','o--'),('nn_rmse_after_m','After update','o-')]:
            axes[0,col].plot(x,[float(r[key])*1000 for r in h],style,label=label,ms=3)
        axes[0,col].set(title=f'{stage}: Euclidean NN error',ylabel='RMSE (mm)',xlabel='Accepted update'); axes[0,col].legend(fontsize=8)
        for key,label,style in [('objective_before_m2','Before update','o--'),('objective_after_m2','After update','o-')]:
            axes[1,col].plot(x,[float(r[key])*1e6 for r in h],style,label=label,ms=3)
        axes[1,col].set(title='Squared distance' if stage=='point' else 'Huber normal-distance objective',
                      ylabel='Mean objective (mm²)',xlabel='Accepted update'); axes[1,col].legend(fontsize=8)
        for row in range(2): axes[row,col].grid(alpha=.2)
    fig.suptitle('Frames 15 to 16: actual ICP iterations')
    save(fig,dest/'03_iteration_updates.png','Each before/after uses fixed pairs. NN pairs refresh next iteration; costs across iterations can change with membership.')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,stage in zip(axes,['point','plane']):
        h=[r for r in pair if r['stage']==stage]; hashes=[r['assignment_hash'] for r in h]
        changes=[0]+[int(a!=b) for a,b in zip(hashes,hashes[1:])]
        x=np.array([int(r['iteration']) for r in h]); y=np.array([int(r['pairs']) for r in h]); changed=np.array(changes,dtype=bool)
        ax.plot(x,y,color='#999999',lw=1)
        ax.scatter(x[changed],y[changed],c='#ee7733',label='Assignment set changed')
        ax.scatter(x[~changed],y[~changed],c='#0077bb',label='Initial / unchanged set')
        ax.set(title=f'{stage}: {sum(changes)} assignment-set changes',xlabel='Accepted update',ylabel='Retained pairs')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Nearest-neighbour correspondences are refreshed during ICP')
    save(fig,dest/'04_nearest_neighbour_refresh.png','Hash changes mean retained (source ID, target ID) sets changed, including filtering. Distance sorting does not affect hashes.')
    audit=raw/'audit_pair_15_16'; clouds=read(audit/'clouds.csv'); stages=read(audit/'stage_poses.csv')
    s=np.array([[float(r[k]) for k in ['x','y','z']] for r in clouds if r['role']=='source'])
    t=np.array([[float(r[k]) for k in ['x','y','z']] for r in clouds if r['role']=='target']); transformed={}
    for r in stages:
        R=np.array([float(r[f'r{a}{b}']) for a in range(3) for b in range(3)]).reshape(3,3)
        transformed[r['stage']]=s@R.T+np.array([float(r[k]) for k in ['tx','ty','tz']])
    fig,axes=plt.subplots(2,4,figsize=(16,8)); show=['unaligned','robust','point','plane']
    data=np.concatenate([t]+[transformed[v] for v in show])
    for col,name in enumerate(show):
        for row,(a,b) in enumerate([(0,2),(0,1)]):
            ax=axes[row,col]; q=transformed[name]
            ax.scatter(t[:,a],t[:,b],s=2,c='#999999',label='Target frame 15',alpha=.45)
            ax.scatter(q[:,a],q[:,b],s=2,c='#0077bb',label='Source frame 16',alpha=.45)
            ax.set(title=name,xlabel=['X','Y','Z'][a]+' (m)',ylabel=['X','Y','Z'][b]+' (m)')
            ax.set_xlim(data[:,a].min()-.005,data[:,a].max()+.005); ax.set_ylim(data[:,b].min()-.005,data[:,b].max()+.005)
            ax.set_aspect('equal',adjustable='box')
            if row==0 and col==0: ax.legend(fontsize=7,markerscale=3)
    fig.suptitle('Frames 15 to 16: depth clouds before and after registration')
    save(fig,dest/'05_cloud_alignment_before_after.png','Exact C++ sampled clouds and transforms. Two projections share fixed limits; small differences are clearer in metrics.')
    pnp_pairs=read(ROOT/'pnp_from_scratch/results/pipeline/pairs.csv'); fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,keys,title,unit,mult in [(axes[0],['linear_rot_err_deg','rot_err_deg'],'PnP rotation error','degrees',1),
             (axes[1],['linear_trans_err_m','trans_err_m'],'PnP translation error','mm',1000)]:
        for key,label in zip(keys,['Before reprojection refinement','After refinement + inlier reselection']):
            ax.plot([int(r['pair']) for r in pnp_pairs],[float(r[key])*mult for r in pnp_pairs],label=label)
        ax.set(title=title,xlabel='Frame pair',ylabel=unit); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Independent scratch PnP: pair errors before / after refinement')
    save(fig,dest/'06_pnp_refinement_before_after.png','PnP uses 3D-to-2D correspondences; it is independent of the 3D-to-3D ICP pipeline.')
    print(f'PNG registration visuals: {dest}')
    for name in ['summary_metrics.csv','pair_metrics.csv','iteration_history.csv','loop_edges.csv']:
        shutil.copyfile(raw/name,dest/name)
    shutil.copytree(audit,dest/'audit_pair_15_16',dirs_exist_ok=True)

def manifest():
    files=[]
    for p in sorted(OUT.rglob('*.png')):
        with Image.open(p) as im: im.verify()
        with Image.open(p) as im: width,height=im.size
        files.append(dict(path=str(p.relative_to(OUT)),width=width,height=height,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (OUT/'image_manifest.json').write_text(json.dumps(files,indent=2)+'\n')

def flowchart():
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(10,12)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    boxes=[(.35,.94,'Read consecutive RGB + depth frames'),
           (.35,.83,'FAST, Harris ranking, 8-level quotas\nOriented BRIEF + mutual ratio matching'),
           (.35,.72,'Depth in both frames -> 3D matches\nRANSAC + rigid SVD initialization'),
           (.35,.61,'Point-to-point ICP\nRefresh NN -> Gauss-Newton -> repeat'),
           (.35,.50,'Point-to-plane ICP\nRefresh NN -> robust normal update -> repeat'),
           (.35,.39,'Accept result, or flag a motion prediction\nCombine relative movements into a trajectory'),
           (.35,.26,'After sequence: verify loop candidates\nOptimize pose graph if a valid loop is found'),
           (.35,.13,'Save poses, metrics and PNG diagnostics\nDisplay camera trajectories in Blender')]
    for x,y,label in boxes:
        ax.add_patch(FancyBboxPatch((x-.25,y-.033),.50,.066,boxstyle='round,pad=.009',fc='#e8f2fa',ec='#0077bb'))
        ax.text(x,y,label,ha='center',va='center',fontsize=10)
    def arrow(a,b,**kw): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#444444',lw=1.3,**kw))
    for (_,a,_),(_,b,_) in zip(boxes,boxes[1:]): arrow((.35,a-.043),(.35,b+.043))
    ax.add_patch(FancyBboxPatch((.70,.60),.27,.26,boxstyle='round,pad=.01',fc='#fff0df',ec='#ee7733'))
    ax.text(.835,.73,'Independent scratch PnP\n\nFirst-frame depth -> 3D\nNext-frame pixels -> 2D\nRANSAC PnP + refinement\n\nSave separate trajectory',ha='center',va='center',fontsize=9)
    arrow((.61,.83),(.69,.83)); ax.text(.83,.9,'Separate comparison',ha='center',fontsize=10)
    ax.plot([.08,.025,.025,.08],[.39,.39,.94,.94],color='#444444',lw=1.3)
    arrow((.08,.94),(.09,.94)); ax.text(.015,.66,'Next frame pair',rotation=90,ha='center',va='center',fontsize=9)
    fig.suptitle('Exact implemented flow: ICP and independent PnP')
    save(fig,OUT/'00_pipeline_flowchart.png','PNG rendering of code flow. Accepted registration does not guarantee a globally correct pose.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--frontend-only',action='store_true')
    p.add_argument('--check-dependencies',action='store_true'); a=p.parse_args()
    if a.check_dependencies:
        print('PNG dependencies available: matplotlib, numpy, Pillow'); raise SystemExit(0)
    for path in sorted((OUT/'frontend').glob('pair_*/matching_decisions.csv')):
        _,i,j=path.parent.name.split('_'); frontend(int(i),int(j))
    if not a.frontend_only: icp()
    flowchart()
    manifest()
