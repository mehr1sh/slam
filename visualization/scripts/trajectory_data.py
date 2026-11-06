"""Strict, dependency-free bridge from C++ T_wc files to Blender's table.
No estimation or trajectory composition happens here; only display metrics.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

# Inputs of the Bunny scene, relative to the repository root. They are the only paths the scene, its
# autoload script, the consistency check and the bridge test use (build_scene.py imports them).
#   ICP_RESULTS     the teammate's corrected-ICP run (tracked evidence)
#   SCRATCH_SOURCE  the frozen T0 scratch_pipeline trajectory the scene shows (tests/fixtures; never a mutable run)
ICP_RESULTS = 'results/benchmarks/rgbd_icp'
SCRATCH_SOURCE = 'tests/fixtures/bunny_t0_scene/trajectory.csv'
# A directory is the repository root if it holds both of these (the first is also the C++ FindRepo marker).
ROOT_MARKERS = ('data/synthetic_bunny/intrinsics.txt', 'visualization/scripts/trajectory_data.py')


def find_repo_root(start=None):
    """Repository root: $SLAM_REPO_ROOT if set, else the nearest directory at or above `start` that holds
    ROOT_MARKERS. Raises RuntimeError (no guessing) if neither works."""
    env = os.environ.get('SLAM_REPO_ROOT')
    if env:
        root = Path(env).resolve()
        if not all((root / m).is_file() for m in ROOT_MARKERS):
            raise RuntimeError(f'SLAM_REPO_ROOT={env} is not the repository root (missing {ROOT_MARKERS})')
        return root
    if start is not None:
        for d in [Path(start).resolve(), *Path(start).resolve().parents]:
            if all((d / m).is_file() for m in ROOT_MARKERS):
                return d
    raise RuntimeError(f'Repository root not found above {start}; set SLAM_REPO_ROOT to the repository root')


def read_tum(path):
    poses=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        values=[float(v) for v in line.split()]
        if len(values)!=8 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Invalid pose in {path}')
        q=values[4:]
        norm=math.sqrt(sum(v*v for v in q))
        if abs(norm-1)>1e-4:
            raise ValueError(f'Non-unit quaternion in {path}')
        poses.append((values[0],values[1:4],[v/norm for v in q]))
    if not poses or any(b[0]<=a[0] for a,b in zip(poses,poses[1:])):
        raise ValueError(f'Empty or non-monotonic trajectory: {path}')
    return poses


def errors(a,b):
    translation=math.sqrt(sum((x-y)**2 for x,y in zip(a[1],b[1])))
    dot=min(1.,abs(sum(x*y for x,y in zip(a[2],b[2]))))
    rotation=math.degrees(2*math.acos(dot))
    return translation,rotation


def _hash(paths):
    h=hashlib.sha256()
    for p in paths:
        h.update(str(p.name).encode());h.update(p.read_bytes())
    return h.hexdigest()


def load_corrected(root,write=False):
    root=Path(root).resolve()
    data=root/'data/synthetic_bunny'
    result=root/ICP_RESULTS
    if not (result/'run_complete.txt').is_file():
        raise RuntimeError('No completed corrected C++ run. Run visualization/rebuild_blender.sh first.')
    selected=(result/'selected_trajectory.txt').read_text().strip()
    if selected not in ('plane','pose_graph'):
        raise ValueError('Invalid corrected ICP method selection')
    path=result/f'{selected}_trajectory.txt'
    gt=read_tum(data/'groundtruth.txt')
    pnp=read_tum(data/'pnp_trajectory.txt')
    icp=read_tum(path)
    if not len(gt)==len(pnp)==len(icp):
        raise ValueError('GT, legacy PnP and corrected ICP frame counts differ')
    for label,poses in [('PnP',pnp),('corrected ICP',icp)]:
        if any(abs(a[0]-b[0])>1e-8 for a,b in zip(gt,poses)):
            raise ValueError(f'{label} frame indices differ from GT')
        t,r=errors(poses[0],gt[0])
        if t>1e-5 or r>.01:
            raise ValueError(f'{label} is not in the dataset world frame')
    with (result/'pair_metrics.csv').open() as f:
        pairs=list(csv.DictReader(f))
    if len(pairs)!=len(gt)-1 or any(int(p['pair'])!=i for i,p in enumerate(pairs)):
        raise ValueError('Corrected pair diagnostics do not cover the trajectory')
    rows=[]
    for i in range(len(gt)):
        row={'frame':i}
        for prefix,poses in [('gt',gt),('pnp',pnp),('icp',icp)]:
            for key,value in zip(('x','y','z','qx','qy','qz','qw'),poses[i][1]+poses[i][2]):
                row[f'{prefix}_{key}']=value
            if prefix!='gt':
                t,r=errors(poses[i],gt[i])
                row[f'{prefix}_translation_error']=t
                row[f'{prefix}_rotation_error']=r
        row['icp_valid']=1 if i==0 or pairs[i-1]['status']=='OK' else 0
        rows.append(row)
    scratch=root/SCRATCH_SOURCE
    if not scratch.is_file():
        raise RuntimeError(f'Frozen scratch trajectory {SCRATCH_SOURCE} is missing (tests/fixtures/bunny_t0_scene_sha256.txt)')
    metadata={'method':selected,'trajectory':str(path.relative_to(root)),
              'source_sha256':_hash([data/'intrinsics.txt',root/'data/meshes/bunny/bun_zipper.ply',data/'groundtruth.txt',data/'pnp_trajectory.txt',path,result/'pair_metrics.csv',result/'selected_trajectory.txt',scratch]),
              'frames':len(rows),'failed_pairs':sum(p['status']!='OK' for p in pairs),
              'ATE_RMSE_m':math.sqrt(sum(r['icp_translation_error']**2 for r in rows)/len(rows)),
              'pnp_note':'Legacy PnP comparison; not changed by corrected ICP'}
    if write:
        dest=result/'blender_trajectory.csv'
        with dest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (result/'blender_source.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return rows,metadata


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    _,metadata=load_corrected(root,write=True)
    print(json.dumps(metadata,indent=2))
