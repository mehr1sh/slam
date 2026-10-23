#!/usr/bin/env python3
"""Verify generated diagnostics against the actual estimator CSVs."""
import argparse,csv,json
from pathlib import Path
from PIL import Image

def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--i',type=int,default=15);p.add_argument('--j',type=int,default=16);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'outputs'/'frontend'/f'pair_{args.i}_{args.j}'
    for frame in (args.i,args.j):
        stages=[read(out/f'frame_{frame:06d}_{s}.csv') for s in ['00_fast_all','01_harris_all','02_fast_quotas','03_harris_quotas']]
        ids=[{r['stable_id'] for r in stage} for stage in stages]
        assert ids[0]==ids[1], 'Harris ranking alone must not change membership'
        for stage,keys in zip(stages,ids):assert len(keys)==len(stage),'IDs must be unique within frame'
        assert ids[2]<=ids[0] and ids[3]<=ids[0]
        for level in range(8):
            responses=[float(r['harris_response']) for r in stages[1] if int(r['level'])==level]
            assert responses==sorted(responses,reverse=True),'Harris order must decrease within each level'
        descriptors={r['stable_id']:r['descriptor_hex'] for r in stages[0]}
        for stage in stages[1:]:
            for r in stage:assert descriptors[r['stable_id']]==r['descriptor_hex'],'Ranking must not alter descriptors'
        levels=read(out/f'frame_{frame:06d}_levels.csv');assert sum(int(r['quota']) for r in levels)==1400
        assert all(int(r['harris_quota'])<=int(r['quota']) for r in levels)
        actual=read(root/f'pnp_from_scratch/results/pipeline/keypoints/frame_{frame:06d}.csv')
        assert len(actual)==len(stages[3])
        for a,b in zip(actual,stages[3]):
            assert a['descriptor_hex']==b['descriptor_hex']
            assert abs(float(a['x'])-float(b['x']))<.001 and abs(float(a['y'])-float(b['y']))<.001
    decisions=read(out/'matching_decisions.csv');kept=set()
    for r in decisions:
        forward=int(r['d1'])<.87*int(r['d2']);reverse=int(r['reverse_d1'])<.87*int(r['reverse_d2'])
        assert forward==bool(int(r['ratio_forward'])) and reverse==bool(int(r['ratio_reverse']))
        accepted=forward and reverse and bool(int(r['mutual_best']))
        assert accepted==bool(int(r['accepted']))
        if accepted:kept.add((int(r['query']),int(r['train'])))
    actual=read(root/f'pnp_from_scratch/results/pipeline/correspondences/pair_{args.i}_{args.j}.csv')
    assert kept=={(int(r['query']),int(r['train'])) for r in actual},'Exported matches must equal estimator matches'
    summary=json.loads((out/'summary.json').read_text())
    assert summary['matching_counts']['mutual_final']==len(kept)
    assert (out/'05_matching_before_after.png').is_file()
    for image in out.glob('*.png'):
        with Image.open(image) as im:im.verify()
    print(f'PASS pair {args.i}->{args.j}: IDs, Harris order, budgets, descriptors, matcher decisions, exact pipeline agreement and PNG integrity')
if __name__=='__main__':main()
