#!/usr/bin/env python3
"""Check direct images, plotted numbers and actual ICP update records."""
import hashlib
import json
from collections import Counter
from render_outputs import ROOT, OUT, read, tum, np, Image

def main():
    manifest=json.loads((OUT/'image_manifest.json').read_text())
    assert manifest,'No PNGs generated'
    assert {str(p.relative_to(OUT)) for p in OUT.rglob('*.png')}=={r['path'] for r in manifest}
    for r in manifest:
        p=OUT/r['path']; assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        with Image.open(p) as image: image.verify()
        assert r['width']>=600 and r['height']>=400
    raw=ROOT/'icp_cpp_results'; history=read(raw/'iteration_history.csv'); counts=Counter()
    for r in history:
        counts[int(r['pair']),r['stage']]+=1
        assert int(r['pairs'])>=70
        numbers=[float(r[k]) for k in r if k not in ['pair','stage','iteration','pairs','assignment_hash']]
        assert np.isfinite(numbers).all()
        assert float(r['objective_after_m2'])<=float(r['objective_before_m2'])+1.1e-14
    for r in read(raw/'pair_metrics.csv'):
        for stage in ['point','plane']:
            assert counts[int(r['pair']),stage]==int(r[f'{stage}_iterations'])
    for stage in ['point','plane']:
        h=[r for r in history if int(r['pair'])==15 and r['stage']==stage]
        assert len({r['assignment_hash'] for r in h})>1,'No changed NN assignment sets recorded for example pair'
    original={r['method']:r for r in read(raw/'summary_metrics.csv')}
    for r in read(OUT/'icp/trajectory_metrics.csv'):
        if r['method'] in original:
            for key in ['ATE_RMSE_m','final_position_error_m']:
                assert abs(float(r[key])-float(original[r['method']][key]))<2e-8,'Plot metric disagrees with C++'
    for r in read(raw/'audit_pair_15_16/stage_poses.csv'):
        R=np.array([float(r[f'r{a}{b}']) for a in range(3) for b in range(3)]).reshape(3,3)
        assert np.allclose(R.T@R,np.eye(3),atol=1e-9) and abs(np.linalg.det(R)-1)<1e-9
    for name in ['summary_metrics.csv','pair_metrics.csv','iteration_history.csv','loop_edges.csv']:
        assert (raw/name).read_bytes()==(OUT/'icp'/name).read_bytes(),'Stale copied data'
    print(f'PASS: {len(manifest)} PNG files; hashes, dimensions, fixed-pair descent, iteration counts, '
          'NN refresh evidence, plotted metrics, rigid poses and copied CSVs')

if __name__=='__main__': main()
