"""Tests only the display-data bridge; estimation remains C++."""
import csv
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('trajectory_data',ROOT/'visualization/scripts/trajectory_data.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for folder,names in [('data/synthetic_bunny',['groundtruth.txt','pnp_trajectory.txt','intrinsics.txt']),('icp_cpp_results',['plane_trajectory.txt','pose_graph_trajectory.txt','selected_trajectory.txt','run_complete.txt','pair_metrics.csv'])]:
            dest=self.root/folder;dest.mkdir(parents=True)
            for name in names:shutil.copy(ROOT/folder/name,dest/name)
        mesh=self.root/'data/meshes/bunny/bun_zipper.ply'
        mesh.parent.mkdir(parents=True);shutil.copy(ROOT/'data/meshes/bunny/bun_zipper.ply',mesh)
        scratch=self.root/'pnp_from_scratch/results/pipeline/trajectory.csv'
        scratch.parent.mkdir(parents=True);shutil.copy(ROOT/'pnp_from_scratch/results/pipeline/trajectory.csv',scratch)
    def tearDown(self):self.temp.cleanup()
    def test_corrected_metrics(self):
        rows,meta=bridge.load_corrected(self.root,True)
        self.assertEqual(len(rows),36);self.assertEqual(meta['method'],'pose_graph')
        self.assertTrue(0<=meta['ATE_RMSE_m']<.02)
    def test_no_completion_marker(self):
        (self.root/'icp_cpp_results/run_complete.txt').unlink()
        with self.assertRaises(RuntimeError):bridge.load_corrected(self.root)
    def test_plane_selection(self):
        (self.root/'icp_cpp_results/selected_trajectory.txt').write_text('plane\n')
        _,meta=bridge.load_corrected(self.root)
        self.assertTrue(0<=meta['ATE_RMSE_m']<.02)
    def test_changed_source_fingerprint(self):
        _,before=bridge.load_corrected(self.root)
        p=self.root/'icp_cpp_results/pose_graph_trajectory.txt'
        p.write_text(p.read_text()+'# changed source\n')
        _,after=bridge.load_corrected(self.root)
        self.assertNotEqual(before['source_sha256'],after['source_sha256'])
    def test_missing_pose(self):
        p=self.root/'icp_cpp_results/pose_graph_trajectory.txt'
        p.write_text('\n'.join(p.read_text().splitlines()[:-1]))
        with self.assertRaises(ValueError):bridge.load_corrected(self.root)
    def test_rejected_pair_is_visible(self):
        p=self.root/'icp_cpp_results/pair_metrics.csv'
        with p.open() as f:r=csv.DictReader(f);keys=r.fieldnames;rows=list(r)
        rows[0]['status']='ICP_REJECTED'
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        data,meta=bridge.load_corrected(self.root)
        self.assertEqual(data[1]['icp_valid'],0);self.assertEqual(meta['failed_pairs'],1)
if __name__=='__main__':unittest.main()
