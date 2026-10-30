import csv
import tempfile
import unittest
from pathlib import Path
from build_report import assemble


FIELDS=['dataset','sequence','pair','extractor','image_1','image_2','keypoints_1',
        'keypoints_2','raw_matches','matches','correct_matches','match_precision','repeatability',
        'fundamental_ransac_inliers','fundamental_ransac_inlier_ratio','extract_1_ms',
        'extract_2_ms','match_ms','geometry']


class MasterReportTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def make_run(self,name,dataset,sequence):
        folder=self.root/name;folder.mkdir()
        with (folder/'pairs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
            for pair in ('1_to_2','1_to_3'):
                for method in ('opencv','scratch'):
                    w.writerow(dict(dataset=dataset,sequence=sequence,pair=pair,
                                    extractor=method,image_1='img1.png',image_2='img2.png',
                                    keypoints_1=100,keypoints_2=90,raw_matches=70,matches=70,
                                    correct_matches=60,match_precision=.85,repeatability=.8,
                                    fundamental_ransac_inliers='',fundamental_ransac_inlier_ratio='',
                                    extract_1_ms=23,extract_2_ms=24,match_ms=1,
                                    geometry='homography_ground_truth'))
        return folder

    def test_index_includes_all_pairs_and_missing_image_fallback(self):
        self.make_run('oxford','oxford','bikes')
        self.make_run('hpatches','hpatches','i_test')
        r,p,v=assemble(self.root,self.root/'index.html')
        self.assertEqual((r,p,v),(2,4,0))
        text=(self.root/'index.html').read_text()
        self.assertIn('Oxford Affine',text)
        self.assertIn('HPatches',text)
        self.assertEqual(text.count('class="pair-card"'),4)
        self.assertEqual(len(list(self.root.rglob('report_pairs/*.html'))),4)
        self.assertIn('Images not generated',text)

    def test_links_to_existing_visualized_pair(self):
        folder=self.make_run('oxford','oxford','bikes')
        (folder/'dashboards').mkdir()
        (folder/'dashboards/000_bikes_1_to_2.html').write_text('comparison')
        (folder/'visualizations').mkdir()
        (folder/'visualizations/000_opencv_1_to_2.jpg').write_bytes(b'img')
        r,p,v=assemble(self.root,self.root/'index.html')
        self.assertEqual((r,p,v),(1,2,1))
        self.assertIn('oxford/dashboards/000_bikes_1_to_2.html',(self.root/'index.html').read_text())
        self.assertTrue((folder/'report_pairs/00001_bikes_1_to_3.html').exists())


if __name__=='__main__':unittest.main()
