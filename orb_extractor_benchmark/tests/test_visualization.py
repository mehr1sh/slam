import unittest
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualization import draw_points, draw_correspondences, write_pair_dashboard
from benchmark import Features


def feats(points):
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    n = len(pts)
    return Features(pts, np.zeros((n, 32), dtype=np.uint8),
                    np.zeros(n), np.ones(n), np.ones(n), np.zeros(n), 1.5)


class VisualTest(unittest.TestCase):
    def test_point_thinning_does_not_mutate_features(self):
        pts = np.array([[15,15],[16,16],[17,17],[70,70]],np.float32)
        before = pts.copy()
        preview, drawn = draw_points(np.zeros((100,100),np.uint8), pts,
                                     max_points=2, spacing=4)
        np.testing.assert_array_equal(pts, before)
        self.assertEqual(preview.shape, (100,100,3))
        self.assertLessEqual(drawn, 2)

    def test_match_visualization_keeps_all_original_matches(self):
        first = np.zeros((100,100),np.uint8)
        a = feats([[15,15],[45,45],[65,65]])
        b = feats([[16,15],[46,45],[66,65]])
        matches = [cv2.DMatch(i,i,0) for i in range(3)]
        canvas, drawn = draw_correspondences(first, first, a, b, matches,
                                             np.array([True,False,True]),
                                             max_matches=1)
        self.assertEqual(canvas.shape,(100,200,3))
        self.assertEqual(drawn,1)
        self.assertEqual(len(matches),3)
