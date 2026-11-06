import unittest
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import Features, geometric_correct, repeatable, synth_cases


def make(points):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    return Features(points, np.zeros((len(points), 32), np.uint8),
                    np.zeros(len(points), np.float32), np.ones(len(points), np.float32),
                    np.ones(len(points), np.float32), np.zeros(len(points), np.int32), 0.)


class GeometryTests(unittest.TestCase):
    def test_repeatability_exact_translation(self):
        a = make([[10, 10], [30, 40], [60, 20]])
        b = make([[12, 11], [32, 41], [62, 21]])
        H = np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]], float)
        self.assertEqual(repeatable(a, b, H, (100, 100), 0.01), (3, 3))

    def test_repeatability_one_to_one(self):
        a = make([[10, 10], [11, 10]])
        b = make([[10, 10]])
        self.assertEqual(repeatable(a, b, np.eye(3), (100, 100), 2), (1, 2))

    def test_match_precision_uses_homography(self):
        a = make([[20, 30], [60, 70]])
        b = make([[21, 30], [80, 70]])
        ms = [cv2.DMatch(0, 0, 0), cv2.DMatch(1, 1, 0)]
        correct = geometric_correct(a, b, ms, np.eye(3), 3)
        self.assertEqual(correct.tolist(), [True, False])

    def test_synthetic_contains_every_transformation(self):
        image = np.zeros((80, 120), np.uint8)
        cases = list(synth_cases(image))
        self.assertGreaterEqual(len(cases), 10)
        self.assertTrue(all(H.shape == (3, 3) for _, _, H, _ in cases))


if __name__ == '__main__':
    unittest.main()
