# Reference ORB features, frame pair 0 → 1 (baseline pipeline)

These are the full keypoint lists (366 and 367) and the raw descriptor matches of the
**baseline** reference pipeline (ORB detector + ORB descriptor + brute-force Hamming)
for frames 0 and 1. The scratch figures (`pnp_from_scratch/tools/make_figures.py`)
use them to compare against the scratch features, without running the reference
pipeline.

**How they were produced.** The `baseline-pre-migration` tag was checked out in a
separate worktree and built, then run:

    slam_trajectory_test <copy of data/synthetic_bunny> --export-dir <dir> --export-pair 0

**Provenance checks** made at that time:
- that run's `pair_0_1_correspondences.csv` is byte-identical to
  `../correspondences/pair_0_1_correspondences.csv`
- its trajectory files are byte-identical to the committed ones

Files:
- `pair_0_1_keypoints.csv`: frame (i = 0, j = 1), index, u, v, size, angle, response, octave
- `pair_0_1_raw_matches.csv`: query_index, train_index, hamming_distance, filtered
