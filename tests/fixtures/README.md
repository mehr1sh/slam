# Test fixtures (frozen)

Frozen inputs and expected values for the tests and regression locks. **Never regenerate or edit them**: they
are copies of archived runs, and every data directory is checked against its SHA-256 manifest by the CTest
`fixture_checksums` (`CMakeLists.txt`). The data files are read-only on disk. Full per-file descriptions:
`docs/FILE_GUIDE.md`.

## Locks and pair lists

| file | what it is | read by |
|---|---|---|
| `baseline_pairs.csv` | the three locked frame pairs (Bunny 0→1, TUM fr1/xyz stride-4 46→47 and 73→74): image paths, intrinsics, ground truth, the export each is cross-checked against | `scratch_baseline` (CTest `scratch_baseline_lock`), `test_baseline` |
| `baseline_expected.csv` | Stage-1 numeric lock of those pairs (the earlier M0 / Stage-2 front end, 17 significant digits) | `scratch_baseline_lock` |
| `baseline_expected_gate.csv` | Stage-2 lock of the pose-validity gate on the same pairs | `scratch_baseline_lock` |
| `baseline_trajectories.csv`, `baseline_trajectories_gate.csv` | raw-ATE locks of 7 + 6 frozen runs (tolerance 1e-6 m) | `tools/pipeline_analysis/run_baseline.sh` step 4 |
| `stage3_pairs.csv` | pair list of the matcher and integration experiments | `scratch_experiments --fixtures` |

The TUM rows name their images as `data/rgbd_dataset_freiburg1_xyz/…`; they are read from `$SLAM_TUM_ROOT` when it
is set (`docs/DATA_AND_ARCHIVE.md`). Without the TUM images those two pairs are skipped (exit 77).

## Frozen data directories

| directory (manifest) | files | what it is | source | read by |
|---|---|---|---|---|
| `stage1_bunny_export/` (`baseline_bunny_sha256.txt`; `baseline_bunny_gate_sha256.txt` is the manifest of the same run with the pose gate on, 76 entries including `pose_validity.csv`, compared by `run_baseline.sh` only) | 75 | the complete `scratch_pipeline` Bunny export of the Stage-1 configuration (36 frames) | external archive `2026-11-04_integration/pnp_from_scratch_results/pipeline/` | `run_baseline.sh` (Bunny byte identity, `bunny_36` ATE lock), `baseline_pairs.csv` export cross-check |
| `stage1_tum_exports/rgbd_dataset_freiburg1_xyz/n100_stride4/` (`stage1_tum_export_sha256.txt`) | 11 | frame list, `pairs.csv` and the fixture pairs' correspondences and keypoints of the frozen TUM stride-4 run | external archive `…/output/scratch/rgbd_dataset_freiburg1_xyz/n100_stride4/` (identical in both snapshots) | `scratch_baseline_lock` export cross-check |
| `stage1_ate_runs/` (`stage1_ate_runs_sha256.txt`) | 24 | `trajectory.csv` + `frames.csv` of the 12 frozen TUM runs of the ATE locks (`scratch/` and `scratch_gate/`) | external archive `2026-11-04_main/output/{scratch,scratch_gate}/…` | `run_baseline.sh` step 4 |
| `bunny_t0_scene/` (`bunny_t0_scene_sha256.txt`) | 2 | `trajectory.csv` + `pairs.csv` of the T0 Bunny run the Blender scene shows (reproduces the scene's fingerprint `42aef737…`) | the milestone-3 CTest run of 2026-11-04 (no re-run) | the Blender scene and its checks, `test_blender_data.py`, `render_outputs.py` |
| `reference_baseline/` (`reference_baseline_sha256.txt`) | 41 | the OpenCV reference pipeline's frozen Bunny export at the `baseline-pre-migration` tag: `correspondences/` (35 pairs), `pair_metrics.csv`, `metrics.json`, `slam_trajectory.csv`, `reference_features/` (ORB keypoints and raw matches of pair 0→1), and `reference_orb_distances_0_1.csv` (all-pairs ORB distance histogram) | `slam_trajectory_test <copy of data/synthetic_bunny> --export-dir DIR --export-pair 0` / `all` built at that tag; the histogram by `tools/pipeline_analysis/export_reference_orb_distances.py` (it asserts that its best-match distances equal `reference_features/pair_0_1_raw_matches.csv`) | `pnp_from_scratch` and `pnp_full_sequence` (milestones 1–2, default input), `test_png`, `make_figures.py`, `descriptor_figure.py`, `trajectory_figures.py` |

Check by hand: `ctest --test-dir build -R fixture_checksums`, or in a data directory
`sha256sum --quiet -c ../<manifest>`.
