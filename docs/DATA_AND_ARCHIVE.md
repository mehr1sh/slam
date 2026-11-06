# Data and archive

## Datasets

| path | tracked | content |
|---|---|---|
| `data/synthetic_bunny/` | yes | 36 rendered RGB-D frames of the Stanford Bunny (`000000.png`, `000000_depth.png`, …), `groundtruth.txt` (TUM format), `intrinsics.txt`. Also holds the historical reference-pipeline trajectories `pnp_trajectory.txt`, `icp_trajectory.txt` and `slam_trajectory.csv` (produced by the book ICP; see the warning below). Its `intrinsics.txt` marks the repository root for every C++ program. |
| `data/tum_sample/` | yes | two RGB-D frames used by the reference demos |
| `data/meshes/` | yes | the Bunny mesh used to render the synthetic dataset |
| TUM RGB-D sequences | no | external dataset, located through `SLAM_TUM_ROOT` (below) |

### TUM RGB-D sequences (external, configurable)

The repository does not contain the TUM sequences and does not depend on any other checkout for them. Tools
that have a default sequence look for `rgbd_dataset_freiburg1_xyz` in this order:

1. `$SLAM_TUM_ROOT/rgbd_dataset_freiburg1_xyz`, if the environment variable `SLAM_TUM_ROOT` is set (the
   directory that **contains** the sequence directories);
2. `data/rgbd_dataset_freiburg1_xyz` inside the repository (ignored by Git; a directory or a symbolic link
   you create yourself).

Setup (once):

```bash
mkdir -p /path/to/tum && cd /path/to/tum
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar xzf rgbd_dataset_freiburg1_xyz.tgz
export SLAM_TUM_ROOT=/path/to/tum            # e.g. in your shell profile
```

Tools that take the sequence as an argument (`--tum-sequence`, `--dataset`, the evaluator's first argument)
accept any path, e.g. `--tum-sequence "$SLAM_TUM_ROOT/rgbd_dataset_freiburg1_xyz"`.

**When the sequence is unavailable:**
- `run_baseline.sh --rerun-*` stops with exit status 2 and names both locations before building anything;
- `scratch_baseline_lock` (CTest) skips its two TUM image cross-checks and still checks everything else;
- `tools/evaluation/test_pose_conventions.py` and `tools/evaluation/diagnostics/test_diagnostics.py` report the
  real-data part as `[SKIPPED]`;
- the dashboard lists the runs but shows "dataset … is not available" instead of images.
No test or lock reads the TUM images otherwise: the tracked fixtures hold the frozen exports.

Frame lists for the production pipeline (`--tum-frames`) are the `frames.csv` files of evaluator runs. The
stride-4 list is tracked with the frozen TUM export: `tests/fixtures/stage1_tum_exports/rgbd_dataset_freiburg1_xyz/n100_stride4/frames.csv`; others are in the archived runs
(e.g. `archive/2026-11-04_main/output/scratch/rgbd_dataset_freiburg1_xyz/full/frames.csv`).
Intrinsics for freiburg1: `--intrinsics 517.3,516.5,318.6,255.3`.

**Do not run the reference `slam_trajectory_test` on `data/synthetic_bunny` directly**: it writes its
trajectories into the dataset directory given to it. Run it on a copy (as `tools/evaluation/run_reference_suite.sh`
does), or use `slam_trajectory_test_main_icp --out-dir DIR` (main's version).

## Tracked evidence

| path | content |
|---|---|
| `tests/fixtures/` | frozen data checked by the CTest `fixture_checksums`; one row per set in `tests/fixtures/README.md` |
| `results/archive/integration_2026-11/` | the final integration analysis behind the production defaults, its timing file and raw CSVs (see its README; the interpretation is in `docs/components/pipeline.md`, "Production defaults (T0) and why") |
| `results/reports/figures/`, `results/blender/`, `results/benchmarks/rgbd_icp/` | the teammate's figures, the Blender scene (Blender 5.2; needs Blender 5.x) and the ICP benchmark output |
| `results/reports/diagnostics/` | reference-pipeline diagnostic reports |

## Archive (outside Git)

Location: `/home/mehirizz/IIITH/RRC/slam/archive/` (outside the repository). Every file is listed with its
SHA-256 in `archive/MANIFEST.sha256` (9348 files, 3.4 GB).

```
archive/
├── README.md
├── MANIFEST.sha256
├── 2026-11-04_main/                     from the original main worktree
│   ├── output/                          3.1 GB: scratch, scratch_gate, stress, diagnostics, blender, … runs
│   ├── pnp_from_scratch_results/        pipeline*, depth_edge, figures* (depth-edge study evidence)
│   ├── results_generated/{data,figures,tables}
│   ├── diagnostics_generated/           cpp/, figures/, *.csv, summary_generated.md
│   ├── visualization_scenes/            generated Blender scenes, including tum/
│   └── preview.png                      Blender render: scene TUM_SLAM, camera Overview_Camera, frame 75
└── 2026-11-04_integration/              from the integration worktree
    ├── output/                          integration experiments (complete working output), partial TUM copies,
    │                                    baseline reports
    └── pnp_from_scratch_results/        includes the frozen Stage 1 Bunny export (pipeline/)
```

Verify the archive:

```bash
cd /home/mehirizz/IIITH/RRC/slam/archive && sha256sum --quiet -c MANIFEST.sha256 && echo OK
```

Development records that are not part of the repository (reorganisation reports, migration checkpoints,
superseded documents, the presentation, the original Blender scene) are kept in
`/home/mehirizz/IIITH/RRC/slam/backups/` with SHA-256 manifests.

### Runtime output and archived runs

All runtime output goes to `results/runs/` (ignored, disposable): `scratch_pipeline` → `results/runs/pipeline/`,
the milestone programs → `results/runs/milestones/`, `depth_edge_experiment` → `results/runs/depth_edge/`,
`icp_sequence` → `results/runs/rgbd_icp/` (the tracked `results/benchmarks/rgbd_icp/` is written only when passed
explicitly), `scratch_baseline`, `tum_trajectory_eval`, the reference demos, the dashboard, `make_diagnostics.py`,
`export_tum_scene.py`, and the reference-results scripts (`generate_results.py`, `generate_correspondence_figure.py`,
`generate_correspondence_map.py`, `export_correspondences.sh`) → `results/runs/reference_results/`. The Blender
scene reads only `results/benchmarks/rgbd_icp/` and the frozen `tests/fixtures/bunny_t0_scene/`.

Archived runs are frozen: read them in place as inputs (`A=/home/mehirizz/IIITH/RRC/slam/archive/2026-11-04_main/output`,
e.g. `--baseline $A/stress/xyz_session1/n100_stride4`). Never link them into `results/runs/`; to browse one in the
dashboard, **copy** it there.

`scratch_baseline_lock` reads the frozen TUM export from `tests/fixtures/stage1_tum_exports/` (tracked); only the
TUM images are needed for its two TUM cross-checks (skipped without them). The ATE locks in
`tests/fixtures/baseline_trajectories*.csv` (`run_baseline.sh` step 4) read the 12 archived TUM runs from
`tests/fixtures/stage1_ate_runs/{scratch,scratch_gate}/rgbd_dataset_freiburg1_xyz/<run>/`: only the two files the
lock reads (`trajectory.csv`, `frames.csv`), byte-identical to the archive; the locks do not need the archive.

### Configuring the archive location

Only `run_baseline.sh --rerun-stride4 / --rerun-full` need the archive (and the TUM sequence): they compare a
fresh run with the **complete** frozen run (`pairs.csv`, trajectories, `summary.txt`, `correspondences/`,
`keypoints/`), which is not tracked. Give the archive root (the directory holding `MANIFEST.sha256` and
`2026-11-04_main/`):

```bash
export SLAM_ARCHIVE_DIR=/path/to/archive        # e.g. ../archive beside the repository on the original machine
export SLAM_TUM_ROOT=/path/to/tum
bash tools/pipeline_analysis/run_baseline.sh --build build --rerun-stride4
# or: bash tools/pipeline_analysis/run_baseline.sh --build build --rerun-stride4 --archive /path/to/archive
```

The script never writes to the archive. Before comparing it verifies every archive file of the reference run
against `MANIFEST.sha256`; with no archive given, a missing manifest, a missing run or a missing TUM sequence it
stops with exit status 2 before building. Partial copies (`tests/fixtures/`) are never used as references.

**Known result: the stride-4 re-run reports `CHANGED`.** The comparison is strict byte identity and stays so;
`tools/pipeline_analysis/describe_run_diff.py` (called automatically) explains the difference without changing
the verdict. `trajectory.txt`, `summary.txt` and `keypoints/` are byte-identical. The differences:
- `trajectory.csv`: one comment line, and 31 cells (13 columns) differing by at most 1e-9 (rounding);
- `pairs.csv`: one value, pair 37 `refine_iterations` 5 → 4;
- `correspondences/`: 13 columns exist only in the frozen run (main's `depth_edge` and `epi_*` export, not part of
  this pipeline), and one `pnp_residual_px` changes 1.2863 → 1.2864.

The numerical differences come from `RefinePnP` computing its cost as `½·norm·norm` instead of `squaredNorm`
(numerically equivalent, not byte-equivalent).

### Regenerating archived runs

Every run directory holds the exact command in `command.txt` (or `config.json`). Paths in those commands
predate the current layout: `pnp_from_scratch/build/<exe>` is now `build/<exe>`, `scripts/evaluation/` is now
`tools/evaluation/`, `output/` is now `results/runs/`. Results are deterministic for a given seed and build.
