# Before/after images, without HTML

From the extracted `slam-main` directory:

```bash
xdg-open outputs/frontend/pair_15_16
```

The supplied PNGs are ready to view. To regenerate one pair:

```bash
bash scripts/visualize_frontend.sh 15 16
```

Use `0 1` or another consecutive pair to change the frames. The renderer uses
matplotlib, NumPy and Pillow; feature extraction and matching remain C++.
Install display dependencies with:

```bash
sudo apt install -y python3-matplotlib python3-numpy python3-pil
```

## Read the comparisons

| PNG filename suffix | What is compared |
| --- | --- |
| `00_feature_stages.png` | FAST all, Harris all, FAST+quotas control, Harris+quotas actual pipeline |
| `01_harris_ranking_before_after.png` | Same candidate pixels; first 12 current array indices show the changed order |
| `identifier_changes.png` | Exact frame-local IDs and their FAST/Harris/final array indices |
| `02_quotas_before_after.png` | Harris ordering fixed; quota disabled versus enabled |
| `03_ranking_same_quota.png` | Same quotas; FAST-selected versus Harris-selected candidates and differences |
| `04_pyramid_quotas.png` | All eight levels: candidates, quota limit and retained count |
| `05_matching_before_after.png` | Raw nearest proposals versus mutual bidirectional ratio survivors |
| `matches_*.png` | Individual raw, forward-ratio, final-mutual and rejected match views |
| `06_matching_filter_counts.png` | Exact matching counts and nearest/second-nearest distance plot |

All stage views for a frame use the same crop and detector parameters. Colours
identify pyramid levels; order labels are array indices. Harris alone does not
remove corners or move them. Its ranking changes which corners survive quotas.
The same-quota control isolates that effect. Quotas are upper limits and unused
capacity is not redistributed, so the final count may be less than 1400.

For frame 15 there are 1573 candidates and 1313 retained features. At fixed quotas,
1192 are shared between FAST and Harris selection, with 121 exclusive to each.
For frames 0 and 1 no level exceeds its quota, so counts do not change; this is
expected, not a broken visualization.

| Pair | FAST candidates | After Harris + quotas | Raw nearest | Forward ratio | Final mutual |
| --- | --- | --- | ---: | ---: | ---: |
| 0 -> 1 | 838 / 815 | 838 / 815 | 838 | 468 | 246 |
| 15 -> 16 | 1573 / 1624 | 1313 / 1324 | 1313 | 730 | 407 |

Match images show at most 100 deterministically sampled lines. Every proposal is
in `matching_decisions.csv`. Passing descriptor tests does not guarantee a correct
physical match; depth filtering and geometric RANSAC follow these images.

## Exact data, alongside the PNGs

`outputs/frontend/pair_15_16/` contains the four stage CSVs per frame, level counts,
`identifier_mapping.csv`, `matching_decisions.csv`, `summary.json` and `config.txt`.
The actual C++ `ExtractFeatures`, `ExtractMultiscale` and `MatchMutualRatio` functions
produce these exports. The validator checks their agreement with the estimator's
keypoint and correspondence files, not an independently simulated detector.

A stable ID such as `f000015_L0_x315_y170` means frame 15, pyramid level 0 and that
integer level pixel. It stays fixed through sorting and selection. It is not a
persistent physical track across frames. Array order is level-first; indices are
not a global Harris ranking across all scales.

## All other images

`outputs/icp/` contains registration and trajectory plots. Open
`outputs/00_pipeline_flowchart.png` for the code flow. The Blender scene is
`outputs/blender/bunny_slam_demo.blend`. See `outputs/README.md` for the complete map.
