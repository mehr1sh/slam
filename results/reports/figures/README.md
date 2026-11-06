# Open these images directly

No HTML is needed. From the project root, use `xdg-open results/reports/figures`, or double-click
a PNG in your file manager. All new comparison images are generated from actual
C++ exports, rather than invented illustrations.

| Location | What is stored |
| --- | --- |
| `00_pipeline_flowchart.png` | Implemented ICP flow and the independent PnP branch |
| `frontend/pair_15_16/` | Harris, quota and matching comparisons where quotas change selection |
| `frontend/pair_0_1/` | Same comparisons for a pair below all quota limits |
| `icp/01_trajectory_stages.png` | Feature SVD -> RANSAC -> point ICP -> plane ICP -> pose graph, plus separate PnP |
| `icp/02_position_errors.png` | Position errors against ground truth |
| `icp/03_iteration_updates.png` | Actual per-update Euclidean error and optimization objectives |
| `icp/04_nearest_neighbour_refresh.png` | Retained NN counts and changed assignment sets |
| `icp/05_cloud_alignment_before_after.png` | Two projections of the actual sampled clouds before/after alignment |
| `icp/06_pnp_refinement_before_after.png` | PnP pair errors before/after reprojection refinement |
| `icp/*.csv` | Matching registration/trajectory metrics and iteration logs |
| `results/benchmarks/rgbd_icp/audit_pair_15_16/` (repository root; the copies here were removed as byte-identical duplicates) | Sampled cloud points, normals and stage transforms |
| `blender/bunny_slam_demo.blend` | Interactive animated Blender scene, not a rendered video |
| `verification/` | Test and consistency-check logs |
| `image_manifest.json` | Complete PNG inventory, dimensions and file hashes |

To reproduce all results, run `bash visualization/rebuild_blender.sh` from the project
root. To redraw PNGs from already-generated C++ results, run
`python3 tools/figures/render_outputs.py` followed by `python3 tools/figures/validate_outputs.py`.

The trajectory stages are cumulative. Their accuracy differences are not a
single-variable Harris ablation. Point-to-plane optimizes normal distances;
its Euclidean NN distance can rise while that objective falls. The scratch PnP
trajectory is independent and does not receive ICP's pose-graph correction.

Ground truth is only used to anchor/evaluate ICP after estimation. The rendered
cloud and keypoint views use observed depth/images. Historical results elsewhere
in the repository are retained as historical data; these are the current plots.
