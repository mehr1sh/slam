# Updated: Gauss–Newton ICP + ORB benchmark

From this directory, run `bash scripts/run_gn_orb.sh`. For the complete Blender rebuild, run `bash scripts/run_gn_orb.sh --blender`.

See [GN_ORB_GUIDE.md](GN_ORB_GUIDE.md) for dependency installation, precise changes, benchmark controls, and the complete maths. Current benchmark measurements are in `outputs/orb_benchmark/RESULTS.md`; dense ICP metadata is in `icp_cpp_results/solver_metadata.txt`.

The production dense point and plane stages both use damped Gauss–Newton. SVD remains for robust feature initialization. Historical measurements and older method descriptions below refer to earlier revisions unless explicitly updated.

---

# Direct images and Blender output

All current visual outputs are in `outputs/`. Open the folder directly:

```bash
xdg-open outputs
```

- `outputs/frontend/pair_15_16/`: PNG before/after comparisons for Harris ranking,
  eight-level quotas and mutual ratio matching, with complete CSV evidence.
- `outputs/frontend/pair_0_1/`: the same comparisons for a pair where quotas do
  not remove any candidates.
- `outputs/icp/`: cloud alignment, iteration updates, trajectory/error PNGs and CSVs.
- `outputs/00_pipeline_flowchart.png`: actual ICP and independent PnP flow.
- `outputs/blender/bunny_slam_demo.blend`: the Blender animation scene.

No HTML gallery or browser is required. PNGs and the Blender scene are already
included; rebuilding reproduces them. See `AUDIT_REPORT.md`, `FRONTEND_VISUALS.md`
and `FINAL_WORKFLOW.md` for the verification, interpretation and commands.

---

# View the final result

Read `FINAL_WORKFLOW.md` for dependencies, commands, measured results and limits.

From the extracted `slam-main` directory:

```bash
bash scripts/rebuild_blender.sh
blender -y outputs/blender/bunny_slam_demo.blend
```

The scene compares GT, corrected ICP, improved scratch PnP and legacy reference
PnP. Press Numpad 0 for the ICP camera; Space plays the animation.
A prebuilt scene and generated results are also included in this ZIP.
