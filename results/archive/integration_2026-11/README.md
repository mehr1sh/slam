# Integration evidence (branch `integration/teammate`, 2026-11-04)

The final tables behind the T0 production decision and their raw inputs. The decision, the configurations
and the interpretation are in `docs/components/pipeline.md`, "Production defaults (T0) and why". Ground truth appears only in
the `eval_*` columns.

| path | content | SHA-256 (prefix) |
|---|---|---|
| `analysis_final2.md` | final tables: fixtures, Bunny, TUM stride 4, TUM full sequence, leave-one-out attribution | `6fed7be0…` |
| `analysis_final2/attribution.csv` | leave-one-out from T0 | `5a36b8b8…` |
| `analysis_final2/fixture_pairs.csv` | per fixture pair and configuration | `49b92cf9…` |
| `analysis_final2/sequences.csv` | per sequence and configuration | `8d596b0d…` |
| `pipeline_stride4_timing.txt` | wall-clock `scratch_pipeline` time on TUM n100_stride4 (99 pairs) | `f041207d…` |
| `raw/<run>/` | `runs.csv`, `matchsets.csv`, `ate.csv` and `command.txt` (or `README.txt`) of the 9 `scratch_experiments` runs | byte-identical copies of the archived runs; `integration_analysis.py` regenerates all four tables from them byte-identically (below) |

Copied on 2026-11-04 from the ignored `output/integration/` of the integration worktree; that full working output
(197 MB) is kept only in the external archive `../archive/2026-11-04_integration/output/integration/`. Only the inputs of `analysis_final2` are tracked: the per-pair
diagnostics, logs and earlier partial analyses stay in the archive.

## Regenerate the tables (byte-identical, verified 2026-11-04)

```bash
R=results/archive/integration_2026-11/raw
python3 tools/pipeline_analysis/integration_analysis.py /tmp/analysis_final2 \
  fixtures=$R/fixtures_20seeds,$R/fixtures_20seeds_T0C bunny=$R/bunny_10seeds_final,$R/bunny_10seeds_T0C \
  stride4=$R/stride4_5seeds,$R/stride4_5seeds_T0C full=$R/full_3seeds,$R/full_3seeds_T0C,$R/full_3seeds_T0-ransac \
  > /tmp/analysis_final2.md
```

The raw runs themselves were produced by the commands in each `raw/<run>/command.txt` (paths as they were
before the reorganisation: `pnp_from_scratch/build/` is now `build/`). `bunny_10seeds_final/README.txt`
explains how that run was assembled.

## How `pipeline_stride4_timing.txt` was measured

One sequential run per configuration on the same machine, no other load, seed 12345; time per pair =
total / 99. The commands, verbatim (pre-reorganisation paths):

```bash
B=pnp_from_scratch/build/scratch_pipeline
COMMON=(--tum-frames output/scratch/rgbd_dataset_freiburg1_xyz/n100_stride4/frames.csv --tum-sequence data/rgbd_dataset_freiburg1_xyz --intrinsics 517.3,516.5,318.6,255.3 --seed 12345)
run() { name=$1; shift; out=output/integration/pipeline_stride4_$name
  [ -e "$out" ] && { echo "exists $out"; exit 1; }
  t0=$(date +%s.%N); $B "${COMMON[@]}" --out $out "$@" > $out.log 2>&1; t1=$(date +%s.%N)
  python3 -c "print('$name', round($t1-$t0,2), 's total', round(($t1-$t0)/99,3), 's/pair')" | tee -a output/integration/pipeline_stride4_timing.txt; }
run M0 --stage2-frontend --pose-gate
run T0
run T0G --pose-gate
run T0C --pnp-iterations 300 --pnp-threshold 8 --feature-budget 0
run T0-ransac --pnp-iterations 300 --pnp-threshold 8   # appended later by the same procedure
```

Timings are machine-dependent; compare ratios, not absolute seconds.
