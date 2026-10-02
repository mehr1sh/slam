#!/usr/bin/env bash
# Runs every test/demo program of the repository with its documented
# arguments (COMMANDS.md) and records, per program, the exit status and a
# log. Used to checkpoint the vision-library migration (docs/migration/).
#
# Usage: scripts/migration/run_test_suite.sh OUT_DIR
#   OUT_DIR/tests/<program>.log   console output (repository/temp paths replaced
#                                 by <repo>/<tmp> so logs are machine-independent)
#   OUT_DIR/tests/summary.tsv     program, kind, exit status, seconds
#
# Programs that write files do so into a temporary copy of the dataset or
# into ../output (gitignored) -- committed data is never modified. The
# synthetic dataset is regenerated into a temporary directory and compared
# pixel-for-pixel by the metrics script, not here.
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: run_test_suite.sh OUT_DIR}"
mkdir -p "$out/tests"
out="$(cd "$out" && pwd)"
build="$repo/build"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -r "$repo/data/synthetic_bunny" "$tmp/ds"
mkdir -p "$tmp/render" "$tmp/diag"
tum=../data/tum_sample
syn="$tmp/ds"

summary="$out/tests/summary.tsv"
printf "program\tkind\texit_status\tseconds\n" > "$summary"

run() {  # run NAME KIND ARGS...
  local name="$1" kind="$2"; shift 2
  local log="$out/tests/$name.log" start end status
  start=$(date +%s%N)
  (cd "$build" && "./$name" "$@") > "$log.raw" 2>&1
  status=$?
  end=$(date +%s%N)
  sed -e "s#$tmp#<tmp>#g" -e "s#$repo#<repo>#g" "$log.raw" > "$log"
  rm -f "$log.raw"
  local ms=$(( (end - start) / 1000000 ))
  printf "%s\t%s\t%d\t%d.%d\n" "$name" "$kind" "$status" $((ms / 1000)) $(((ms % 1000) / 100)) >> "$summary"
  printf "%-32s %-12s exit %d\n" "$name" "$kind" "$status"
}

# self-checking (exit status is the verdict)
run fast_test                   self-check
run brief_test                  self-check
run essential_matrix_test       self-check
run vo_pipeline_test            self-check
run trajectory_validation_test  self-check  ../data/synthetic_bunny
run features_fast_test          self-check
run features_orientation_test   self-check
run features_brief_test         self-check
run features_matcher_test       self-check
# demonstrations (print results; exit status only says they ran)
run gauss_newton_curve_fit_test demo
run g2o_curve_fit_test          demo
run orb_from_scratch_test       demo
run feature_matching_test       demo  $tum/1.png $tum/2.png
run two_view_pose_test          demo  $tum/1.png $tum/2.png
run triangulation_test          demo  $tum/1.png $tum/2.png
run pnp_test                    demo  $tum/1.png $tum/2.png $tum/1_depth.png $tum/2_depth.png
run icp_test                    demo  $tum/1.png $tum/2.png $tum/1_depth.png $tum/2_depth.png
run render_bunny_test           demo  ../data/meshes/bunny/bun_zipper.ply "$tmp/render" --num-frames 36 --radius-scale 4 --height-scale 1
run pnp_synthetic_test          demo  ../data/synthetic_bunny 0 1
run icp_synthetic_test          demo  ../data/synthetic_bunny 0 1
run pnp_multiframe_test         demo  ../data/synthetic_bunny 0 10
run synthetic_pose_eval_test    demo  ../data/synthetic_bunny/groundtruth.txt 0 1
run slam_trajectory_test        demo  "$syn"
run trajectory_diagnostics_test demo  "$syn" "$tmp/diag"
# regenerated dataset vs the committed one (byte-wise; differing files listed)
regen="$out/tests/dataset_regeneration.txt"
n=0; diff_files=""
for f in "$repo"/data/synthetic_bunny/0000*.png "$repo"/data/synthetic_bunny/groundtruth.txt "$repo"/data/synthetic_bunny/intrinsics.txt; do
  b=$(basename "$f"); n=$((n + 1))
  cmp -s "$f" "$tmp/render/$b" || diff_files="$diff_files $b"
done
{ echo "files compared: $n"; echo "byte-identical: $([ -z "$diff_files" ] && echo yes || echo no)"; echo "differing:${diff_files:- none}"; } > "$regen"
cat "$regen"
echo "wrote $summary"
