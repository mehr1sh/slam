#!/usr/bin/env bash
# Exports the raw 3D->2D correspondences of ALL consecutive pairs with the
# existing pipeline's export-only flags (slam_trajectory_test --export-dir
# --export-pair all). It runs on a temporary copy of the dataset and checks
# that the trajectory outputs are byte-identical to the committed ones.
#
# Output: results/data/raw_correspondences/pair_<i>_<i+1>_correspondences.csv
#   (one row per Hamming-filtered ORB match: u_i, v_i, depth_raw_i, X_i, Y_i,
#    Z_i in camera-i coordinates, u_j, v_j in frame i+1, ...). The full-sequence
#    scratch PnP uses every row with valid frame-i depth; it never reads the
#    reference pipeline's RANSAC columns. The milestone programs default to the
#    frozen baseline export in docs/migration/baseline/correspondences/; this
#    script exports the CURRENT reference pipeline instead.
#
# Run from anywhere after `pixi run build` (repository root build/).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
exe="$repo/build/slam_trajectory_test"
out="$repo/results/data/raw_correspondences"
[[ -x "$exe" ]] || { echo "missing $exe -- run 'pixi run build' in $repo first" >&2; exit 2; }
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -r "$repo/data/synthetic_bunny" "$tmp/ds"
mkdir -p "$out"
(cd "$repo/build" && "$exe" "$tmp/ds" --export-dir "$out" --export-pair all > "$out/slam_trajectory_test.log" 2>&1)
for f in pnp_trajectory.txt icp_trajectory.txt slam_trajectory.csv; do
  cmp -s "$tmp/ds/$f" "$repo/data/synthetic_bunny/$f" || { echo "export run changed $f -- aborting" >&2; exit 1; }
done
n=$(ls "$out"/pair_*_correspondences.csv | wc -l)
echo "exported $n pair correspondence files to ${out#$repo/} (trajectory outputs unchanged)"
