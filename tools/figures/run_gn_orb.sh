#!/usr/bin/env bash
# Reproduce GN ICP plus the controlled ORB frontend benchmark.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
if [[ "${1:-}" == --blender ]]; then
  exec bash visualization/rebuild_blender.sh
fi
if [[ $# != 0 ]]; then echo 'Usage: bash tools/figures/run_gn_orb.sh [--blender]' >&2; exit 1; fi
cmake -S experiments/rgbd_icp/icp_cpp -B build-icp -DCMAKE_BUILD_TYPE=Release -DBUILD_ORB_BENCHMARK=ON
cmake --build build-icp -j2
ctest --test-dir build-icp --output-on-failure
set +e
./build-icp/icp_sequence data/synthetic_bunny results/benchmarks/rgbd_icp
icp_status=$?
set -e
if [[ "$icp_status" != 0 && "$icp_status" != 2 ]]; then exit "$icp_status"; fi
if [[ "$icp_status" == 2 ]]; then echo 'ICP has failed pairs; inspect pair_metrics.csv.' >&2; fi
./build-icp/orb_benchmark data/synthetic_bunny results/reports/figures/orb_benchmark --repeats 3
python3 tools/figures/render_orb_benchmark.py
python3 tools/figures/render_orb_keypoints.py
printf '%s\n' 'ICP: results/benchmarks/rgbd_icp/summary_metrics.csv' 'ORB: results/reports/figures/orb_benchmark/summary.csv' 'Plots: results/reports/figures/orb_benchmark/*.png'
