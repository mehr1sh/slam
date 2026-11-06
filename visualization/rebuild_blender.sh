#!/usr/bin/env bash
# Run from any directory. Estimation is C++; Python draws images and builds Blender (5.x: the scene is saved by
# Blender 5.2 and needs Blender >= 5.0 to open).
# The Bunny scene reads only frozen or tracked inputs: the corrected-ICP run in results/benchmarks/rgbd_icp/ (written
# below) and the frozen T0 scratch trajectory tests/fixtures/bunny_t0_scene/ (never the fresh run below).
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python3 tools/figures/render_outputs.py --check-dependencies
cmake -S . -B build-scratch -DCMAKE_BUILD_TYPE=Release
cmake --build build-scratch -j2
ctest --test-dir build-scratch --output-on-failure
./build-scratch/scratch_pipeline --repo "$repo_dir" --out results/runs/pipeline
# Informational: does today's T0 run still equal the frozen trajectory the scene shows? (not fatal)
if cmp -s results/runs/pipeline/trajectory.csv tests/fixtures/bunny_t0_scene/trajectory.csv; then
  echo 'scratch_pipeline T0 run == frozen scene trajectory (tests/fixtures/bunny_t0_scene)'
else
  echo 'NOTE: scratch_pipeline T0 run differs from the frozen scene trajectory; the scene keeps the frozen one' >&2
fi
# Controlled diagnostics: Python draws direct PNGs from C++ exports.
for pair in "0 1" "15 16"; do
  read -r frame_i frame_j <<< "$pair"
  ./build-scratch/feature_audit --repo "$repo_dir" --i "$frame_i" --j "$frame_j"
  python3 tools/figures/render_frontend_audit.py --i "$frame_i" --j "$frame_j"
  python3 tools/figures/validate_frontend_audit.py --i "$frame_i" --j "$frame_j"
done
cmake -S experiments/rgbd_icp/icp_cpp -B build-icp -DCMAKE_BUILD_TYPE=Release -DBUILD_ORB_BENCHMARK=ON
cmake --build build-icp -j2
ctest --test-dir build-icp --output-on-failure
mkdir -p results/benchmarks/rgbd_icp
set +e
./build-icp/icp_sequence data/synthetic_bunny results/benchmarks/rgbd_icp
icp_status=$?
set -e
if [[ "$icp_status" != 0 && "$icp_status" != 2 ]]; then exit "$icp_status"; fi
if [[ "$icp_status" == 2 ]]; then
  echo 'Some pairs failed; fallback transitions will be flagged in the output.' >&2
fi
./build-icp/orb_benchmark data/synthetic_bunny results/reports/figures/orb_benchmark --repeats 3
python3 tools/figures/render_orb_benchmark.py
python3 tools/figures/render_orb_keypoints.py
python3 tools/figures/render_outputs.py
python3 tools/figures/validate_outputs.py
blender --background --factory-startup --python-exit-code 1 --python visualization/scripts/build_scene.py
blender --background --factory-startup -y results/blender/bunny_slam_demo.blend --python-exit-code 1 \
  --python visualization/check_blender_consistency.py -- results/benchmarks/rgbd_icp/blender_consistency.csv
printf '%s\n' 'Ready: blender -y results/blender/bunny_slam_demo.blend'
printf '%s\n' 'Images: xdg-open results/reports/figures'
