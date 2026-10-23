#!/usr/bin/env bash
# Run from any directory. Estimation is C++; Python draws images and builds Blender.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python3 scripts/render_outputs.py --check-dependencies
cmake -S pnp_from_scratch -B build-scratch -DCMAKE_BUILD_TYPE=Release
cmake --build build-scratch -j2
ctest --test-dir build-scratch --output-on-failure
./build-scratch/scratch_pipeline --repo "$repo_dir"
# Controlled diagnostics: Python draws direct PNGs from C++ exports.
for pair in "0 1" "15 16"; do
  read -r frame_i frame_j <<< "$pair"
  ./build-scratch/feature_audit --repo "$repo_dir" --i "$frame_i" --j "$frame_j"
  python3 scripts/render_frontend_audit.py --i "$frame_i" --j "$frame_j"
  python3 scripts/validate_frontend_audit.py --i "$frame_i" --j "$frame_j"
done
cmake -S icp_cpp -B build-icp -DCMAKE_BUILD_TYPE=Release -DBUILD_ORB_BENCHMARK=ON
cmake --build build-icp -j2
ctest --test-dir build-icp --output-on-failure
mkdir -p icp_cpp_results
set +e
./build-icp/icp_sequence data/synthetic_bunny icp_cpp_results
icp_status=$?
set -e
if [[ "$icp_status" != 0 && "$icp_status" != 2 ]]; then exit "$icp_status"; fi
if [[ "$icp_status" == 2 ]]; then
  echo 'Some pairs failed; fallback transitions will be flagged in the output.' >&2
fi
./build-icp/orb_benchmark data/synthetic_bunny outputs/orb_benchmark --repeats 3
python3 scripts/render_orb_benchmark.py
python3 scripts/render_orb_keypoints.py
python3 scripts/render_outputs.py
python3 scripts/validate_outputs.py
blender --background --python-exit-code 1 --python visualization/scripts/build_scene.py
blender --background -y outputs/blender/bunny_slam_demo.blend --python-exit-code 1 \
  --python scripts/check_blender_consistency.py -- icp_cpp_results/blender_consistency.csv
printf '%s\n' 'Ready: blender -y outputs/blender/bunny_slam_demo.blend'
printf '%s\n' 'Images: xdg-open outputs'
