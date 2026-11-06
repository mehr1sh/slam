#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
python3 tools/figures/render_outputs.py --check-dependencies
frame_i="${1:-15}"
frame_j="${2:-16}"
cmake -S . -B build-scratch -DCMAKE_BUILD_TYPE=Release
cmake --build build-scratch --target feature_audit -j2
./build-scratch/feature_audit --repo "$repo_dir" --i "$frame_i" --j "$frame_j"
python3 tools/figures/render_frontend_audit.py --i "$frame_i" --j "$frame_j"
python3 tools/figures/validate_frontend_audit.py --i "$frame_i" --j "$frame_j"
printf '%s\n' "Images: results/reports/figures/frontend/pair_${frame_i}_${frame_j}/"
