#!/usr/bin/env bash
# Stage-1 regression harness for the scratch pipeline (include/slam/evaluation/baseline.hpp),
# extended in Stage 2 with the pose-validity gate (include/slam/pipeline/validity.hpp). The Stage-1 locks are checked
# with the Stage-1 configuration (--stage2-frontend; branch integration/teammate); the Stage-2
# locks with --stage2-frontend --pose-gate (the gate is off by default on this branch).
#
# Usage: tools/pipeline_analysis/run_baseline.sh --build DIR [--rerun-stride4] [--rerun-full] [--archive DIR]
#
#   --build DIR  build directory, REQUIRED (or the environment variable SLAM_BUILD_DIR); there is no default, so a
#                stale or unrelated tree is never used by accident. (On 2026-11-05 a pre-reorganisation build/
#                ran its old tests and rewrote 13 of their by-products; a later "ctest -N" check rewrote that
#                tree's Testing/Temporary/LastTest.log. Hence no default, and the checks below only read
#                files as text and never run anything in DIR.) Before anything is configured or written, an existing DIR must
#                (a) have been configured from this repository (CMakeCache.txt CMAKE_HOME_DIRECTORY) and (b) already
#                register this repository's CTest scratch_baseline_lock (read from CTestTestfile.cmake as text;
#                nothing is run in DIR); otherwise the script stops with exit 2.
#                A new or empty DIR is configured from scratch. ctest always runs in DIR.
#   1. builds the production build; Bunny byte identity: scratch_pipeline into a temporary directory,
#      --stage2-frontend against a subset of tests/fixtures/baseline_bunny_sha256.txt (Stage 1) and
#      --stage2-frontend --pose-gate against a subset of tests/fixtures/baseline_bunny_gate_sha256.txt (Stage 2)
#   2. the ctest suite (milestone3 writes into the build tree)
#   3. scratch_baseline on the three fixture pairs (tests/fixtures/baseline_pairs.csv) against the
#      Stage-1 lock (baseline_expected.csv), the Stage-2 gate lock (baseline_expected_gate.csv) and
#      the existing exports; report -> results/runs/scratch_baseline/stage1/{baseline_pairs.json,baseline_pairs.csv}
#   4. raw ATE of the frozen runs in tests/fixtures/baseline_trajectories.csv (Stage 1) and
#      baseline_trajectories_gate.csv (Stage 2); their trajectory.csv/frames.csv are tracked in
#      tests/fixtures/stage1_ate_runs/ (copied from the archive); read-only, tolerance 1e-6 m;
#      a run whose directory is absent is reported as skipped
#   --rerun-stride4 / --rerun-full: also re-run scratch_pipeline --stage2-frontend on the TUM n100_stride4
#      (~1.5 min) or full (~12 min) frame list into a temporary directory and require byte-identical
#      pairs.csv, trajectory.csv, trajectory.txt, summary.txt, correspondences/ and keypoints/ against the
#      complete frozen run in the external archive. They need:
#        --archive DIR (or the environment variable SLAM_ARCHIVE_DIR): the archive root, the directory that
#          holds MANIFEST.sha256 and 2026-11-04_main/ (docs/DATA_AND_ARCHIVE.md);
#        the TUM sequence rgbd_dataset_freiburg1_xyz: $SLAM_TUM_ROOT/rgbd_dataset_freiburg1_xyz if SLAM_TUM_ROOT
#          is set, otherwise data/rgbd_dataset_freiburg1_xyz (docs/DATA_AND_ARCHIVE.md).
#      Without them the script stops with exit status 2 before building. The reference run is read in place
#      and first verified against the archive's MANIFEST.sha256 (every listed file present and unchanged);
#      partial copies elsewhere (tests/fixtures/) are never used as references. When a file differs,
#      tools/pipeline_analysis/describe_run_diff.py explains how (FORMAT: comment lines, columns, rows;
#      NUMERICAL: per column, count of differing cells and max |diff|; TEXT); it never changes the verdict,
#      which stays byte identity (tolerance 0) for every compared file.
# Temporary directories are deleted on exit; no existing output is written; the archive is only read.
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${SLAM_BUILD_DIR:-}"
archive="${SLAM_ARCHIVE_DIR:-}"
reruns=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rerun-stride4) reruns+=(n100_stride4) ;;
    --rerun-full) reruns+=(full) ;;
    --archive) [ $# -ge 2 ] || { echo "--archive needs a directory" >&2; exit 2; }; archive="$2"; shift ;;
    --build) [ $# -ge 2 ] || { echo "--build needs a directory" >&2; exit 2; }; build="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
# Build-directory guard (before anything is configured or written).
if [ -z "$build" ]; then
  echo "ERROR: give the build directory: --build DIR (or SLAM_BUILD_DIR); there is no default" >&2; exit 2
fi
build="$(cd "$(dirname "$build")" 2>/dev/null && pwd)/$(basename "$build")" || { echo "ERROR: parent of $build does not exist" >&2; exit 2; }
if [ -f "$build/CMakeCache.txt" ]; then
  home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$build/CMakeCache.txt")"
  if [ "$home" != "$repo" ]; then
    echo "ERROR: $build was configured from '$home', not from $repo; use another --build directory" >&2; exit 2
  fi
  # read-only check: plain text of the generated test file (running ctest, even "ctest -N", writes
  # Testing/Temporary/LastTest.log into the tree)
  if ! grep -qs '^add_test(\[=\[scratch_baseline_lock\]=\]\|^add_test(scratch_baseline_lock ' "$build/CTestTestfile.cmake"; then
    echo "ERROR: $build does not register this repository's CTest scratch_baseline_lock (stale or unrelated" >&2
    echo "       build tree, e.g. the pre-reorganisation build/); use a new --build directory" >&2; exit 2
  fi
elif [ -e "$build" ] && [ -n "$(ls -A "$build" 2>/dev/null)" ]; then
  echo "ERROR: $build exists, is not empty and is not a CMake build tree; use a new --build directory" >&2; exit 2
fi
echo "build directory: $build"
# Relative path of the frozen Stage-1 TUM runs inside the archive.
archive_runs="2026-11-04_main/output/scratch/rgbd_dataset_freiburg1_xyz"
# The external TUM sequence (only the --rerun-* steps read it).
if [ -n "${SLAM_TUM_ROOT:-}" ]; then
  tum_seq="$SLAM_TUM_ROOT/rgbd_dataset_freiburg1_xyz"
else
  tum_seq="$repo/data/rgbd_dataset_freiburg1_xyz"
fi
if [ ${#reruns[@]} -gt 0 ]; then
  if [ -z "$archive" ]; then
    echo "ERROR: --rerun-* needs the external archive: pass --archive DIR or set SLAM_ARCHIVE_DIR" >&2
    echo "       (the directory holding MANIFEST.sha256 and $archive_runs/; see docs/DATA_AND_ARCHIVE.md)" >&2
    exit 2
  fi
  if [ ! -f "$archive/MANIFEST.sha256" ]; then
    echo "ERROR: archive not available: $archive/MANIFEST.sha256 not found" >&2; exit 2
  fi
  for run in "${reruns[@]}"; do
    if [ ! -d "$archive/$archive_runs/$run" ]; then
      echo "ERROR: archived run not available: $archive/$archive_runs/$run" >&2; exit 2
    fi
  done
  if [ ! -f "$tum_seq/rgb.txt" ]; then
    echo "ERROR: --rerun-* needs the TUM sequence rgbd_dataset_freiburg1_xyz; looked in $tum_seq" >&2
    echo "       (set SLAM_TUM_ROOT to the directory containing it, or place it in data/; docs/DATA_AND_ARCHIVE.md)" >&2
    exit 2
  fi
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fail=0
step() { printf "\n==== %s ====\n" "$1"; }

step "build"
cmake -S "$repo" -B "$build" > /dev/null && cmake --build "$build" -j > /dev/null || exit 1

# Branch integration/teammate: the teammate's scratch_pipeline writes a different output format (extra
# comment lines, fewer correspondence columns), so the byte manifests are checked on the files whose format is
# unchanged: trajectory.txt and keypoints/ (Stage-1 configuration = --stage2-frontend), plus pose_validity.csv
# (Stage-2 configuration = --stage2-frontend --pose-gate).
check_manifest_subset() {  # LABEL MANIFEST FILE_REGEX SCRATCH_PIPELINE_ARGS...
  local label="$1" manifest="$repo/tests/fixtures/$2" regex="$3"; shift 3
  rm -rf "$tmp/bunny"
  (cd "$repo" && "$build/scratch_pipeline" "$@" --out "$tmp/bunny") > "$tmp/bunny.log" 2>&1 || { echo "scratch_pipeline failed"; fail=1; }
  grep -E "$regex" "$manifest" > "$tmp/subset.txt"
  if (cd "$tmp/bunny" && sha256sum --quiet -c "$tmp/subset.txt"); then
    echo "$label: byte-identical ($(wc -l < "$tmp/subset.txt") of $(wc -l < "$manifest") manifest files checked)"
  else
    echo "$label: CHANGED"; fail=1
  fi
}
step "Bunny byte identity (SHA-256 manifest subsets)"
check_manifest_subset "Bunny, Stage-1 configuration vs Stage-1 lock" baseline_bunny_sha256.txt \
  '\./(trajectory\.txt|keypoints/)' --stage2-frontend
check_manifest_subset "Bunny, Stage-2 configuration vs Stage-2 lock" baseline_bunny_gate_sha256.txt \
  '\./(trajectory\.txt|pose_validity\.csv|keypoints/)' --stage2-frontend --pose-gate

step "ctest"
(cd "$repo" && ctest --test-dir "$build" --output-on-failure) || fail=1

step "fixture pairs vs locked baseline"
(cd "$repo" && "$build/scratch_baseline")
st=$?
[ $st -eq 0 ] || [ $st -eq 77 ] || fail=1

for lock in baseline_trajectories.csv baseline_trajectories_gate.csv; do
  step "trajectory ATE of existing runs ($lock)"
  while IFS=, read -r name dir frames ate; do
    [ "$name" = name ] && continue
    if [ ! -f "$repo/$dir/trajectory.csv" ]; then
      printf "%-22s SKIPPED (%s not present)\n" "$name" "$dir"
      continue
    fi
    (cd "$repo" && "$build/scratch_baseline" --ate "$dir" --ate-expected "$ate") || fail=1
  done < "$repo/tests/fixtures/$lock"
done

compare_run() {  # compare_run REFERENCE_DIR NEW_DIR LABEL
  local ok=1 differing=()
  for f in pairs.csv trajectory.csv trajectory.txt summary.txt; do
    cmp -s "$1/$f" "$2/$f" || { echo "  $f DIFFERS"; ok=0; differing+=("$f"); }
  done
  for d in correspondences keypoints; do
    diff -rq "$1/$d" "$2/$d" > /dev/null || { echo "  $d/ DIFFERS"; ok=0; differing+=("$d"); }
  done
  if [ $ok = 1 ]; then echo "$3: byte-identical"; else
    echo "$3: CHANGED (byte comparison, tolerance 0)"; fail=1
    python3 -I -B "$repo/tools/pipeline_analysis/describe_run_diff.py" "$1" "$2" "${differing[@]}" ||
      echo "  (describe_run_diff.py failed; the verdict above stands)"
  fi
}

for run in "${reruns[@]}"; do
  step "re-run TUM $run"
  ref="$archive/$archive_runs/$run"
  # the reference must be the complete, unchanged archived run: every file the archive manifest lists for it
  grep -F "  $archive_runs/$run/" "$archive/MANIFEST.sha256" > "$tmp/ref_manifest.txt"
  if [ ! -s "$tmp/ref_manifest.txt" ] || ! (cd "$archive" && sha256sum --quiet -c "$tmp/ref_manifest.txt"); then
    echo "ERROR: $ref is not listed in, or does not match, $archive/MANIFEST.sha256"; fail=1
    continue
  fi
  echo "reference: $ref ($(wc -l < "$tmp/ref_manifest.txt") files verified against the archive manifest)"
  (cd "$repo" && "$build/scratch_pipeline" --stage2-frontend --tum-frames "$ref/frames.csv" --tum-sequence "$tum_seq" \
     --intrinsics 517.3,516.5,318.6,255.3 --out "$tmp/$run") > /dev/null || fail=1
  compare_run "$ref" "$tmp/$run" "TUM $run"
  rm -rf "${tmp:?}/$run"
done

step "result"
if [ $fail = 0 ]; then echo "Stage-1 baseline: ALL CHECKS PASSED"; else echo "Stage-1 baseline: FAILURES (see above)"; fi
exit $fail
