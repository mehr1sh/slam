cmake_minimum_required(VERSION 3.16)
project(rgbd_visual_slam CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Every executable lands directly in the build directory (build/<name>), as before the
# reorganisation; the reference demos rely on it (they read ../data from build/).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# OFF (default): the production pipeline only -- Eigen and the C++ standard library.
# ON: also the OpenCV/g2o reference library (reference/), the OpenCV experiments
# (experiments/rgbd_icp, experiments/orb_opencv) and main's ICP variant.
option(SLAM_WITH_OPENCV "Build the OpenCV/g2o reference library and the OpenCV experiments" OFF)

# Eigen (matrices, vectors, SVD) and the C++ standard library only.
find_package(Eigen3 REQUIRED NO_MODULE)

# Project-owned feature modules (FAST, orientation, rotated BRIEF, Hamming);
# a single target definition, also used by experiments/rgbd_icp/icp_cpp.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/feature_core.cmake)

# Production library: PNG decoding, front end, projection, linear PnP, essential matrix,
# RANSAC, refinement, validity, per-pair pipeline.
add_library(scratch_core src/pipeline/png.cpp src/pipeline/frontend.cpp src/pipeline/projection.cpp
            src/pipeline/pnp.cpp src/pipeline/essential.cpp src/pipeline/ransac.cpp src/pipeline/refine.cpp
            src/pipeline/validity.cpp src/pipeline/pipeline.cpp)
target_include_directories(scratch_core PUBLIC include/slam/pipeline)
target_link_libraries(scratch_core PUBLIC Eigen3::Eigen feature_core)
target_compile_options(scratch_core PRIVATE -Wall -Wextra)

# Milestone 1: linear PnP on one pair from the frozen reference correspondences (self-checking).
add_executable(pnp_from_scratch apps/milestones/main.cpp src/pipeline/pnp.cpp src/pipeline/projection.cpp)
target_include_directories(pnp_from_scratch PRIVATE include/slam/pipeline)
target_link_libraries(pnp_from_scratch PRIVATE Eigen3::Eigen)
target_compile_options(pnp_from_scratch PRIVATE -Wall -Wextra)

# Milestone 2: the same solver on all 35 pairs (raw correspondences, no RANSAC).
add_executable(pnp_full_sequence apps/milestones/sequence_main.cpp src/pipeline/pnp.cpp src/pipeline/projection.cpp)
target_include_directories(pnp_full_sequence PRIVATE include/slam/pipeline)
target_link_libraries(pnp_full_sequence PRIVATE Eigen3::Eigen)
target_compile_options(pnp_full_sequence PRIVATE -Wall -Wextra)

# Production application: the complete self-contained pipeline (images -> trajectory).
add_executable(scratch_pipeline apps/pipeline_main.cpp)
target_link_libraries(scratch_pipeline PRIVATE scratch_core)
target_compile_options(scratch_pipeline PRIVATE -Wall -Wextra)

# Read-only experiment (not part of the pipeline): the geometric stage with and
# without depth-edge correspondences (docs/components/pipeline.md, "Depth edges").
add_executable(depth_edge_experiment experiments/depth_edge/depth_edge_experiment.cpp)
target_link_libraries(depth_edge_experiment PRIVATE scratch_core)
target_compile_options(depth_edge_experiment PRIVATE -Wall -Wextra)

# Stage-1 measurement / regression harness and Stage-3C experiments (ported from main; branch
# integration/teammate): fixture locks, multi-seed evaluation, integration configurations.
add_library(scratch_baseline_lib src/evaluation/baseline.cpp)
target_include_directories(scratch_baseline_lib PUBLIC include/slam/evaluation)
target_link_libraries(scratch_baseline_lib PUBLIC scratch_core)
target_compile_options(scratch_baseline_lib PRIVATE -Wall -Wextra)
add_library(scratch_experiment_lib src/evaluation/experiment.cpp)
target_include_directories(scratch_experiment_lib PUBLIC include/slam/evaluation)
target_link_libraries(scratch_experiment_lib PUBLIC scratch_core)
target_compile_options(scratch_experiment_lib PRIVATE -Wall -Wextra)
add_executable(scratch_baseline apps/baseline_main.cpp)
target_link_libraries(scratch_baseline PRIVATE scratch_baseline_lib scratch_experiment_lib)
target_compile_options(scratch_baseline PRIVATE -Wall -Wextra)
find_package(Threads REQUIRED)
add_executable(scratch_experiments apps/experiments_main.cpp)
target_link_libraries(scratch_experiments PRIVATE scratch_experiment_lib scratch_baseline_lib Threads::Threads)
target_compile_options(scratch_experiments PRIVATE -Wall -Wextra)

# ---- tests (ctest --test-dir build) ----
# The programs find the repository root by walking up from the working directory
# (data/synthetic_bunny/intrinsics.txt), so every test runs from the repository root.
enable_testing()
set(_repo ${CMAKE_CURRENT_SOURCE_DIR})
# the feature modules' own unit tests
foreach(t features_fast_test features_orientation_test features_brief_test features_matcher_test
        features_pyramid_test features_matcher_ranked_test)
  add_executable(${t} tests/unit/${t}.cpp)
  target_link_libraries(${t} PRIVATE feature_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
foreach(t test_png test_frontend test_essential test_ransac test_pipeline_0_1 test_multiscale_0_1
        test_refine test_validity test_ranked_frontend)
  add_executable(${t} tests/pipeline/${t}.cpp)
  target_include_directories(${t} PRIVATE tests/pipeline)
  target_link_libraries(${t} PRIVATE scratch_core)
  add_test(NAME ${t} COMMAND ${t} WORKING_DIRECTORY ${_repo})
endforeach()
# the milestone programs are self-checking too (exit status 0 = all counted checks pass)
add_test(NAME milestone1_pnp_from_scratch COMMAND pnp_from_scratch WORKING_DIRECTORY ${_repo})
add_test(NAME milestone2_pnp_full_sequence COMMAND pnp_full_sequence WORKING_DIRECTORY ${_repo})
# (written into the build tree so that ctest never overwrites a run in results/runs/pipeline)
add_test(NAME milestone3_scratch_pipeline COMMAND scratch_pipeline --out ${CMAKE_CURRENT_BINARY_DIR}/milestone3_pipeline
         WORKING_DIRECTORY ${_repo})
add_executable(test_baseline tests/pipeline/test_baseline.cpp)
target_include_directories(test_baseline PRIVATE tests/pipeline)
target_link_libraries(test_baseline PRIVATE scratch_baseline_lib)
add_test(NAME test_baseline COMMAND test_baseline WORKING_DIRECTORY ${_repo})
add_executable(test_experiment tests/pipeline/test_experiment.cpp)
target_include_directories(test_experiment PRIVATE tests/pipeline)
target_link_libraries(test_experiment PRIVATE scratch_experiment_lib)
add_test(NAME test_experiment COMMAND test_experiment WORKING_DIRECTORY ${_repo})
# the locked Stage-1/2 fixtures in main's Stage-2 configuration (exit 77 = TUM images absent: skipped;
# the images are read from $SLAM_TUM_ROOT when it is set, docs/DATA_AND_ARCHIVE.md)
add_test(NAME scratch_baseline_lock COMMAND scratch_baseline --out ${CMAKE_CURRENT_BINARY_DIR}/scratch_baseline_report
         WORKING_DIRECTORY ${_repo})
set_tests_properties(scratch_baseline_lock PROPERTIES SKIP_RETURN_CODE 77)
# the frozen fixture directories must match their SHA-256 manifests (detects any overwrite of the
# Bunny export, the TUM export, the 12 archived ATE-lock runs, the Blender scene's T0 trajectory or the
# reference-pipeline baseline read by the milestone programs and test_png; read-only; tests/fixtures/README.md)
add_test(NAME fixture_checksums
         COMMAND sh -c "(cd stage1_bunny_export && sha256sum --quiet -c ../baseline_bunny_sha256.txt) && (cd stage1_tum_exports/rgbd_dataset_freiburg1_xyz/n100_stride4 && sha256sum --quiet -c ../../../stage1_tum_export_sha256.txt) && (cd stage1_ate_runs && sha256sum --quiet -c ../stage1_ate_runs_sha256.txt) && (cd bunny_t0_scene && sha256sum --quiet -c ../bunny_t0_scene_sha256.txt) && (cd reference_baseline && sha256sum --quiet -c ../reference_baseline_sha256.txt)"
         WORKING_DIRECTORY ${_repo}/tests/fixtures)

add_executable(test_improvements tests/pipeline/test_improvements.cpp)
target_link_libraries(test_improvements PRIVATE scratch_core)
add_test(NAME scratch_improvements COMMAND test_improvements)

# Controlled frontend diagnostics; uses the same C++ feature functions as estimation.
add_executable(feature_audit apps/feature_audit.cpp)
target_link_libraries(feature_audit PRIVATE scratch_core)
add_test(NAME feature_audit_pair_0_1 COMMAND feature_audit --repo ${_repo})

# ---- OpenCV components (not production) ----
if(SLAM_WITH_OPENCV)
  add_subdirectory(experiments/rgbd_icp)
  add_subdirectory(reference/opencv_slambook2)
  include(cmake/orb_benchmark.cmake)
endif()
