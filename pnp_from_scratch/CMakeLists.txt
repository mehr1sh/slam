cmake_minimum_required(VERSION 3.16)
project(pnp_from_scratch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen (matrices, vectors, SVD) and the C++ standard library only.
find_package(Eigen3 REQUIRED NO_MODULE)

# Project-owned feature modules (FAST, orientation, rotated BRIEF, Hamming);
# the same single target definition the repository root uses.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/feature_core.cmake)

# Scratch library: PNG decoding, front end, projection, linear PnP, essential matrix,
# RANSAC, per-pair pipeline.
add_library(scratch_core src/png.cpp src/frontend.cpp src/projection.cpp src/pnp.cpp src/essential.cpp src/ransac.cpp src/refine.cpp src/pipeline.cpp)
target_include_directories(scratch_core PUBLIC src)
target_link_libraries(scratch_core PUBLIC Eigen3::Eigen feature_core)
target_compile_options(scratch_core PRIVATE -Wall -Wextra)

add_executable(pnp_from_scratch src/main.cpp src/pnp.cpp src/projection.cpp)
target_link_libraries(pnp_from_scratch PRIVATE Eigen3::Eigen)
target_compile_options(pnp_from_scratch PRIVATE -Wall -Wextra)

# Milestone 2: the same solver on all 35 pairs (raw correspondences, no RANSAC).
add_executable(pnp_full_sequence src/sequence_main.cpp src/pnp.cpp src/projection.cpp)
target_link_libraries(pnp_full_sequence PRIVATE Eigen3::Eigen)
target_compile_options(pnp_full_sequence PRIVATE -Wall -Wextra)

# Milestone 3: the complete self-contained pipeline (images -> 36-pose trajectory).
add_executable(scratch_pipeline src/pipeline_main.cpp)
target_link_libraries(scratch_pipeline PRIVATE scratch_core)
target_compile_options(scratch_pipeline PRIVATE -Wall -Wextra)

# Read-only experiment (not part of the pipeline): the geometric stage with and
# without depth-edge correspondences (docs/migration/05_depth_edges).
add_executable(depth_edge_experiment experiments/depth_edge_experiment.cpp)
target_link_libraries(depth_edge_experiment PRIVATE scratch_core)
target_compile_options(depth_edge_experiment PRIVATE -Wall -Wextra)

# ---- tests (ctest --test-dir build) ----
enable_testing()
# the feature modules' own unit tests, built from the repository's sources
foreach(t features_fast_test features_orientation_test features_brief_test features_matcher_test
        features_pyramid_test)
  add_executable(${t} ${CMAKE_CURRENT_SOURCE_DIR}/../tests/unit/${t}.cpp)
  target_link_libraries(${t} PRIVATE feature_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
foreach(t test_png test_frontend test_essential test_ransac test_pipeline_0_1 test_multiscale_0_1
        test_refine)
  add_executable(${t} tests/${t}.cpp)
  target_include_directories(${t} PRIVATE tests)
  target_link_libraries(${t} PRIVATE scratch_core)
  add_test(NAME ${t} COMMAND ${t} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
# the milestone programs are self-checking too (exit status 0 = all counted checks pass)
add_test(NAME milestone1_pnp_from_scratch COMMAND pnp_from_scratch WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME milestone2_pnp_full_sequence COMMAND pnp_full_sequence WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME milestone3_scratch_pipeline COMMAND scratch_pipeline WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
