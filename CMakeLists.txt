cmake_minimum_required(VERSION 3.10)
project(visual_slam CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(OpenGL_GL_PREFERENCE GLVND)  # silence g2o's FindOpenGL GLVND-vs-legacy CMake warning

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(Sophus REQUIRED)

# --- camera: PinholeCamera, pixel2cam() ---
add_library(camera src/camera/camera.cpp)
target_include_directories(camera PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera PUBLIC ${OpenCV_LIBS})

# --- feature_core: project-owned FAST, intensity-centroid orientation,
# rotated BRIEF, brute-force Hamming matching. Standard library only.
# Defined once in cmake/feature_core.cmake (also used by pnp_from_scratch/). ---
include(cmake/feature_core.cmake)

# --- features: find_feature_matches() (feature_core pipeline + distance filter) ---
add_library(features src/features/features.cpp)
target_include_directories(features PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(features PUBLIC ${OpenCV_LIBS} PRIVATE feature_core)

# --- geometry: 2D-2D pose (F/E/H + recoverPose), triangulation ---
# camera is PRIVATE: pixel2cam() is used inside geometry.cpp only, not part
# of geometry.hpp's own interface.
add_library(geometry src/geometry/geometry.cpp)
target_include_directories(geometry PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(geometry PUBLIC ${OpenCV_LIBS} PRIVATE camera)

# --- optimization: shared g2o SE3 pose vertex (header-only) ---
add_library(optimization INTERFACE)
target_include_directories(optimization INTERFACE include)
target_link_libraries(optimization INTERFACE g2o::core Sophus::Sophus)

# --- tracking: PnP and ICP pose estimation, with g2o/Sophus BA refinement.
# g2o itself is PRIVATE: VertexPose/solver setup are implementation details
# of pnp.cpp/icp.cpp, not part of pnp.hpp/icp.hpp's interface (which only
# expose cv::Mat/Sophus::SE3d/Eigen types).
add_library(tracking
  src/tracking/pnp.cpp
  src/tracking/icp.cpp
  src/tracking/essential_matrix.cpp
  src/tracking/fast.cpp
  src/tracking/brief.cpp
  src/tracking/hamming_matching.cpp)
# hamming_matching.cpp uses _mm_popcnt_u32 (SSE4.2), same as this repo's
# existing tests/unit/orb_from_scratch_test.cpp -- same compile flag reused.
target_compile_options(tracking PRIVATE -msse4)
target_include_directories(tracking PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tracking
  PUBLIC ${OpenCV_LIBS} Eigen3::Eigen Sophus::Sophus
  PRIVATE optimization g2o::core g2o::stuff g2o::solver_dense camera)

# --- render: PLY mesh I/O, CPU software rasterizer, orbit camera trajectory,
# and combined mesh+trajectory+frustum PLY visualization export.
# camera is PUBLIC: PinholeCamera appears directly in rasterizer.hpp's
# RenderMesh() signature. Eigen/Sophus are PUBLIC for the same reason
# (Vector3f/SE3d types appear in mesh.hpp/trajectory.hpp/visualization.hpp).
add_library(render
  src/render/mesh.cpp
  src/render/ply_mesh_io.cpp
  src/render/rasterizer.cpp
  src/render/trajectory.cpp
  src/render/visualization.cpp)
target_include_directories(render PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(render PUBLIC ${OpenCV_LIBS} Eigen3::Eigen Sophus::Sophus camera)

add_executable(render_bunny_test tests/synthetic/render_bunny_test.cpp)
target_link_libraries(render_bunny_test PRIVATE render)

add_executable(synthetic_pose_eval_test tests/synthetic/synthetic_pose_eval_test.cpp)
target_link_libraries(synthetic_pose_eval_test PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Sophus::Sophus)
target_include_directories(synthetic_pose_eval_test PRIVATE ${OpenCV_INCLUDE_DIRS})

# name: target/executable name (matches the .cpp file's own basename).
# folder: subdirectory under tests/ -- unit (self-contained, no external
# dataset), integration (validated against data/tum_sample/), synthetic
# (validated against data/synthetic_bunny/).
function(add_test_exe name folder)
  add_executable(${name} tests/${folder}/${name}.cpp)
  target_link_libraries(${name} PRIVATE camera features geometry tracking)
endfunction()

add_test_exe(feature_matching_test integration)
add_test_exe(two_view_pose_test integration)
add_test_exe(triangulation_test integration)
add_test_exe(pnp_test integration)
add_test_exe(icp_test integration)
add_test_exe(pnp_synthetic_test synthetic)
add_test_exe(pnp_multiframe_test synthetic)
add_test_exe(icp_synthetic_test synthetic)
add_test_exe(slam_trajectory_test synthetic)
add_test_exe(trajectory_validation_test synthetic)
add_test_exe(trajectory_diagnostics_test synthetic)

# Unit tests of the project-owned feature modules: they link feature_core only.
foreach(t features_fast_test features_orientation_test features_brief_test features_matcher_test)
  add_executable(${t} tests/unit/${t}.cpp)
  target_link_libraries(${t} PRIVATE feature_core)
endforeach()

# Self-contained book demos: no dependency on src/, so they don't link the
# library targets above.
add_executable(orb_from_scratch_test tests/unit/orb_from_scratch_test.cpp)
target_include_directories(orb_from_scratch_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orb_from_scratch_test PRIVATE ${OpenCV_LIBS})
target_compile_options(orb_from_scratch_test PRIVATE -msse4)

add_executable(gauss_newton_curve_fit_test tests/unit/gauss_newton_curve_fit_test.cpp)
target_include_directories(gauss_newton_curve_fit_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gauss_newton_curve_fit_test PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(g2o_curve_fit_test tests/unit/g2o_curve_fit_test.cpp)
target_include_directories(g2o_curve_fit_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(g2o_curve_fit_test PRIVATE ${OpenCV_LIBS} Eigen3::Eigen g2o::core g2o::stuff g2o::solver_dense)

add_executable(essential_matrix_test tests/unit/essential_matrix_test.cpp)
target_link_libraries(essential_matrix_test PRIVATE camera tracking Eigen3::Eigen)

add_executable(fast_test tests/unit/fast_test.cpp)
target_link_libraries(fast_test PRIVATE tracking)

add_executable(brief_test tests/unit/brief_test.cpp)
target_link_libraries(brief_test PRIVATE tracking)

add_executable(vo_pipeline_test tests/unit/vo_pipeline_test.cpp)
target_link_libraries(vo_pipeline_test PRIVATE camera tracking Eigen3::Eigen)
