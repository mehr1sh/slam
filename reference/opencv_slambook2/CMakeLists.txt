# OpenCV/g2o reference library (slambook2-derived). NOT production code.
# Added by the root CMakeLists.txt with -DSLAM_WITH_OPENCV=ON (C++17 and the Release
# default come from the root). Targets, sources and flags are unchanged from the former
# root build; only paths changed in the reorganisation. Executables are written to the
# root build directory (build/<name>) and read ../data from there.
set(OpenGL_GL_PREFERENCE GLVND)  # silence g2o's FindOpenGL GLVND-vs-legacy CMake warning

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(Sophus REQUIRED)

# The teammate's RGB-D registration (rgbd_icp) is defined in experiments/rgbd_icp/;
# the tracking library below links it (src/tracking/icp.cpp wraps rgbd_icp::rigidSVD).

# --- camera: PinholeCamera, pixel2cam() ---
add_library(camera src/camera/camera.cpp)
target_include_directories(camera PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera PUBLIC ${OpenCV_LIBS})

# --- feature_core (production, include/slam/features) is defined once by the root
# CMakeLists.txt through cmake/feature_core.cmake. ---

# --- features: find_feature_matches() (feature_core pipeline + distance filter) ---
add_library(features src/features/features.cpp)
target_include_directories(features PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(features PUBLIC ${OpenCV_LIBS} PRIVATE feature_core)

# --- geometry: 2D-2D pose (F/E/H + recoverPose), triangulation ---
# camera is PRIVATE: pixel2cam() is used inside geometry.cpp only, not part
# of geometry.hpp's own interface.
add_library(geometry src/geometry/geometry.cpp)
target_include_directories(geometry PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(geometry PUBLIC ${OpenCV_LIBS} PRIVATE camera Eigen3::Eigen)
# geometry.cpp includes the production header geometry/fundamental.hpp (Eigen only).
target_include_directories(geometry PRIVATE ${CMAKE_SOURCE_DIR}/include/slam)

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
  PRIVATE optimization g2o::core g2o::stuff g2o::solver_dense camera rgbd_icp)

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

# (The feature_core unit tests features_*_test are defined by the root CMakeLists.txt.)

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

# --- dataset: TUM RGB-D adapter (file lists, association, ground-truth
# interpolation, official intrinsics, depth conversion). Standard library + Eigen.
# Ported verbatim from main (P4), with its unit test and the OpenCV ORB diagnostic.
add_library(dataset src/dataset/tum.cpp)
target_include_directories(dataset PUBLIC include)
target_link_libraries(dataset PUBLIC Eigen3::Eigen)
add_executable(tum_dataset_test tests/unit/tum_dataset_test.cpp)
target_link_libraries(tum_dataset_test PRIVATE dataset)
# OpenCV ORB correspondences on the pairs of a TUM run (diagnostic; separate from the reference features)
add_executable(tum_orb_correspondences tests/tum/tum_orb_correspondences.cpp)
target_link_libraries(tum_orb_correspondences PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

# --- main's ICP variant and its consumers, including the TUM evaluator tum_trajectory_eval
# (reference/icp_main_variant; never linked with this directory's tracking library) ---
option(BUILD_MAIN_ICP_VARIANT "Build main's ICP variant (tracking_main_icp) and tum_trajectory_eval" ON)
if(BUILD_MAIN_ICP_VARIANT)
  add_subdirectory(${CMAKE_SOURCE_DIR}/reference/icp_main_variant ${CMAKE_CURRENT_BINARY_DIR}/icp_main_variant)
endif()
