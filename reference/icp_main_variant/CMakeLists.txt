# main's ICP variant (implementation worktree, branch main, uncommitted state of 2026-11-04), preserved
# verbatim: the book's SVD pose_estimation_3d3d and g2o bundleAdjustment, plus FitRigid and
# EstimateRigidRansac. Added by reference/opencv_slambook2/CMakeLists.txt (inherits its OpenCV, g2o and
# Sophus variables and its camera/features/geometry/optimization targets).
#
# Never linked together with the teammate's variant (reference/opencv_slambook2/src/tracking/icp.cpp,
# which wraps rgbd_icp): both define pose_estimation_3d3d and bundleAdjustment. Both variants also have a
# tracking/icp.hpp, so every target here puts include/ (main's header) BEFORE the reference include path.
set(_ref ${CMAKE_SOURCE_DIR}/reference/opencv_slambook2)
set(_main_icp_include ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Same sources, flags and links as main's tracking library; only icp.cpp is main's.
add_library(tracking_main_icp
  ${_ref}/src/tracking/pnp.cpp
  src/icp.cpp
  ${_ref}/src/tracking/essential_matrix.cpp
  ${_ref}/src/tracking/fast.cpp
  ${_ref}/src/tracking/brief.cpp
  ${_ref}/src/tracking/hamming_matching.cpp)
target_compile_options(tracking_main_icp PRIVATE -msse4)
target_include_directories(tracking_main_icp BEFORE PUBLIC ${_main_icp_include})
target_include_directories(tracking_main_icp PUBLIC ${_ref}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tracking_main_icp
  PUBLIC ${OpenCV_LIBS} Eigen3::Eigen Sophus::Sophus
  PRIVATE optimization g2o::core g2o::stuff g2o::solver_dense camera)

# name: executable; src: source file. Linked as main's add_test_exe (camera features geometry tracking).
function(add_main_icp_exe name src)
  add_executable(${name} ${src})
  target_include_directories(${name} BEFORE PRIVATE ${_main_icp_include})
  target_link_libraries(${name} PRIVATE camera features geometry tracking_main_icp)
endfunction()

add_main_icp_exe(icp_rigid_test tests/icp_rigid_test.cpp)
# The OpenCV TUM evaluator (ported from main, P4): the reference pipeline on a TUM sequence.
add_main_icp_exe(tum_trajectory_eval ${_ref}/tests/tum/tum_trajectory_eval.cpp)
target_link_libraries(tum_trajectory_eval PRIVATE dataset)
# main's versions of the two synthetic trajectory tests (RANSAC -> g2o on the inliers; --out-dir).
# They include "../report_utils.hpp", resolved here through the reference tests/synthetic directory.
add_main_icp_exe(slam_trajectory_test_main_icp tests/slam_trajectory_test.cpp)
add_main_icp_exe(trajectory_validation_test_main_icp tests/trajectory_validation_test.cpp)
foreach(t slam_trajectory_test_main_icp trajectory_validation_test_main_icp)
  target_include_directories(${t} PRIVATE ${_ref}/tests/synthetic)
endforeach()
