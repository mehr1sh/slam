# Teammate's corrected RGB-D registration (provisional ICP; no final ICP decision yet).
# Built from the root with -DSLAM_WITH_OPENCV=ON; also independently buildable through
# icp_cpp/ (pixi run icp-build). Target definitions unchanged from the former root CMakeLists.txt.
find_package(OpenCV REQUIRED)

add_library(rgbd_icp src/rgbd_icp.cpp)
target_include_directories(rgbd_icp PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rgbd_icp PUBLIC ${OpenCV_LIBS} Eigen3::Eigen feature_core)
add_executable(icp_sequence apps/icp_sequence.cpp)
target_link_libraries(icp_sequence PRIVATE rgbd_icp)
add_executable(rgbd_icp_test tests/rgbd_icp_test.cpp)
target_link_libraries(rgbd_icp_test PRIVATE rgbd_icp)
add_test(NAME rgbd_icp_geometry COMMAND rgbd_icp_test)
