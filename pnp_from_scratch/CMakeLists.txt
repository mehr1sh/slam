cmake_minimum_required(VERSION 3.16)
project(pnp_from_scratch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen (matrices, vectors, SVD) and the C++ standard library only.
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(pnp_from_scratch src/main.cpp src/pnp.cpp src/projection.cpp)
target_link_libraries(pnp_from_scratch PRIVATE Eigen3::Eigen)
target_compile_options(pnp_from_scratch PRIVATE -Wall -Wextra)

# Milestone 2: the same solver on all 35 pairs (raw correspondences, no RANSAC).
add_executable(pnp_full_sequence src/sequence_main.cpp src/pnp.cpp src/projection.cpp)
target_link_libraries(pnp_full_sequence PRIVATE Eigen3::Eigen)
target_compile_options(pnp_full_sequence PRIVATE -Wall -Wextra)
