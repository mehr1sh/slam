cmake_minimum_required(VERSION 3.12)
project(standalone_orb_extractor LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_library(feature_core STATIC
  src/features/types.cpp
  src/features/fast.cpp
  src/features/orientation.cpp
  src/features/brief.cpp
  src/features/matcher.cpp
  src/features/pyramid.cpp
  src/features/multiscale.cpp)
target_include_directories(feature_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(orb_extract src/orb_extract.cpp)
target_link_libraries(orb_extract PRIVATE feature_core)
