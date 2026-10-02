# feature_core: the project-owned feature modules (FAST, intensity-centroid
# orientation, rotated BRIEF, brute-force Hamming matching). C++ standard
# library only. Single definition, included by the root CMakeLists.txt and
# by pnp_from_scratch/CMakeLists.txt.
get_filename_component(_feature_core_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(feature_core
  ${_feature_core_root}/src/features/types.cpp
  ${_feature_core_root}/src/features/fast.cpp
  ${_feature_core_root}/src/features/orientation.cpp
  ${_feature_core_root}/src/features/brief.cpp
  ${_feature_core_root}/src/features/matcher.cpp)
target_include_directories(feature_core PUBLIC ${_feature_core_root}/include)
