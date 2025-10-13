add_library(common INTERFACE)

target_include_directories(
  common INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                   $<INSTALL_INTERFACE:include>)

add_library(dk_perception::common ALIAS common)

install(
  TARGETS common
  EXPORT dk_perception-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  EXPORT dk_perception-targets
  FILE dk_perception-targets.cmake
  NAMESPACE dk_perception::
  DESTINATION lib/cmake/dk_perception)
