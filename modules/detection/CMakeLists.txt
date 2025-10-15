# get file paths

file(GLOB_RECURSE SOURCES src/**/*.cpp)

add_library(detection SHARED ${SOURCES})
target_include_directories(
  detection
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include> ${EIGEN3_INCLUDE_DIRS}
         ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

target_link_libraries(
  detection
  PUBLIC ${OpenCV_LIBRARIES}
         ${PCL_LIBRARIES}
         dk_perception::type
         dk_perception::geometry
         dk_perception::dnn
         dk_perception::pcproc
         dk_perception::optimization)

add_library(dk_perception::detection ALIAS detection)
install(
  TARGETS detection
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

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
