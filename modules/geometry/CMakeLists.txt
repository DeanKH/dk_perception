# get file paths
set(MODULE_NAME geometry)
file(GLOB_RECURSE SOURCES src/*.cpp)
add_library(${MODULE_NAME} SHARED ${SOURCES})
target_include_directories(
  ${MODULE_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include> ${EIGEN3_INCLUDE_DIRS}
         ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

target_link_libraries(
  ${MODULE_NAME} PUBLIC ${OpenCV_LIBRARIES} ${PCL_LIBRARIES}
                        dk_perception::type)

add_library(dk_perception::${MODULE_NAME} ALIAS ${MODULE_NAME})
install(
  TARGETS ${MODULE_NAME}
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
