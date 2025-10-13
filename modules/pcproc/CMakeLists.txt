set(MODULE_NAME pcproc)
add_library(${MODULE_NAME} INTERFACE)

target_include_directories(
  ${MODULE_NAME}
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include> ${EIGEN3_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

target_link_libraries(${MODULE_NAME} INTERFACE ${OpenCV_LIBRARIES}
                                               ${PCL_LIBRARIES})
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
