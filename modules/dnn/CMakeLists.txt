find_package(onnxruntime REQUIRED)

file(GLOB_RECURSE DNN_SOURCES src/*.cpp)
add_library(dnn SHARED ${DNN_SOURCES})
target_include_directories(
  dnn PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
             $<INSTALL_INTERFACE:include> ${Eigen3_INCLUDE_DIRS})

target_link_libraries(dnn PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

add_library(dk_perception::dnn ALIAS dnn)

# install weights directory
install(DIRECTORY weights/ DESTINATION share/dk_perception/weights)

install(
  TARGETS dnn
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
