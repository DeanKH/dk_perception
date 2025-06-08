find_package(onnxruntime REQUIRED)

add_library(dnn SHARED src/superpoint.cpp)
target_include_directories(
  dnn PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
             $<INSTALL_INTERFACE:include> ${Eigen3_INCLUDE_DIRS})

target_link_libraries(dnn PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

# add_executable(superpoint_onnx_demo src/demo.cpp) target_link_libraries(
# superpoint_onnx_demo PRIVATE superpoint_onnx ${OpenCV_LIBS}
# onnxruntime::onnxruntime)

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
