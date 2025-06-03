add_executable(ortho_projection_example ortho_projection.cpp)
target_include_directories(
  ortho_projection_example PUBLIC include ${EIGEN3_INCLUDE_DIRS}
                                  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ortho_projection_example dk_perception::image_processing
                      dk_perception::type)

add_executable(pointcloud_processing_example pointcloud_splitlines_example.cpp)
target_include_directories(
  pointcloud_processing_example PUBLIC include ${EIGEN3_INCLUDE_DIRS}
                                       ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pointcloud_processing_example
                      dk_perception::pointcloud_processing dk_perception::type)

add_executable(superpoint_onnx_demo demo.cpp)
target_link_libraries(superpoint_onnx_demo PRIVATE ${OpenCV_LIBS}
                                                   dk_perception::dnn)
