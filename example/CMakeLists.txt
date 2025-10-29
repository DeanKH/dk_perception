include(FetchContent)
FetchContent_Declare(
  rerun_sdk
  URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

add_executable(ortho_projection_example ortho_projection.cpp)
target_include_directories(
  ortho_projection_example PUBLIC include ${EIGEN3_INCLUDE_DIRS}
                                  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ortho_projection_example dk_perception::image_processing
                      dk_perception::type)

add_executable(pcproc_example pointcloud_splitlines_example.cpp)
target_include_directories(pcproc_example PUBLIC include ${EIGEN3_INCLUDE_DIRS}
                                                 ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pcproc_example dk_perception::pcproc dk_perception::type)

add_executable(superpoint_onnx_demo demo.cpp)
target_link_libraries(superpoint_onnx_demo PRIVATE ${OpenCV_LIBS}
                                                   dk_perception::dnn)

add_executable(deeplsd_example deeplsd_example.cpp)
target_link_libraries(
  deeplsd_example PRIVATE ${OpenCV_LIBS} dk_perception::dnn dk_perception::type
                          dk_perception::detection)

# Performance profiling option
option(USE_PERF "Enable gperftools profiling" ON)

if(USE_PERF)
  find_library(GPERFTOOLS_PROFILER profiler)
  if(GPERFTOOLS_PROFILER)
    message(STATUS "Found gperftools profiler: ${GPERFTOOLS_PROFILER}")
  else()
    message(FATAL_ERROR "gperftools profiler not found!")
  endif()
endif()

add_executable(radial_extremum_detect_example
               radial_extremum_detect_example.cpp)
target_include_directories(
  radial_extremum_detect_example PUBLIC include ${EIGEN3_INCLUDE_DIRS}
                                        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(radial_extremum_detect_example dk_perception::detection
                      dk_perception::type rerun_sdk)

add_executable(radial_splitter_example radial_splitter_example.cpp)

target_include_directories(
  radial_splitter_example PUBLIC include ${EIGEN3_INCLUDE_DIRS}
                                 ${OpenCV_INCLUDE_DIRS})
target_link_libraries(radial_splitter_example dk_perception::detection
                      dk_perception::type rerun_sdk)

if(USE_PERF)
  target_link_libraries(radial_extremum_detect_example ${GPERFTOOLS_PROFILER})
  # target_compile_definitions(radial_extremum_detect_example PRIVATE USE_PERF)
endif()

add_executable(tsdf_reconstruction_example tsdf_reconstruction_example.cpp)
target_include_directories(
  tsdf_reconstruction_example PUBLIC include ${EIGEN3_INCLUDE_DIRS}
                                     ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tsdf_reconstruction_example dk_perception::reconstruction
                      dk_perception::detection dk_perception::type rerun_sdk)

add_executable(tsdf_reconstruction_on_box_example
               tsdf_reconstruction_on_box_example.cpp)
target_include_directories(
  tsdf_reconstruction_on_box_example PUBLIC include ${EIGEN3_INCLUDE_DIRS}
                                            ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
  tsdf_reconstruction_on_box_example dk_perception::reconstruction
  dk_perception::detection dk_perception::type rerun_sdk)
