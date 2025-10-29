cmake_minimum_required(VERSION 3.20)
project(dk_perception)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build tests" ON)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} \
-isystem /usr/include/c++/11 \
-isystem /usr/include/x86_64-linux-gnu/c++/11 \
-isystem /usr/include/x86_64-linux-gnu/c++/11/bits \
")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/dk_perception-config.cmake
  INSTALL_DESTINATION lib/cmake/dk_perception)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dk_perception-config.cmake
        DESTINATION lib/cmake/dk_perception)

if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1)
  FetchContent_MakeAvailable(Catch2)
endif()

file(
  GLOB MODULE_DIRS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "modules/*")
foreach(MODULE_DIR ${MODULE_DIRS})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR}/CMakeLists.txt)
      add_subdirectory(${MODULE_DIR})
    endif()
  endif()
endforeach()

option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
  add_subdirectory(example)
endif()
