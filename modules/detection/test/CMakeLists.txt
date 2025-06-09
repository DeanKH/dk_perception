add_executable(test_exclude_invalid_triangles
               test_exclude_invalid_triangles.cpp)
target_link_libraries(test_exclude_invalid_triangles
                      PRIVATE dk_perception::detection Catch2::Catch2WithMain)
