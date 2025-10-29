add_executable(test_line test_line.cpp)
target_link_libraries(test_line PRIVATE dk_perception::${MODULE_NAME}
                                        Catch2::Catch2WithMain)

add_executable(test_triangle_constructor test_triangle_constructor.cpp)
target_link_libraries(
  test_triangle_constructor PRIVATE dk_perception::${MODULE_NAME}
                                    Catch2::Catch2WithMain)
