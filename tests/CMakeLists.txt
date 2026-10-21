add_executable(tests tests.cpp)

include(P2M)
target_link_libraries(tests PRIVATE mantis P2M)
target_compile_definitions(tests PRIVATE ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets/")

add_test(NAME tests COMMAND tests)
