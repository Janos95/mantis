FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest
)

FetchContent_MakeAvailable(doctest)

FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
)

FetchContent_MakeAvailable(eigen)

add_library(P2M
        P2M/KDTree.cpp
        P2M/KDTree.h
        P2M/Model.cpp
        P2M/Model.h
        P2M/RTree.cpp
        P2M/RTree.h
        P2M/tetgen.cpp
        P2M/tetgen.h
        P2M/VoronoiTetgen.cpp
        P2M/VoronoiTetgen.h
)

target_link_libraries(P2M PUBLIC Eigen3::Eigen)
target_include_directories(P2M PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/P2M")

add_executable(tests tests.cpp)

target_link_libraries(tests PRIVATE doctest::doctest mantis P2M)
target_compile_definitions(tests PRIVATE ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets/")

enable_testing()
add_test(NAME MantisTest COMMAND tests)
