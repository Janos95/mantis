cmake_minimum_required(VERSION 3.26)
project(mantis)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
include(CheckCSourceCompiles)

option(MANTIS_BUILD_EXAMPLES "Build the examples" OFF)
option(MANTIS_BUILD_BENCH "Build the benchmarks" OFF)
option(MANTIS_BUILD_TESTS "Build the tests" ON)

add_library(mantis
        Delaunay_psm.h
        Delaunay_psm.cpp
        mantis.h
        mantis.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(mantis PRIVATE "-march=native")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(mantis PRIVATE "/arch:AVX512")
endif()

target_include_directories(mantis PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if (MANTIS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (MANTIS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (MANTIS_BUILD_BENCH)
    add_subdirectory(bench)
endif()