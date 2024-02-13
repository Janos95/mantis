cmake_minimum_required(VERSION 3.26)
project(mantis)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(MANTIS_BUILD_EXAMPLES "Build the examples" OFF)
option(MANTIS_BUILD_BENCH "Build the benchmarks" OFF)
option(MANTIS_BUILD_TESTS "Build the tests" ON)

add_library(mantis
        Delaunay_psm.h
        Delaunay_psm.cpp
        mantis.h
        mantis.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(mantis PRIVATE "/arch:AVX")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(mantis PRIVATE "-mavx")
endif()

target_include_directories(mantis PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

#enable address sanitizer
#target_compile_options(mantis PUBLIC -fsanitize=address)
#target_link_options(mantis PUBLIC -fsanitize=address)

#if (MANTIS_BUILD_TESTS)
#    enable_testing()
#    add_subdirectory(tests)
#endif()

if (MANTIS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

#if (MANTIS_BUILD_BENCH)
#    add_subdirectory(bench)
#endif()