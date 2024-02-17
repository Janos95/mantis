add_executable(mantis_benchmark benchmark.cpp)

add_subdirectory(fcpw)

target_link_libraries(mantis_benchmark PRIVATE mantis P2M)

target_link_libraries(mantis_benchmark PRIVATE fcpw)
target_include_directories(mantis_benchmark PUBLIC ${FCPW_EIGEN_INCLUDES})
target_include_directories(mantis_benchmark PUBLIC ${FCPW_ENOKI_INCLUDES})

target_compile_definitions(mantis_benchmark PRIVATE ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets/")
