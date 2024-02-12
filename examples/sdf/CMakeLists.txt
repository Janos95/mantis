
find_package(TBB CONFIG)

add_executable(sdf main.cpp implicit_meshing.cpp implicit_meshing.h)
target_link_libraries(sdf PRIVATE mantis utils polyscope)

if (NOT TARGET TBB::tbb)
    target_compile_definitions(sdf PRIVATE USE_SERIAL)
else()
    target_link_libraries(sdf PRIVATE TBB::tbb)
    target_compile_definitions(sdf PUBLIC USE_TBB)
endif()

target_include_directories(sdf PRIVATE external)
target_compile_definitions(sdf PRIVATE ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../assets/")