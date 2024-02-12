FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git)

FetchContent_MakeAvailable(polyscope)

add_library(utils INTERFACE)
target_sources(utils INTERFACE util.h)
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(utils INTERFACE polyscope)

add_subdirectory(wos)
add_subdirectory(sdf)
#add_subdirectory(debugger)
