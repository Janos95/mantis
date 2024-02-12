add_executable(wos main.cpp)
target_link_libraries(wos PRIVATE mantis utils)
#assets folder
target_compile_definitions(wos PRIVATE ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../assets/")