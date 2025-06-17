# P2M (https://github.com/Martin-JC-Xu/P2M)
# License: AGPL-3.0
if(TARGET P2M::P2M)
    return()
endif()

message(STATUS "Third-party: creating target 'P2M::P2M'")

include(CPM)
CPMAddPackage(
    URI "gh:Martin-JC-Xu/P2M#ae56568ece94651ff51685d242dae3f3e4a40c26"
    DOWNLOAD_ONLY YES
)

add_library(P2M
    ${P2M_SOURCE_DIR}/KDTree.cpp
    ${P2M_SOURCE_DIR}/KDTree.h
    ${P2M_SOURCE_DIR}/Model.cpp
    ${P2M_SOURCE_DIR}/Model.h
    ${P2M_SOURCE_DIR}/RTree.cpp
    ${P2M_SOURCE_DIR}/RTree.h
    ${P2M_SOURCE_DIR}/tetgen.cpp
    ${P2M_SOURCE_DIR}/tetgen.h
    ${P2M_SOURCE_DIR}/VoronoiTetgen.cpp
    ${P2M_SOURCE_DIR}/VoronoiTetgen.h
)
add_library(P2M::P2M ALIAS P2M)

target_include_directories(P2M PUBLIC "${P2M_SOURCE_DIR}")

# Disable warning: 'sprintf' is deprecated: This function is provided for
# compatibility reasons only. Due to security concerns inherent in the design
# of sprintf(3), it is highly recommended that you use snprintf(3) instead.
target_compile_options(P2M PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-deprecated-declarations>
)

include(eigen)
target_link_libraries(P2M PUBLIC Eigen3::Eigen)

include(onetbb)
target_link_libraries(P2M PUBLIC TBB::tbb)

# Folder name for IDE
set_target_properties(P2M PROPERTIES FOLDER "ThirdParty")