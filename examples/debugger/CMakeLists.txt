add_executable(mantis_debugger
        debugger.cpp
)

target_link_libraries(mantis_debugger PRIVATE mantis utils)
