
cmake_policy(SET CMP0167 OLD)

find_package(Boost REQUIRED COMPONENTS system)

set(SRCS_RUN_PROCESSES
    run_processes.cpp
    subprocess.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/path_to_executables.cpp
)

add_executable( run_processes ${SRCS_RUN_PROCESSES} )
target_link_libraries(run_processes PRIVATE Boost::headers Boost::system)
target_include_directories(run_processes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/path_to_executables.cpp INPUT ${CMAKE_CURRENT_SOURCE_DIR}/path_to_executables.cpp.in )
