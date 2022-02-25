cmake_minimum_required(VERSION 3.11)
project(block_nullspace)

find_package(amgcl)
find_package(VexCL)

function(add_example EXAMPLE)
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp read_problem.cpp)
    target_link_libraries(${EXAMPLE} amgcl::amgcl)
endfunction()

add_example(scalar)
add_example(block)
add_example(ns_scalar)
add_example(ns_hybrid1)
add_example(ns_hybrid2)
add_example(ns_block)

if (TARGET VexCL::OpenCL)
    add_example(ns_hybrid2_vexcl)
    target_link_libraries(ns_hybrid2_vexcl VexCL::OpenCL)

    add_example(ns_block_vexcl)
    target_link_libraries(ns_block_vexcl VexCL::OpenCL)
endif()
