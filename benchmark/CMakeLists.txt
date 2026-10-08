find_package(benchmark REQUIRED)

file(GLOB_RECURSE bm_FILES "*.cpp")
foreach(f ${bm_FILES})
    get_filename_component(e ${f} NAME)
    string(REPLACE ".cpp" "" e ${e})
    add_executable(${e} ${f})
    target_link_libraries(${e} PRIVATE 
        benchmark::benchmark
        numerical_calculate
    )
endforeach()
