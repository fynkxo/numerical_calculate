cmake_minimum_required(VERSION 3.28)
project(numerial_calculate
  LANGUAGES CXX C
  VERSION 0.0.1
)

set(CMAKE_CXX_STANDARD 23 )
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(xsimd REQUIRED)

file(GLOB_RECURSE hFiles "*.h" "*.hpp")
add_library(numerical_calculate INTERFACE 
  ${hFiles}
)
target_include_directories(numerical_calculate INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include/numerical_calculate>
)

add_subdirectory(benchmark)
add_subdirectory(tests)
