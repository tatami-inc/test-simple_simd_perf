cmake_minimum_required(VERSION 3.24)

project(testing
    VERSION 1.0.0
    DESCRIPTION "A C++ interface for assorted matrix representations"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(testing test.cpp)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG master
)

FetchContent_MakeAvailable(tatami)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG 291c587
)

FetchContent_MakeAvailable(cli11)

find_package(Threads REQUIRED)

target_compile_options(testing PRIVATE -mavx2 -fno-tree-vectorize)
target_link_libraries(testing tatami CLI11::CLI11 Threads::Threads)
