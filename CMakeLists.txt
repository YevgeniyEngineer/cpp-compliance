cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cpp_compliance)

add_executable(serialization src/serialization.cpp)
add_executable(memcpy src/memcpy.cpp)