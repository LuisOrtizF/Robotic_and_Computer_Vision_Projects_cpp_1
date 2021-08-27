cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(correct_position)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (correct_position main.cpp)
target_link_libraries (correct_position ${PCL_LIBRARIES})