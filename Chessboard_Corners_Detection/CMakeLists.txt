cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project("corners")

find_package(OpenCV 3.4.1 REQUIRED)
find_package(Threads REQUIRED)

add_compile_options(-std=c++11)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})