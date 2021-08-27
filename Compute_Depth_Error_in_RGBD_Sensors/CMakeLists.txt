SET(execName depth_error)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(${execName})

#find_package(ZED 2 REQUIRED) #ZED
#find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED) #ZED
find_package(OpenCV 3.4.1 REQUIRED)
find_package(PCL 1.8 REQUIRED)

add_definitions(-Wno-format-extra-args) #ZED
SET(SPECIAL_OS_LIBS "pthread" "X11") #ZED

#include_directories(${ZED_INCLUDE_DIRS}) #ZED
#include_directories(${CUDA_INCLUDE_DIRS}) #ZED
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

#link_directories(${ZED_LIBRARY_DIR}) #ZED
#link_directories(${CUDA_LIBRARY_DIRS}) #ZED
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES "main.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
add_definitions(-std=c++11 -g -O3)  #ZED AND OPENCV 3.4

include_directories ("${PROJECT_SOURCE_DIR}/dCP")
add_subdirectory (dCP)

TARGET_LINK_LIBRARIES(${execName} 
#    ${ZED_LIBRARIES} #ZED
#    ${SPECIAL_OS_LIBS} #ZED
#    ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED} #ZED
    ${OpenCV_LIBRARIES} 
    ${PCL_LIBRARIES}
    dCP
)
