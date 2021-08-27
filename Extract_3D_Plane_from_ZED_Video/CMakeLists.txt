SET(execName extract_plane)
CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)
PROJECT(${execName})

find_package(ZED 3 REQUIRED)
find_package(OpenCV 3.4.1 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
find_package(PCL 1.8 REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

ADD_EXECUTABLE(${execName} main.cpp)
add_definitions(-std=c++11 -g -O3)
add_definitions(${PCL_DEFINITIONS})

SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED})

TARGET_LINK_LIBRARIES(${execName} 
${ZED_LIBS} 
${OpenCV_LIBRARIES}
${PCL_LIBRARIES}
)
