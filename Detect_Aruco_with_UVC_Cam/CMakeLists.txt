SET(execName aruco_webcam)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(${execName})

if(COMMAND cmake_policy) #ZED
	cmake_policy(SET CMP0003 OLD) #ZED
	cmake_policy(SET CMP0015 OLD) #ZED
endif(COMMAND cmake_policy) #ZED

#set(EXECUTABLE_OUTPUT_PATH ./${CMAKE_BUILD_TYPE}) #FOR DEBUG

find_package(ZED 3 REQUIRED) #ZED
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED) #ZED
find_package(OpenCV 3.4.1 REQUIRED)
find_package(PCL REQUIRED)
find_package(aruco REQUIRED)

add_definitions(-Wno-format-extra-args) #OPENCV
SET(SPECIAL_OS_LIBS "pthread" "X11") #OPENCV

include_directories(${ZED_INCLUDE_DIRS}) #ZED
include_directories(${CUDA_INCLUDE_DIRS}) #ZED
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) #ZED
include_directories("/usr/local/Aria/include") #ARIA

link_directories(${ZED_LIBRARY_DIR}) #ZED
link_directories(${CUDA_LIBRARY_DIRS}) #ZED
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories("/usr/local/Aria/lib") #ARIA

FILE(GLOB_RECURSE SRC_FILES "main.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
add_definitions(-std=c++11 -g -O3) #ZED

TARGET_LINK_LIBRARIES(${execName}
  ${ZED_LIBRARIES} #ZED
  ${SPECIAL_OS_LIBS} #ZED
  ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED} #ZED
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  libAria.so libpthread.so libdl.so librt.so #ARIA
  aruco
)
