SET(execName capture)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(${execName})

find_package(ZED 3 REQUIRED) #ZED
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED) #ZED
find_package(OpenCV 3.4.1 REQUIRED)
find_package(freenect2 REQUIRED)
find_package(PCL REQUIRED)

add_definitions(-Wno-format-extra-args) #ZED
SET(SPECIAL_OS_LIBS "pthread" "X11") #ZED

include_directories(${ZED_INCLUDE_DIRS}) #ZED
include_directories(${CUDA_INCLUDE_DIRS}) #ZED
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($ENV{OPENNI_INCLUDE}) #KINECT V1
include_directories("/usr/include/libusb-1.0/") #KINECT V1
include_directories(${freenect2_INCLUDE_DIRS}) 
include_directories(${PCL_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR}) #ZED
link_directories(${CUDA_LIBRARY_DIRS}) #ZED
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES "main.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
add_definitions(-std=c++11 -g -O3) #ZED AND OPENCV 3.4

TARGET_LINK_LIBRARIES(${execName}
  ${ZED_LIBRARIES} #ZED
  ${SPECIAL_OS_LIBS} #ZED
  ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED} #ZED
  ${OpenCV_LIBRARIES}
  ${OPENNI_LIB} #KINECT V1
  ${freenect2_LIBRARIES} #KINECT V2
  ${PCL_LIBRARIES}
)
