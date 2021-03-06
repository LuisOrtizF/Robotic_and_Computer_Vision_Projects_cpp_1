SET(execName aruco_image)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(${execName})

IF(COMMAND CMAKE_POLICY) #ZED
	CMAKE_POLICY(SET CMP0003 OLD) #ZED
	CMAKE_POLICY(SET CMP0015 OLD) #ZED
ENDIF(COMMAND CMAKE_POLICY) #ZED

#set(EXECUTABLE_OUTPUT_PATH ./${CMAKE_BUILD_TYPE}) #FOR DEBUG

FIND_PACKAGE(ZED 3 REQUIRED) #ZED
FIND_PACKAGE(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED) #ZED
FIND_PACKAGE(OpenCV 3.4.1 REQUIRED)
FIND_PACKAGE(PCL REQUIRED)
FIND_PACKAGE(aruco REQUIRED )

ADD_DEFINITIONS(-Wno-format-extra-args) #OPENCV
SET(SPECIAL_OS_LIBS "pthread" "X11") #OPENCV

# set(auruco_Dir "/usr/local/share/aruco")

INCLUDE_DIRECTORIES(${ZED_INCLUDE_DIRS}) #ZED
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) #ZED
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include) #ZED
INCLUDE_DIRECTORIES("/usr/local/Aria/include") #ARIA

LINK_DIRECTORIES(${ZED_LIBRARY_DIR}) #ZED
LINK_DIRECTORIES(${CUDA_LIBRARY_DIRS}) #ZED
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})
LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
LINK_DIRECTORIES("/usr/local/Aria/lib") #ARIA

FILE(GLOB_RECURSE SRC_FILES "main.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
ADD_DEFINITIONS(-std=c++11 -g -O3) #ZED

TARGET_LINK_LIBRARIES(${execName}
  ${ZED_LIBRARIES} #ZED
  ${SPECIAL_OS_LIBS} #ZED
  ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED} #ZED
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  libAria.so libpthread.so libdl.so librt.so #ARIA
  aruco
)