SET(execName depth_rmse)
CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)
PROJECT(${execName})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(ZED 3 REQUIRED)
find_package(OpenCV 3.4.1 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
find_package(aruco REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost 1.58.0 COMPONENTS system filesystem REQUIRED)
find_package(ucoslam REQUIRED)

# FIND_PATH(CSPARSE_INCLUDE_DIR NAMES cs.h PATH_SUFFIXES suitesparse csparse EXTERNAL/suitesparse EXTERNAL/csparse) 

include_directories(${CSPARSE_INCLUDE_DIR} /usr/local/include/g2o/solvers/csparse)

find_package( G2O REQUIRED )
message(STATUS "G2O Found: " ${G2O_FOUND})

find_package( CSparse REQUIRED )
message(STATUS "CSPARSE Found: " ${CSPARSE_FOUND})

find_package( Cholmod )
message(STATUS "CHOLMOD Found: " ${CHOLMOD_FOUND})

SET(SPECIAL_OS_LIBS "pthread" "X11") #ZED
set (LIBVISO2_SRC_DIR /home/luis/Instaladores/libviso2/src/lib)
set (CMAKE_CXX_FLAGS "-msse3")

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
include_directories("${LIBVISO2_SRC_DIR}")
include_directories(${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIR})

FILE(GLOB LIBVISO2_SRC_FILES "/home/luis/Instaladores/libviso2/src/lib/*.cpp")

list(APPEND G2O_LIBS
    ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_CLI_LIBRARY} ${G2O_SOLVER_CHOLMOD}
    ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_SOLVER_DENSE}
    ${G2O_SOLVER_PCG} ${G2O_SOLVER_SLAM2D_LINEAR} ${G2O_SOLVER_STRUCTURE_ONLY}
    ${G2O_SOLVER_EIGEN} ${G2O_TYPES_DATA} ${G2O_TYPES_ICP} ${G2O_TYPES_SBA}
    ${G2O_TYPES_SCLAM2D} ${G2O_TYPES_SIM3} ${G2O_TYPES_SLAM2D} ${G2O_TYPES_SLAM3D}
    ${CSPARSE_LIBRARY} ${CHOLMOD_LIBRARIES}
)

ADD_EXECUTABLE(${execName} main.cpp ${LIBVISO2_SRC_FILES})

add_definitions(-std=c++14 -O3) #ZED

SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

TARGET_LINK_LIBRARIES(${execName} ${ZED_LIBS} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} aruco png ${G2O_LIBS} ${ucoslam_LIBS})