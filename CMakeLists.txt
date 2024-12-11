cmake_minimum_required(VERSION 3.5)
project(MathMorph_seq LANGUAGES C CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
find_package(CUDAToolkit REQUIRED)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include/")
#include_directories(/usr/local/cuda-12.6/)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(MathMorph_seq main.cpp
        MMops.cpp
        MMops.h
)

set_target_properties(MathMorph_seq PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_library(OpenCV SHARED IMPORTED)
set_target_properties(OpenCV PROPERTIES
        IMPORTED_LOCATION ""
        IMPORTED_IMPLIB ""
        INTERFACE_INCLUDE_DIRECTORIES ""
)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(MathMorph_seq ${OpenCV_LIBS})





