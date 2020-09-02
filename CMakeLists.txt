cmake_minimum_required(VERSION 3.15)
project(19_opencv)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(19_opencv main.cpp)

target_link_libraries(19_opencv ${OpenCV_LIBS})