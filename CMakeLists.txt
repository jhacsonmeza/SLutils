cmake_minimum_required(VERSION 3.10)
project(phase_unwrapping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)


# Structure Light tools library
add_library(PhaseUnwrap STATIC
    src/fringe_analysis.cpp
    src/graycoding.cpp
    src/phase_graycoding.cpp
)
target_include_directories(PhaseUnwrap PUBLIC include)
target_link_libraries(PhaseUnwrap ${OpenCV_LIBS})
