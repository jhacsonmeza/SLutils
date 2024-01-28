cmake_minimum_required(VERSION 3.24)

# If not target architecture is provided, automatically detect it
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(phase_unwrapping LANGUAGES CXX)


# Handle the FindCUDA removal policy change for OpenCV:
# `Policy CMP0146 is not set: The FindCUDA module is removed.`
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# Options
option(WITH_CUDA "Use CUDA version" ON)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV as external dependency
find_package(OpenCV REQUIRED)


# Include CheckLanguage to check if CUDA is enabled
include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND WITH_CUDA)
    enable_language(CUDA)
    message(STATUS "Using CUDA!")
    
    # set CUDA C++17 standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    set(PHASE_UNWRAP_SRC
        src/fringe_analysis.cu
    )
else()
    message(STATUS "CUDA is not supported")
    
    set(PHASE_UNWRAP_SRC
        src/fringe_analysis.cpp
        src/graycoding.cpp
        src/phase_graycoding.cpp
    )
endif()


# Create static lib
add_library(PhaseUnwrap STATIC ${PHASE_UNWRAP_SRC})
# Add include directories and link external libs
target_include_directories(PhaseUnwrap PUBLIC include)
target_link_libraries(PhaseUnwrap ${OpenCV_LIBS})
