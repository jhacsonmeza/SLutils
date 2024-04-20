cmake_minimum_required(VERSION 3.24)

# If no target architecture is provided, automatically detect it
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(structured_light_unwrapping LANGUAGES CXX)


# Handle FindCUDA removal policy, which is needed for OpenCV:
# `Policy CMP0146 is not set: The FindCUDA module is removed.`
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# If not specified, set build type to release
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not specified - defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()


# Options
option(SLU_WITH_CUDA "Build CUDA version" ON)
option(SLU_BUILD_SAMPLES "Build phase unwrapping samples" OFF)


# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV as external dependency
find_package(OpenCV REQUIRED)


# Use CheckLanguage to check if CUDA is enabled
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND SLU_WITH_CUDA)
    enable_language(CUDA)
    message(STATUS "Compiling the CUDA version")
    
    # Define HAVE_CUDA macro
    add_compile_definitions(HAVE_CUDA)
    
    # set CUDA C++17 standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    set(SLU_SOURCES
        src/fringe_analysis.cu
        src/graycoding.cu
        src/phase_graycoding.cu
    )
else()
    message(STATUS "Compiling the CPU version")
    
    set(SLU_SOURCES
        src/fringe_analysis.cpp
        src/graycoding.cpp
        src/phase_graycoding.cpp
        src/centerline.cpp
    )
endif()


# Create static lib
add_library(SLunwrap STATIC ${SLU_SOURCES})
# Add include directories and link external libs
target_include_directories(SLunwrap PUBLIC include)
target_link_libraries(SLunwrap ${OpenCV_LIBS})


if(SLU_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
