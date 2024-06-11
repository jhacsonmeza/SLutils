cmake_minimum_required(VERSION 3.24)

# If no target architecture is provided, automatically detect it
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(structured_light_utils LANGUAGES CXX)


# Change the behavior of the FindCUDA removal policy to avoid the
# warning `Policy CMP0146 is not set: The FindCUDA module is removed.`
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
option(SLU_BUILD_SAMPLES "Build code samples" OFF)
option(SLU_PYTHON_BINDINGS "Build Python bindings" OFF)


# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV as external dependency
find_package(OpenCV REQUIRED)


# Use CheckLanguage to check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND SLU_WITH_CUDA)
    # Check if OpenCV has CUDA support
    if(DEFINED OpenCV_CUDA_VERSION)
        message(STATUS "OpenCV was built with CUDA version ${OpenCV_CUDA_VERSION}")
    else()
        message(FATAL_ERROR "OpenCV does not have CUDA support")
    endif()
    
    # Enable CUDA
    enable_language(CUDA)
    message(STATUS "Building SLutils CUDA version")
    
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
    
    set(SLU_BINDINGS_SRC python/gpu_bindings.cpp)
else()
    message(STATUS "Building SLutils CPU version")
    
    set(SLU_SOURCES
        src/fringe_analysis.cpp
        src/graycoding.cpp
        src/phase_graycoding.cpp
        src/centerline.cpp
    )
    
    set(SLU_BINDINGS_SRC python/cpu_bindings.cpp)
endif()


# Create static lib
add_library(SLutils STATIC ${SLU_SOURCES})
# Add include directories and link external libs
target_include_directories(SLutils PUBLIC include)
target_link_libraries(SLutils ${OpenCV_LIBS})
# Enable PIC to avoid problems with the Python bindings
set_target_properties(SLutils PROPERTIES POSITION_INDEPENDENT_CODE ON)


if(SLU_PYTHON_BINDINGS)
    # Check if nanobind was properly cloned in third_party/
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanobind/CMakeLists.txt")
	    message(FATAL_ERROR
		    "nanobind is missing. Python bindings cannot be built without it. "
		    "You probably did not clone the project with \"--recursive\" flag. "
		    "This can be fixed by calling \"git submodule update --init --recursive\"."
	    )
    endif()
    
    # Find Python
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    
    message(STATUS "Building SLutils Python bindings")
    
    # Add nanobind
    add_subdirectory(third_party/nanobind)
    
    nanobind_add_module(sl ${SLU_BINDINGS_SRC})
    target_link_libraries(sl PRIVATE SLutils ${OpenCV_LIBS})
    # Link against CUDA runtime only if we're building the CUDA bindings
    if(CMAKE_CUDA_COMPILER AND SLU_WITH_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(sl PRIVATE CUDA::cudart)
    endif()
endif()


if(SLU_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
