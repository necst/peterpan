cmake_minimum_required(VERSION 3.10)

# Set the project name
project(TestRegistration)
link_directories(/usr/local/lib)

# Set build type to Release (for optimization)
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the optimization flag -O3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find ITK package
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Check if SRCS variable is provided; if not, show a message
if(NOT DEFINED SRCS)
  message(FATAL_ERROR "Please specify the source file using -DSRCS=<filename.cpp>")
endif()

# Add the executable using the SRCS variable
add_executable(TestRegistration ${SRCS})

# Link ITK libraries
target_link_libraries(TestRegistration ${ITK_LIBRARIES})