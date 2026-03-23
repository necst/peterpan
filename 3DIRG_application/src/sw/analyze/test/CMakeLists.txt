cmake_minimum_required(VERSION 3.10)
project(select_subvolume_itk)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(select_subvolume_itk analyze_and_extract_and_iso_and_resize_optimize_v2.cpp)

# --- Link ITK (keyword form) ---
target_link_libraries(select_subvolume_itk PUBLIC ${ITK_LIBRARIES})

# ===== Optimization Flags =====
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose Release or Debug" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# --- Detect and enable OpenMP ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(select_subvolume_itk PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "✅ OpenMP enabled")
else()
    message(WARNING "⚠️ OpenMP NOT found, build will be single-threaded")
endif()

# --- Release optimizations ---
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling high-performance optimization flags")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
endif()
