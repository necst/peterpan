cmake_minimum_required(VERSION 3.10)
project(analyze_lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_library(analyze_lib STATIC
    utils.cpp
    select_subvolume_lib.cpp
)

target_include_directories(analyze_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(analyze_lib PUBLIC ${ITK_LIBRARIES})

# --- OpenMP ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(analyze_lib PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "✅ OpenMP enabled for analyze_lib")
else()
    message(WARNING "⚠️ OpenMP NOT found, build will be single-threaded")
endif()

# --- Ottimizzazioni ---
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose Release or Debug" FORCE)
endif()

message(STATUS "Building analyze_lib in ${CMAKE_BUILD_TYPE} mode")

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling optimization flags for analyze_lib")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
endif()
