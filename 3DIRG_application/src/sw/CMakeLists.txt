cmake_minimum_required(VERSION 3.16)
project(peterpan_pyramidal_sw LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type = Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

# Options
option(HW_REG "Enable HW_REG compile definition" OFF)
option(TRILLI_X "Use Versal3DIR_TRILLIX implementation" OFF)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module" OFF)

# ============================================================
#  ITK
# ============================================================
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# ============================================================
#  OpenCV
# ============================================================
find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
"OpenCV not found.

To fix this:
  1) Make sure the OpenCV development package is installed
     (the one that provides OpenCVConfig.cmake).
  2) Pass to CMake the directory containing OpenCVConfig.cmake, e.g.:

     cmake -S . -B build -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4

     or add the installation prefix to CMAKE_PREFIX_PATH, e.g.:

     cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/opencv

Then re-run CMake.")
endif()

message(STATUS "✅ OpenCV found: ${OpenCV_VERSION}")

# ============================================================
#  OpenMP
# ============================================================
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "✅ OpenMP enabled")
    set(OpenMP_LIB OpenMP::OpenMP_CXX)
else()
    message(WARNING "⚠️ OpenMP not found — single-threaded")
    set(OpenMP_LIB "")
endif()

# ============================================================
#  Common paths / sources
# ============================================================
set(PROJ_ROOT ${CMAKE_SOURCE_DIR})

set(COMMON_INCLUDE_DIRS
    ${PROJ_ROOT}
    ${PROJ_ROOT}/analyze
    ${PROJ_ROOT}/app
    ${PROJ_ROOT}/core
    ${PROJ_ROOT}/core/domain
    ${PROJ_ROOT}/include
    ${PROJ_ROOT}/include/image_utils
    ${PROJ_ROOT}/include/software_mi
    ${PROJ_ROOT}/infastructure
    ${PROJ_ROOT}/interfaces
)

set(COMMON_CORE_SOURCES
    ${PROJ_ROOT}/peterpan_runner.cpp
    ${PROJ_ROOT}/analyze/select_subvolume_lib.cpp
    ${PROJ_ROOT}/analyze/utils.cpp
    ${PROJ_ROOT}/include/image_utils/image_utils.cpp
    ${PROJ_ROOT}/include/software_mi/software_mi.cpp
)

set(EXECUTABLE_SOURCES
    ${PROJ_ROOT}/pyramidal_main.cpp
    ${COMMON_CORE_SOURCES}
)

# ============================================================
#  XRT setup (only if HW_REG)
# ============================================================
if(HW_REG)
    message(STATUS "⚙️ HW_REG enabled")

    if(NOT DEFINED ENV{XILINX_XRT})
        message(FATAL_ERROR "❌ XRT environment missing. Run 'source <XRT>/setup.sh'.")
    endif()

    set(XRT_ROOT $ENV{XILINX_XRT})
    message(STATUS "📦 XRT root: ${XRT_ROOT}")
else()
    message(STATUS "⚙️ HW_REG disabled → SW_REG enabled")
endif()

# ============================================================
#  Executable
# ============================================================
add_executable(peterpan_pyramidal_sw ${EXECUTABLE_SOURCES})

target_include_directories(peterpan_pyramidal_sw PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(peterpan_pyramidal_sw
    PRIVATE
        ${ITK_LIBRARIES}
        ${OpenCV_LIBS}
        ${OpenMP_LIB}
)

if(HW_REG)
    target_include_directories(peterpan_pyramidal_sw PRIVATE
        ${PROJ_ROOT}/include/versal_3dir
        ${XRT_ROOT}/include
    )

    target_link_directories(peterpan_pyramidal_sw PRIVATE ${XRT_ROOT}/lib)

    target_link_libraries(peterpan_pyramidal_sw PRIVATE
        xilinxopencl
        xrt_coreutil
        xrt_core
        uuid
    )

    if(TRILLI_X)
        message(STATUS "➡️ Adding HW-specific tile sources: Versal3DIR_TRILLIX.cpp")
        target_sources(peterpan_pyramidal_sw PRIVATE
            ${PROJ_ROOT}/include/versal_3dir/Versal3DIR_TRILLIX.cpp
        )
        target_compile_definitions(peterpan_pyramidal_sw PRIVATE TRILLI_X)
    else()
        message(STATUS "➡️ Adding HW-specific tile sources: Versal3DIR.cpp")
        target_sources(peterpan_pyramidal_sw PRIVATE
            ${PROJ_ROOT}/include/versal_3dir/Versal3DIR.cpp
        )
    endif()

    target_compile_definitions(peterpan_pyramidal_sw PRIVATE HW_REG)
else()
    target_compile_definitions(peterpan_pyramidal_sw PRIVATE SW_REG)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling optimization flags")
    target_compile_options(peterpan_pyramidal_sw PRIVATE
        -O3
        -march=native
        -ffast-math
        -funroll-loops
    )
    target_link_options(peterpan_pyramidal_sw PRIVATE -flto)
endif()

# ============================================================
#  Optional Python bindings
# ============================================================
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "🐍 Building Python bindings with pybind11")

    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 REQUIRED)

    set(PYBIND_SOURCES
        ${PROJ_ROOT}/peterpan_pybind.cpp
        ${COMMON_CORE_SOURCES}
    )

    pybind11_add_module(peterpan_cpp ${PYBIND_SOURCES})

    target_include_directories(peterpan_cpp PRIVATE ${COMMON_INCLUDE_DIRS})

    target_link_libraries(peterpan_cpp
        PRIVATE
            ${ITK_LIBRARIES}
            ${OpenCV_LIBS}
            ${OpenMP_LIB}
    )

    if(HW_REG)
        target_include_directories(peterpan_cpp PRIVATE
            ${PROJ_ROOT}/include/versal_3dir
            ${XRT_ROOT}/include
        )

        target_link_directories(peterpan_cpp PRIVATE ${XRT_ROOT}/lib)

        target_link_libraries(peterpan_cpp PRIVATE
            xilinxopencl
            xrt_coreutil
            xrt_core
            uuid
        )

        if(TRILLI_X)
            target_sources(peterpan_cpp PRIVATE
                ${PROJ_ROOT}/include/versal_3dir/Versal3DIR_TRILLIX.cpp
            )
            target_compile_definitions(peterpan_cpp PRIVATE TRILLI_X)
        else()
            target_sources(peterpan_cpp PRIVATE
                ${PROJ_ROOT}/include/versal_3dir/Versal3DIR.cpp
            )
        endif()

        target_compile_definitions(peterpan_cpp PRIVATE HW_REG)
    else()
        target_compile_definitions(peterpan_cpp PRIVATE SW_REG)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(peterpan_cpp PRIVATE
            -O3
            -march=native
            -ffast-math
            -funroll-loops
        )
        target_link_options(peterpan_cpp PRIVATE -flto)
    endif()
endif()