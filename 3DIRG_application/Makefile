# MIT License

#######################################################
# directory stuffs
TOP ?= $(shell pwd)
EXECUTABLE ?= trilli_app
EXECUTABLE_SW ?= trilli_app_sw

SRC_DIR=$(TOP)/src
SW_DIR=$(SRC_DIR)/sw
PYBIND_WRAPPER_DIR=$(TOP)/pybind_wrapper
BUILD_DIR=$(TOP)/build

#######################################################
# Compiler and flags
CC := g++
CXXFLAGS = -std=c++17 -O2 -fPIC -Wno-deprecated-declarations

# Paths for HW_REG (if needed)
INCL_DRVR := -I$(XILINX_XRT)/include
XILINX_XRT ?= /opt/xrt/2022.1/opt/xilinx/xrt
LDFLAGS := -L$(XILINX_XRT)/lib -lxrt_coreutil
# OpenCV flags

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)
# pybind11 include (for pybind build)
PYBIND11_INCLUDE = $(shell python3 -m pybind11 --includes)

#######################################################
# Sources for host builds (non-pybind)
HOST_SRCS := $(SW_DIR)/main.cpp \
             $(SW_DIR)/rigid_registration_runner.cpp \
             $(SW_DIR)/include/image_utils/image_utils.cpp
             #$(SW_DIR)/include/software_mi/software_mi.cpp

#######################################################
# Sources for pybind module build
PYBIND_SRCS := $(SW_DIR)/rigid_registration_runner.cpp \
               $(PYBIND_WRAPPER_DIR)/wrapper.cpp \
               $(SW_DIR)/include/image_utils/image_utils.cpp

#######################################################
# Targets

all: build_host

build_host: $(EXECUTABLE)
$(EXECUTABLE): $(HOST_SRCS)
	$(CC) $(CXXFLAGS) $(INCL_DRVR) -o $@ $^ $(LDFLAGS) $(OPENCV_CFLAGS) $(OPENCV_LIBS) -DHW_REG

build_host_sw: $(EXECUTABLE_SW)
$(EXECUTABLE_SW): $(HOST_SRCS)
	$(CC) $(CXXFLAGS) -o $@ $^ $(OPENCV_CFLAGS) $(OPENCV_LIBS)

PYBIND_MODE ?= hw

build_pybind: $(BUILD_DIR)/trilli_wrapper.so

ifeq ($(PYBIND_MODE), hw)
$(BUILD_DIR)/trilli_wrapper.so: $(PYBIND_SRCS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CXXFLAGS) -DHW_REG $(PYBIND11_INCLUDE) -I$(SW_DIR) \
	$(INCL_DRVR) $(OPENCV_CFLAGS) -shared -o $@ $^ \
	$(LDFLAGS) $(OPENCV_LIBS)
else
$(BUILD_DIR)/trilli_wrapper.so: $(PYBIND_SRCS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CXXFLAGS) $(PYBIND11_INCLUDE) -I$(SW_DIR) \
	$(OPENCV_CFLAGS) -shared -o $@ $^ \
	$(OPENCV_LIBS)
endif

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE_SW)
	rm -rf $(BUILD_DIR)
	rm -f ./overlay_hw.xclbin ./overlay_hw_emu.xclbin

.PHONY: all clean build_host build_host_sw build_pybind
