# MIT License

# Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide Conficconi, Eleonora D'Arnese

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

ECHO=@echo

.PHONY: help

APP_NAME ?= trilli_app
TARGET := hw
#PLATFORM := xilinx_vck5000_gen4x8_xdma_2_202210_1
PLATFORM=xilinx_vck5000_gen4x8_qdma_2_202220_1

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make build_hw [TARGET=hw_emu]"
	$(ECHO) ""
	$(ECHO) "  make build_sw"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) ""

CFG = default.cfg
CONFIG = $(abspath $(CFG))
include ${CONFIG}

# can be either STEP or TX
TASK := STEP

hw_dependencies := compile_aie compile_data_movers
ifeq ($(TASK),STEP)
    hw_dependencies += compile_krnl_mutual_info
else ifneq ($(TASK),TX)
    $(error TASK must be either STEP or TX: `$(TASK)` is not valid)
endif

build_hw: $(hw_dependencies) hw_link

compile_aie:
	make -C ./aie aie_compile

compile_krnl_mutual_info:
	make -C ./mutual_info compile TARGET=$(TARGET) PLATFORM=$(PLATFORM)

compile_data_movers:
	make -C ./data_movers compile TARGET=$(TARGET) PLATFORM=$(PLATFORM)

hw_link:
	make -C ./hw all TARGET=$(TARGET) PLATFORM=$(PLATFORM)

# Build software object
build_sw: 
	make -C ./sw all

build_app:
	make -C ./3DIRG_application build_host EXECUTABLE=$(APP_NAME)

config:
	$(info using config file $(CONFIG))
	$(info )
	$(info ************ Generating configuration files ************)
	$(info - TASK             $(TASK))
	$(info - DIMENSION        $(DIMENSION))
	$(info - N_ROWS           $(N_ROWS))
	$(info - N_COLS           $(N_COLS))
	$(info - N_COUPLES        $(N_COUPLES))
	$(info - N_COUPLES_MAX    $(N_COUPLES_MAX))
	$(info - HIST_PE          $(HIST_PE))
	$(info - ENTROPY_PE       $(ENTROPY_PE))
	$(info - INT_PE           $(INT_PE))
	$(info - PIXELS_PER_READ  $(PIXELS_PER_READ))
	$(info - DS_PE            $(DS_PE))
	$(info ********************************************************)
	$(info )
	cd common/generator && python3 generator.py -vts -nrows $(N_ROWS) -ncols $(N_COLS) -id $(DIMENSION) -ncm $(N_COUPLES_MAX) -pe $(HIST_PE) -pen $(ENTROPY_PE) -intpe $(INT_PE) -op ../ -ppr $(PIXELS_PER_READ) -dspe $(DS_PE)
	mv common/mutual_info.hpp mutual_info/include/hw/mutualInfo
	make -C ./aie generate_input_data
	make -C ./hw generate_config
	make -C ./data_movers generate_movers_body
	make -C ./sw switch_dataset

config_and_build:
	echo make config
	echo make build_hw
	echo make build_sw

testbench:
	make -C ./data_movers testbench_all

testbench_c:
	make -C ./aie aie_compile_x86
	make -C ./data_movers testbench_all

testbench_noaie:
	make -C ./data_movers testbench_noaie

NAME := hw_build
XCLBIN := hw/overlay_hw.xclbin 
pack:
	mkdir -p build/$(NAME)/dataset
	mkdir -p build/$(NAME)/dataset_output
	mkdir -p build/$(NAME)/dataset_sw_output
	rm -f sw/dataset/*.png
	cp -r sw/dataset/** build/$(NAME)/dataset/
	cp sw/host_overlay.exe build/$(NAME)/
	cp $(XCLBIN) build/$(NAME)/overlay_hw.xclbin
	cp sw/generate_dataset.sh build/$(NAME)/
	cp sw/remove_dataset.sh build/$(NAME)/
	cp scripts/run_scaling_depth.sh build/$(NAME)/
	cp scripts/run_for_SoA_comparison.sh build/$(NAME)/
	cp $(CONFIG) build/$(NAME)/
	$(info )
	$(info Packed application in build/$(NAME)/ using bitstream $(XCLBIN))
	$(info )

pack_step:
	mkdir -p build/$(NAME)/dataset
	mkdir -p build/$(NAME)/dataset_output
	mkdir -p build/$(NAME)/dataset_sw_output
	rm -f sw/dataset/*.png
	cp -r sw/dataset/** build/$(NAME)/dataset/
	cp sw/host_overlay.exe build/$(NAME)/
	cp $(XCLBIN) build/$(NAME)/overlay_hw.xclbin
	cp sw/generate_dataset.sh build/$(NAME)/
	cp sw/remove_dataset.sh build/$(NAME)/
	cp scripts/run_registration_step.sh build/$(NAME)/
	cp $(CONFIG) build/$(NAME)/
	$(info )
	$(info Packed application in build/$(NAME)/ using bitstream $(XCLBIN))
	$(info )

######################### AE PeterPan application packing #########################
PACK_APP_DIR ?= build/PeterPan_APP
APP_SRC_DIR ?= 3DIRG_application
APP_DST_DIR := $(PACK_APP_DIR)/3DIRG_application
BITSTREAM_SRC ?= bitstreams/PeterPan.xclbin
BITSTREAM_DST := $(APP_DST_DIR)/src/sw/PeterPan.xclbin

pack_peterpan_app:
	mkdir -p $(PACK_APP_DIR)
	rm -rf $(APP_DST_DIR)
	cp -r $(APP_SRC_DIR) $(PACK_APP_DIR)/
	cp $(BITSTREAM_SRC) $(BITSTREAM_DST)
	$(info )
	$(info Packed application in $(PACK_APP_DIR))
	$(info Bitstream copied to $(BITSTREAM_DST))
	$(info )

pack_trilli_app: build_app
	mkdir -p build/$(NAME)/PET/
	mkdir -p build/$(NAME)/CT/
	cp scripts/README.txt build/$(NAME)/PET/
	cp scripts/README.txt build/$(NAME)/CT/
	cp $(XCLBIN) build/$(NAME)/overlay_hw.xclbin
	cp -r 3DIRG_application/$(APP_NAME) build/$(NAME)/
	cp exec.sh build/$(NAME)/
	cp 3DIRG_application/generate_dataset.sh build/$(NAME)/
	cp 3DIRG_application/remove_dataset.sh build/$(NAME)/
	cp sw/dataset/duplicate_slices.sh build/$(NAME)/PET/
	cp sw/dataset/duplicate_slices.sh build/$(NAME)/CT/
	cp -r sw/dataset/size_512 build/$(NAME)/PET/
	cp -r sw/dataset/size_512 build/$(NAME)/CT/
	cp $(CONFIG) build/$(NAME)/
	$(info )
	$(info Packed application in build/$(NAME)/ using bitstream $(XCLBIN))
	$(info )

pack_for_hacc:
	mkdir -p build/hacc/$(NAME)/sw
	cp -r sw/dataset build/hacc/$(NAME)/sw/
	cp -r sw/include build/hacc/$(NAME)/sw/
	cp -r sw/host_code.cpp build/hacc/$(NAME)/sw/
	cp -r sw/Makefile build/hacc/$(NAME)/sw/
	cp -r hw/overlay_hw.xclbin build/hacc/$(NAME)/sw/
	cp -r common build/hacc/$(NAME)/
	cp default.cfg build/hacc/$(NAME)/
	cp Makefile build/hacc/$(NAME)/

build_and_pack_app:
	$(info )
	$(info *********************** Building ***********************)
	$(info - NAME          $(NAME))
	$(info - TARGET        $(TARGET))
	$(info - PLATFORM      $(PLATFORM))
	$(info ********************************************************)
	$(info )
	make config
	make build_hw
	make build_app
	make pack_app

build_and_pack:
	$(info )
	$(info *********************** Building ***********************)
	$(info - NAME          $(NAME))
	$(info - TARGET        $(TARGET))
	$(info - PLATFORM      $(PLATFORM))
	$(info ********************************************************)
	$(info )
	make config
	make build_hw
	make build_sw
	make pack

get_config_string:
	@echo "$(TASK)_$(N_ROWS)x$(N_COLS)_IPE$(INT_PE)_DSPE$(DS_PE)_PPR$(PIXELS_PER_READ)_HPE$(HIST_PE)_EPE$(ENTROPY_PE)"

python-venv: .venv
.venv:
	python3 -m venv $@
	. $@/bin/activate && \
	python3 -m pip install -r requirements.txt

# Clean objects
clean: clean_aie clean_mutual_info clean_data_movers clean_hw clean_sw

clean_mutual_info:
	make -C ./mutual_info clean
clean_aie:
	make -C ./aie clean

clean_krnl_histogram:
	make -C ./krnl_histogram clean

clean_data_movers:
	make -C ./data_movers clean

clean_hw:
	make -C ./hw clean

clean_sw: 
	make -C ./sw clean

clean_app:
	make -C ./3DIRG_application EXECUTABLE=$(APP_NAME) clean