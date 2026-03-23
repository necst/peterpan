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

CONFIG = ../default.cfg
include ${CONFIG}

help::
	$(ECHO) ""
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<hw/hw_emu>"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""

PLATFORM := xilinx_vck5000_gen4x8_qdma_2_202220_1
TARGET := hw

FREQUENCY := 250
XOCCLFLAGS := --kernel_frequency $(FREQUENCY) --platform $(PLATFORM) -t $(TARGET)  -s -g

AIE_OBJ := ../aie/libadf.a
XOS     := ../data_movers/setup_aie_$(TARGET).xo 
XOS     += ../data_movers/fetcher_A_$(TARGET).xo
XOS     += ../data_movers/fetcher_B_$(TARGET).xo
XOS     += ../data_movers/fetcher_C_$(TARGET).xo
XOS     += ../data_movers/fetcher_D_$(TARGET).xo
XOS     += ../data_movers/scheduler_IPE_$(TARGET).xo
# XOS 	+= ../data_movers/setup_interpolator_$(TARGET).xo
# XOS 	+= ../data_movers/pixels_merger_$(TARGET).xo


XSA_OBJ := overlay_$(TARGET).xsa
XCLBIN  := overlay_$(TARGET).xclbin

.phony: clean

TASK ?= STEP
ifeq ($(TASK),TX)
	CFG_TEMPLATE_IN := xclbin_overlay_onlyTX.template.cfg
	CFG_TEMPLATE_OUT := xclbin_overlay.cfg
	XOS 	+= ../data_movers/writer_$(TARGET).xo

else ifeq ($(TASK),STEP)
	CFG_TEMPLATE_IN := xclbin_overlay.template.cfg
	CFG_TEMPLATE_OUT := xclbin_overlay.cfg
	XOS 	+= ../data_movers/setup_mi_$(TARGET).xo
	XOS     += ../mutual_info/mutual_information_master_$(TARGET).xo 

else
	$(error TASK must be either STEP or TX: `$(TASK)` is not valid)
endif

all: generate_config $(XCLBIN)

$(XCLBIN): $(XSA_OBJ) $(AIE_OBJ)
	v++ -p -t $(TARGET) -f $(PLATFORM) $^ -o $@ --package.boot_mode=ospi

$(XSA_OBJ): $(XOS) $(AIE_OBJ)
	v++ -l $(XOCCFLAGS) $(XOCCLFLAGS) --config $(CFG_TEMPLATE_OUT) -o $@ $^

generate_config:
	$(info Generating configuration file $(CFG_TEMPLATE_OUT) from $(CFG_TEMPLATE_IN) with INT_PE=$(INT_PE))
	python3 config_generator.py -in $(CFG_TEMPLATE_IN) -out $(CFG_TEMPLATE_OUT) -intpe $(INT_PE) --task $(TASK)

clean:
	$(RM) -r _x .Xil .ipcache *.ltx *.log *.jou *.info *.xclbin *.xo.* *.str *.xsa *.cdo.bin *bif *BIN *.package_summary *.link_summary
	
