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

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) ""
	$(ECHO) "  make compile"
	$(ECHO) "      Command to generate xo kernel file"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."

PLATFORM := xilinx_vck5000_gen4x8_qdma_2_202220_1


TARGET := hw

CONFIG = ../default.cfg
include ${CONFIG}

.phony: clean

################## hardware build 

XOCCFLAGS := --platform $(PLATFORM) -t $(TARGET)  -s -g

TASK := STEP

# XOS := setup_aie_$(TARGET).xo fetcher_A_$(TARGET).xo fetcher_B_$(TARGET).xo fetcher_C_$(TARGET).xo fetcher_D_$(TARGET).xo setup_interpolator_$(TARGET).xo pixels_merger_$(TARGET).xo
XOS := setup_aie_$(TARGET).xo fetcher_A_$(TARGET).xo fetcher_B_$(TARGET).xo fetcher_C_$(TARGET).xo fetcher_D_$(TARGET).xo scheduler_IPE_$(TARGET).xo
ifeq ($(TASK), STEP)
	XOS += setup_mi_$(TARGET).xo
else ifeq ($(TASK), TX)
	XOS += writer_$(TARGET).xo
else
	$(error TASK must be either STEP or TX: `$(TASK)` is not valid)
endif

compile: $(XOS)

setup_mi_$(TARGET).xo: ./setup_mi.cpp
	v++ $(XOCCFLAGS) --kernel setup_mi -c -o $@ $<

setup_aie_$(TARGET).xo: ./setup_aie.cpp
	v++ $(XOCCFLAGS) --kernel setup_aie -c -o $@ $<

setup_mutualInfo_$(TARGET).xo: ./setup_mutualInfo.cpp
	v++ $(XOCCFLAGS) --kernel setup_mutualInfo -c -o $@ $<

fetcher_A_$(TARGET).xo: ./fetcher_A.cpp
	v++ $(XOCCFLAGS) --kernel fetcher_A -c -o $@ $<

fetcher_B_$(TARGET).xo: ./fetcher_B.cpp
	v++ $(XOCCFLAGS) --kernel fetcher_B -c -o $@ $<

fetcher_C_$(TARGET).xo: ./fetcher_C.cpp
	v++ $(XOCCFLAGS) --kernel fetcher_C -c -o $@ $<

fetcher_D_$(TARGET).xo: ./fetcher_D.cpp
	v++ $(XOCCFLAGS) --kernel fetcher_D -c -o $@ $<

scheduler_IPE_$(TARGET).xo: ./scheduler_IPE.cpp
	v++ $(XOCCFLAGS) --kernel scheduler_IPE -c -o $@ $<

writer_$(TARGET).xo: ./writer.cpp
	v++ $(XOCCFLAGS) --kernel writer -c -o $@ $<

################## testbench

testbench/ultimate/testbench_ultimate.exe: testbench/ultimate/testbench_ultimate.cpp ../sw/include/software_mi/software_mi.cpp testbench/utils.hpp | scheduler_IPE.cpp
	g++ -std=c++17 -I. -I$(XILINX_HLS)/include -o $@ $^ -O2 `pkg-config --libs opencv4` `pkg-config --cflags opencv4`
run_testbench_ultimate: testbench/ultimate/testbench_ultimate.exe
	cd testbench/ultimate && ./testbench_ultimate.exe $(N_COUPLES)
run_testbench_ultimate_t: testbench/ultimate/testbench_ultimate.exe
	cd testbench/ultimate && ./testbench_ultimate.exe $(N_COUPLES) $(N_ROWS) $(N_COLS) $(TX) $(TY) $(ANG)

generate_movers_body: 
	python3 ./generators/movers_generator.py $(INT_PE) $(PIXELS_PER_READ) $(HIST_PE)
	python3 ./generators/setup_mi_generator.py $(INT_PE) 

################## clean up
clean:
	$(RM) -rf *.xo *.xclbin *.xclbin.info *.xclbin.link_summary *.jou *.log *.xo.compile_summary _x .Xil
