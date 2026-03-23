/*
MIT License

Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide Conficconi, Eleonora D'Arnese

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#include "../common/common.h"
#include "../mutual_info/include/hw/mutualInfo/mutual_info.hpp"
#include "hls_burst_maxi.h"
#include "writer_body.hpp"

constexpr int WR_TRIP_VARIABLE_DIM = DIMENSION;
constexpr int WR_TRIP_VARIABLE_AREA = DIMENSION * DIMENSION;

constexpr int WR_TRIP_VARIABLE_LAYERS_MIN = 1;
constexpr int WR_TRIP_VARIABLE_LAYERS_MAX = 512;
constexpr int WR_TRIP_VARIABLE_LAYERS_AVG = 256;

constexpr int WR_TRIP_VARIABLE_VOLUME_MIN = WR_TRIP_VARIABLE_AREA * WR_TRIP_VARIABLE_LAYERS_MIN;
constexpr int WR_TRIP_VARIABLE_VOLUME_MAX = WR_TRIP_VARIABLE_AREA * WR_TRIP_VARIABLE_LAYERS_MAX;
constexpr int WR_TRIP_VARIABLE_VOLUME_AVG = WR_TRIP_VARIABLE_AREA * WR_TRIP_VARIABLE_LAYERS_AVG;

constexpr int WR_fifo_in_depth_smi = N_COUPLES_MAX * DIMENSION * DIMENSION / NUM_PIXELS_PER_READ;

void writer(ap_uint<INPUT_DATA_BITWIDTH_FETCHER>* pixels_out, int n_couples, int n_row, int n_col, WRITER_SIGNATURE) {
    WRITER_PRAGMAS
#pragma HLS DATAFLOW
#pragma HLS INTERFACE m_axi port = pixels_out depth = WR_fifo_in_depth_smi offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = pixels_out bundle = control
#pragma HLS interface s_axilite port = n_couples bundle = control
#pragma HLS interface s_axilite port = n_row bundle = control
#pragma HLS interface s_axilite port = n_col bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    const int VOLUME = n_row * n_col * (n_couples >> NUM_PIXELS_PER_READ_EXPO) >> DIV_EXPO;

    ap_uint<INPUT_DATA_BITWIDTH_FETCHER> pixels;

    for (int i = 0; i < VOLUME; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = WR_TRIP_VARIABLE_VOLUME_MIN max = WR_TRIP_VARIABLE_VOLUME_MAX avg = \
    WR_TRIP_VARIABLE_VOLUME_AVG

        WRITER_BODY
    }
}
