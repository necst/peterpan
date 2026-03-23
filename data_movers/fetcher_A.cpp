/*
MIT License

Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide
Conficconi, Eleonora D'Arnese

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
#include "read_volume.hpp"

void fetcher_A(
    hls::stream<ap_axis<COORD_BITWIDTH, 0, 0, 0>>& coords_in,
    hls::stream<ap_uint<INPUT_DATA_BITWIDTH_FETCHER>>& float_out,
    hls::burst_maxi<ap_uint<INPUT_DATA_BITWIDTH_FETCHER>> float_original,
    int n_couples,
    int n_row,
    int n_col) {
#pragma HLS DATAFLOW

#pragma HLS INTERFACE m_axi port = float_original depth = fifo_in_depth_smi offset = slave bundle = gmemA
#pragma HLS interface axis port = coords_in depth = 262144
#pragma HLS interface axis port = float_out depth = 262144

#pragma HLS INTERFACE s_axilite port = float_original bundle = control
#pragma HLS INTERFACE s_axilite port = n_couples bundle = control
#pragma HLS INTERFACE s_axilite port = n_row bundle = control
#pragma HLS INTERFACE s_axilite port = n_col bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    const int n_couples_per_PE = n_couples >> NUM_PIXELS_PER_READ_EXPO;

    read_original_whole(coords_in, float_original, float_out, n_couples_per_PE, n_row, n_col);
}
