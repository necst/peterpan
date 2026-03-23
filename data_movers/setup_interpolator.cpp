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
#include "setup_interpolator_body.hpp"

constexpr int CHUNK_SIZE = 32;
constexpr int TRIP_VARIABLE_AREA_SI = DIMENSION * DIMENSION * N_COUPLES_MAX;
constexpr int TRIP_VARIABLE_AREA_CHUNKED_SI =
    DIMENSION * DIMENSION * N_COUPLES_MAX / (NUM_PIXELS_PER_READ / CHUNK_SIZE);

void setup_interpolator(
    hls::stream<ap_uint<INPUT_DATA_BITWIDTH_FETCHER>>& Left_in,
    hls::stream<ap_uint<INPUT_DATA_BITWIDTH_FETCHER>>& Right_in,
    SETUP_INTERPOLATOR_SIGNATURE,
    int n_couples,
    int n_row,
    int n_col) {
    // #pragma HLS DATAFLOW

#pragma HLS INTERFACE s_axilite port = n_couples bundle = control
#pragma HLS INTERFACE s_axilite port = n_row bundle = control
#pragma HLS INTERFACE s_axilite port = n_col bundle = control
#pragma HLS interface s_axilite port = return bundle = control

#pragma HLS interface axis depth = 262144 port = Left_in
#pragma HLS interface axis depth = 262144 port = Right_in
    SETUP_INTERPOLATOR_PRAGMAS

    static_assert(NUM_PIXELS_PER_READ % 32 == 0, "NUM_PIXELS_PER_READ must be multiple of 32");

    // Assigning the IDs to the interpolator PEs
    ap_uint<INPUT_DATA_BITWIDTH_FETCHER_MIN> data;

    // sends to each output stream the id of the interpolator
    SETUP_INTERPOLATOR_AIE_STARTER

    ap_uint<INPUT_DATA_BITWIDTH_FETCHER> A;
    ap_uint<INPUT_DATA_BITWIDTH_FETCHER> B;

    const int VOLUME = n_row * n_col * (n_couples >> NUM_PIXELS_PER_READ_EXPO) >> DIV_EXPO;

    for (int i = 0; i < VOLUME; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = TRIP_VARIABLE_AREA_CHUNKED_SI
#pragma HLS PIPELINE

        // automatically generated code for scheduling the data to the interpolator
        // PEs
        SETUP_INTERPOLATOR_BODY
    }
}
