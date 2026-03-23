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
#include <hls_stream.h>

#include "../common/constants.h"

extern "C" {

void setup_aie(float tx, float ty, float ang, int n_couples, int n_row, int n_col, hls::stream<float>& s) {
#pragma HLS interface axis port = s
#pragma HLS interface s_axilite port = tx bundle = control
#pragma HLS interface s_axilite port = ty bundle = control
#pragma HLS interface s_axilite port = ang bundle = control
#pragma HLS interface s_axilite port = n_couples bundle = control
#pragma HLS interface s_axilite port = n_row bundle = control
#pragma HLS interface s_axilite port = n_col bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    s.write(tx);                // 0
    s.write(ty);                // 1
    s.write(ang);               // 2
    s.write((float)n_couples);  // 3
    s.write((float)n_row);      // 4
    s.write((float)n_col);      // 5

    int size = n_row * n_col;
    s.write((float)(n_row >> 1));  // 6 - HALF_DIM_ROW
    s.write((float)(n_col >> 1));  // 7 - HALF_DIM_COL

    s.write(0.0f);                 // 8 - ROW_DIV_32 (unused)
    s.write((float)(n_col >> 5));  // 9 - COL_DIV_32
    s.write((float)(n_col >> 4));  // 10 - COL_DIV_16

    s.write((float)size);  // 11 - TOTAL SIZE

    int loop_int = (n_couples * size) >> (INT_PE_EXPO + 5);
    s.write((float)loop_int);  // 12 - loop_size_final

    s.write(0);
    s.write(0);
    s.write(0);
}
}
