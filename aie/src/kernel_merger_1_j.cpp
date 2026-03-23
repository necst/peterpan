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

#include "kernel_merger_1_j.h"

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"
#include "common.h"

// #define INCR 32
// #define SIZE DIMENSION *DIMENSION / INCR

void merger_1_j(input_window_float* in0, input_window_float* in1, output_window_float* R_out) {
    window_acquire(in0);
    aie::vector<float, 32> tmp = window_readincr_v32(in0);
    window_release(in0);

    const float n_row_float = tmp.get(4);
    const int n_row = aie::to_fixed(n_row_float, 0);
    const int dimension_div_32_col = aie::to_fixed(tmp.get(9), 0);
    const int size = n_row * dimension_div_32_col;

    aie::vector<float, 32> vector_0;
    aie::vector<float, 32> vector_1;

    for (int i = 0; i < size; i += 1) chess_loop_range(32, ) chess_prepare_for_pipelining {
            window_acquire(in0);
            vector_0 = window_readincr_v32(in0);
            window_release(in0);
            window_acquire(in1);
            vector_1 = window_readincr_v32(in1);
            window_release(in1);

            std::pair<aie::vector<float, 32>, aie::vector<float, 32>> vec_interleaved =
                aie::interleave_zip(vector_0, vector_1, 1);

            chess_separator_scheduler();
            window_acquire(R_out);
            window_writeincr(R_out, vec_interleaved.first);
            window_writeincr(R_out, vec_interleaved.second);
            window_release(R_out);
        }
}
