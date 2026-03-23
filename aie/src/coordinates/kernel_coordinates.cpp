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

#include "kernel_coordinates.h"

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"
#include "common.h"

using namespace std;

// INIT COLS was used before to move computation across columns. Now it is
// computed at runtime as offset[0 - 31] broadcast<32>(-n_col/2) add(offset,
// broadcast) alignas(32) const float init_col_const[32] = INIT_COLS;
//  constexpr float HALF_DIMENSION = DIMENSION / 2.f;
//  constexpr int DIMENSION_DIV_32 = DIMENSION / 32;
// clang-format off
alignas(32) const float offsets_arr[32] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
// clang-format on

void coordinates(
    input_window_float* restrict params_in,
    output_window_float* restrict coords_out_cc,
    output_window_float* restrict coords_out_rr) {
    // configuration of the volume transform
    aie::vector<float, 16> tmp = window_readincr_v16(params_in);

    const float tx = tmp.get(0);
    const float ty = tmp.get(1);
    const float ang = tmp.get(2);
    const int n_couples = aie::to_fixed(tmp.get(3), 0);
    const int n_row = aie::to_fixed(tmp.get(4), 0);
    const int n_col = aie::to_fixed(tmp.get(5), 0);
    const float half_dimension_row = tmp.get(6);
    const float half_dimension_col = tmp.get(7);
    const float neg_half_dimension_col = -half_dimension_col;
    // const int dimension_div_32_row = aie::to_fixed(tmp.get(8), 0); // unused
    const int dimension_div_32_col = aie::to_fixed(tmp.get(9), 0);

    window_acquire(coords_out_cc);
    window_writeincr(coords_out_cc, tmp);
    window_writeincr(coords_out_cc, tmp);
    window_release(coords_out_cc);
    chess_separator_scheduler();

    const float p_cos = cos(ang);
    const float p_sin = sin(ang);
    const float n_sin = -p_sin;
    chess_separator_scheduler();

    // counters for the columns
    aie::vector<float, 32> v_offsets = aie::load_v<32>(offsets_arr);
    aie::vector<float, 32> start_point = aie::broadcast<float, 32>(neg_half_dimension_col);
    aie::vector<float, 32> init_cols = aie::add(v_offsets, start_point);

    aie::vector<float, 32> cc_interval = aie::broadcast<float, 32>(32.f);

    const float p_cos_32 = p_cos * 32;
    const float p_sin_32 = p_sin * 32;

    aie::vector<float, 32> init_cols_times_p_cos = aie::mul(aie::sub(init_cols, tx), p_cos);
    aie::vector<float, 32> cc_first_tra =
        aie::add((half_dimension_row + ty) * p_sin + half_dimension_col - p_cos_32, init_cols_times_p_cos);

    aie::vector<float, 32> init_rows_times_p_sin = aie::mul(aie::sub(init_cols, tx), p_sin);
    aie::vector<float, 32> rr_first_tra =
        aie::add((-half_dimension_row - ty) * p_cos + half_dimension_row - p_sin_32, init_rows_times_p_sin);

    aie::vector<float, 32> cc_tra = aie::broadcast<float, 32>(0.f);
    aie::vector<float, 32> rr_tra = aie::broadcast<float, 32>(0.f);

loop_rows:
    for (int r = -half_dimension_row; r < -half_dimension_row + n_row; r++)

        chess_loop_range(32, ) chess_prepare_for_pipelining {
            cc_tra = cc_first_tra;
            cc_first_tra = aie::add(cc_first_tra, n_sin);

            rr_tra = rr_first_tra;
            rr_first_tra = aie::add(rr_first_tra, p_cos);
            chess_separator_scheduler();

        loop_columns:
            for (int c = 0; c < dimension_div_32_col; c++) chess_loop_range(32, ) chess_prepare_for_pipelining {
                    cc_tra = aie::add(cc_tra, p_cos_32);
                    rr_tra = aie::add(rr_tra, p_sin_32);
                    chess_separator_scheduler();

                    // 4) write output
                    window_acquire(coords_out_cc);
                    window_writeincr(coords_out_cc, cc_tra);
                    window_release(coords_out_cc);

                    window_acquire(coords_out_rr);
                    window_writeincr(coords_out_rr, rr_tra);
                    window_release(coords_out_rr);
                }
        }
}