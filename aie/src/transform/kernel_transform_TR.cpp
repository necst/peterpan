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

#include "common.h"

//
// ###############  SET KERNEL ROLE HERE  ###############
#define CURR_AIE_ROLE AIE_TOP_RIGHT
// #######################################################
//

#if CURR_AIE_ROLE == AIE_TOP_LEFT
#include "kernel_transform_TL.h"
#elif CURR_AIE_ROLE == AIE_TOP_RIGHT
#include "kernel_transform_TR.h"
#elif CURR_AIE_ROLE == AIE_BOTTOM_LEFT
#include "kernel_transform_BL.h"
#elif CURR_AIE_ROLE == AIE_BOTTOM_RIGHT
#include "kernel_transform_BR.h"
#endif

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"

using namespace std;

#if CURR_AIE_ROLE == AIE_TOP_LEFT
#define FUNC_NAME transform_TL
#elif CURR_AIE_ROLE == AIE_TOP_RIGHT
#define FUNC_NAME transform_TR
#elif CURR_AIE_ROLE == AIE_BOTTOM_LEFT
#define FUNC_NAME transform_BL
#elif CURR_AIE_ROLE == AIE_BOTTOM_RIGHT
#define FUNC_NAME transform_BR
#endif
// #define FLOOR(x) aie::to_fixed((x), 0)

#define FLOOR(x) aie::to_fixed((aie::sub(x, 0.5f)), 0)
#define CEIL(x) aie::add(FLOOR(x), 1)

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
//
// Generates a stream of transformed coordinates using the parameters passed in
// params_in. params_in must be a stream of 4 floats:
//  - tx: translation along the x axis
//  - ty: translation along the y axis
//  - ang: rotation angle in radians
//  - n_couples: number of slices to be transformed
//
void FUNC_NAME(
    input_window_float* restrict coords_in_cc,
    input_window_float* restrict coords_in_rr,
    output_window_int32* restrict coords_out) {
    // configuration of the volume transform
    window_acquire(coords_in_cc);
    aie::vector<float, 32> tmp = window_readincr_v32(coords_in_cc);
    window_release(coords_in_cc);

    const float tx = tmp.get(0);
    const float ty = tmp.get(1);
    const float ang = tmp.get(2);
    const float n_couples_float = tmp.get(3);
    const int n_couples = aie::to_fixed(n_couples_float, 0);
    const float n_row_float = tmp.get(4);
    const int n_row = aie::to_fixed(n_row_float, 0);
    const float n_col_float = tmp.get(5);
    const int n_col = aie::to_fixed(n_col_float, 0);
    const float half_dimension_row = tmp.get(6);
    const float half_dimension_col = tmp.get(7);
    // const int dimension_div_32_row = aie::to_fixed(tmp.get(8), 0);
    const int dimension_div_32_col = aie::to_fixed(tmp.get(9), 0);
    const float neg_half_dimension_col = -half_dimension_col;

    // printf("[!]AIE TRA --> tx: %f, ty: %f, ang: %f, n_couples: %d\n", tx, ty,
    // ang, n_couples);

    // constant values used during transformation phase
    // const aie::vector<float, 32> tx_v = aie::broadcast<float, 32>(tx);
    // const aie::vector<float, 32> ty_v = aie::broadcast<float, 32>(ty);
    // const aie::vector<float, 32> to_center = aie::broadcast<float,
    // 32>(HALF_DIMENSION); const aie::vector<float, 32> from_center_tx =
    // aie::sub(to_center, tx_v); const aie::vector<float, 32> from_center_ty =
    // aie::sub(to_center, ty_v); const float p_cos = cos(ang); const float p_sin
    // = sin(ang); const float n_sin = -p_sin;

    // constant values used to compute buffer offset from the (c,r) coordinates
    const aie::vector<int32, 32> n_couples_v = aie::broadcast<int32, 32>(n_couples);
    const aie::vector<int32, 32> size_n_couples_v = aie::broadcast<int32, 32>(n_col * n_couples);

    // counters for the columns
    aie::vector<float, 32> v_offsets = aie::load_v<32>(offsets_arr);
    aie::vector<float, 32> start_point = aie::broadcast<float, 32>(neg_half_dimension_col);
    aie::vector<float, 32> init_cols = aie::add(v_offsets, start_point);
    aie::vector<float, 32> cc_interval = aie::broadcast<float, 32>(32.f);

    int k_index = 0;

    // #if CURR_AIE_ROLE == AIE_TOP_LEFT
    // writeincr(coeff_out, n_couples_float);
    // #endif

    aie::set_rounding(aie::rounding_mode::floor);

loop_rows:
    for (int r = -half_dimension_row + k_index * n_row; r < -half_dimension_row + (k_index + 1) * n_row; r++)
        chess_loop_range(32, ) chess_prepare_for_pipelining {
            aie::vector<float, 32> rr = aie::broadcast<float, 32>(r);
            aie::vector<float, 32> cc = init_cols;

        loop_columns:
            for (int c = 0; c < dimension_div_32_col; c++) chess_loop_range(32, ) chess_prepare_for_pipelining {
                    // 1) translate to (-SIZE/2, -SIZE/2)
                    // aie::vector<float, 32> cc_center = aie::sub(cc, to_center);
                    // aie::vector<float, 32> rr_center = aie::sub(rr, to_center);

                    // 2) rotate of ang radians around the center
                    // aie::vector<float, 32> cc_p_cos = aie::mul(cc, p_cos);
                    // aie::vector<float, 32> cc_p_sin = aie::mul(cc, p_sin);
                    // aie::vector<float, 32> rr_n_sin = aie::mul(rr, n_sin);
                    // aie::vector<float, 32> rr_p_cos = aie::mul(rr, p_cos);
                    // aie::vector<float, 32> cc_center_tra = aie::add(cc_p_cos,
                    // rr_n_sin); // c' = c * cos(ang) - r * sin(ang) aie::vector<float,
                    // 32> rr_center_tra = aie::add(cc_p_sin, rr_p_cos); // r' = c *
                    // sin(ang) + r * cos(ang)

                    // // 3) translate to (-SIZE/2, -SIZE/2) and then to (tx, ty)
                    // aie::vector<float, 32> cc_tra = aie::add(cc_center_tra,
                    // from_center_tx); aie::vector<float, 32> rr_tra =
                    // aie::add(rr_center_tra, from_center_ty);

                    window_acquire(coords_in_cc);
                    aie::vector<float, 32> cc_tra = window_readincr_v32(coords_in_cc);
                    window_release(coords_in_cc);

                    window_acquire(coords_in_rr);
                    aie::vector<float, 32> rr_tra = window_readincr_v32(coords_in_rr);
                    window_release(coords_in_rr);
                    chess_separator_scheduler();

#if CURR_AIE_ROLE == AIE_TOP_LEFT
                    aie::vector<int32, 32> cc_rounded = FLOOR(cc_tra);  // cc_rounded = floor(cc_tra)
                    aie::vector<int32, 32> rr_rounded = FLOOR(rr_tra);  // rr_rounded = floor(rr_tra)
                                                                        // aie::vector<float, 32> R = aie::sub(rr_tra,
            // aie::to_float(rr_rounded)); // R = rr_tra - rr_rounded(floor) [Rj]

            // aie::vector<float, 32> R = aie::broadcast<float, 32>(1);

#elif CURR_AIE_ROLE == AIE_TOP_RIGHT
                    aie::vector<int32, 32> cc_rounded = CEIL(cc_tra);   // cc_rounded = ceil(cc_tra)
                    aie::vector<int32, 32> rr_rounded = FLOOR(rr_tra);  // rr_rounded = floor(rr_tra)
            // aie::vector<float, 32> R = aie::sub(aie::to_float(cc_rounded),
            // cc_tra); // R = cc_rounded(ceil) - cc_tra [Ri_inv]

#elif CURR_AIE_ROLE == AIE_BOTTOM_LEFT
                    aie::vector<int32, 32> cc_rounded = FLOOR(cc_tra);  // cc_rounded = floor(cc_tra)
                    aie::vector<int32, 32> rr_rounded = CEIL(rr_tra);   // rr_rounded = ceil(rr_tra)
                                                                        // aie::vector<float, 32> R = aie::sub(cc_tra,
                                                                        // aie::to_float(cc_rounded)); // R = rr_tra -
                                                                        // rr_rounded(floor) [Ri]

#elif CURR_AIE_ROLE == AIE_BOTTOM_RIGHT
                    aie::vector<int32, 32> cc_rounded = CEIL(cc_tra);  // cc_rounded = ceil(cc_tra)
                    aie::vector<int32, 32> rr_rounded = CEIL(rr_tra);  // rr_rounded = ceil(rr_tra)
// aie::vector<float, 32> R = aie::sub(aie::to_float(rr_rounded), rr_tra); // R
// = rr_rounded(ceil) - rr_tra [Rj_inv]
#endif

                    // // 4) floor TODO check if another rounding method is needed
                    // cc_rounded = aie::to_fixed(cc_tra, 0);
                    // rr_rounded = aie::to_fixed(rr_tra, 0);

                    // 5) compute index: I = (r' * n_col * n_couples) + (c' * n_couples)
                    auto cc_rounded_n_couples = aie::mul(cc_rounded, n_couples_v);  // cc_rounded * n_couples
                    auto rr_rounded_size_n_couples =
                        aie::mul(rr_rounded, size_n_couples_v);  // rr_rounded * n_col * n_couples

                    aie::vector<int32, 32> indexes = aie::add(
                        cc_rounded_n_couples.to_vector<int32>(0),
                        rr_rounded_size_n_couples.to_vector<int32>(
                            0));  // indexes[i] = rr_rounded[i] * n_col * n_couples
                                  // + cc_rounded[i] * n_couples
                    chess_separator_scheduler();

                    // 6) check if the index is out of boundary
                    auto rr_mask_lt =
                        aie::lt(rr_rounded, aie::broadcast<int32, 32>(0));  // rr_mask_lt[i] = rr_rounded[i] < 0
                    auto rr_mask_ge = aie::ge(
                        rr_rounded, aie::broadcast<int32, 32>(n_row));  // rr_mask_ge[i] = rr_rounded[i] >= n_row
                    auto cc_mask_lt =
                        aie::lt(cc_rounded, aie::broadcast<int32, 32>(0));  // cc_mask_lt[i] = cc_rounded[i] < 0
                    auto cc_mask_ge = aie::ge(
                        cc_rounded, aie::broadcast<int32, 32>(n_col));  // cc_mask_ge[i] = cc_rounded[i] >= n_col
                    auto mask_invalid = rr_mask_ge | cc_mask_lt | rr_mask_lt | cc_mask_ge;
                    chess_separator_scheduler();

                    // 7) set to -1 the indexes that are out of boundary
                    aie::vector<int32, 32> validated = aie::select(
                        indexes,
                        aie::broadcast<int32, 32>(-1),
                        mask_invalid);  // validated[i] = (mask_invalid[i] ==
                                        // 0 ? indexes[i] : -1;

                    // 8) write
                    window_acquire(coords_out);
                    window_writeincr(coords_out, validated);
                    window_release(coords_out);
                    // writeincr(coeff_out, R);

                    // 9) increase row indexes
                    cc = aie::add(cc, cc_interval);
                }
        }
}
