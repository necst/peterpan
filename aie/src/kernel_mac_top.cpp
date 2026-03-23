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

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"
#include "kernel_mac_bottom.h"

#define ID 0
using namespace std;

// const uint64_t aie_patterns[INT_PE][2] = AIE_PATTERNS;
// const int aie_pattern_offsets[INT_PE][2] = AIE_PATTERN_OFFSETS;

inline aie::vector<float, 32> fromUint8ToFloat32(aie::vector<uint8, 32> vec) {
    aie::accum<acc32, 32> acc;
    acc.from_vector(vec, 0);
    aie::vector<int32, 32> vec_int32 = acc.to_vector<int32>();
    return aie::to_float(vec_int32, 0);
}

inline aie::vector<uint8, 32> fromFloat32ToUint8(aie::vector<float, 32> vec) {
    // initial vector: [float-pixel-value, float-pixel-value, ...]

    // 1) convert to fixed point (by rounding), getting: [int32-pixel-value, int32-pixel-value, ...]
    aie::vector<int32, 32> int32x32 = aie::to_fixed(vec, 0);
    // 2) cast to uint8, getting: [uint8-pixel-value, 0, 0, 0, uint8-pixel-value, 0, 0, 0, ...]
    aie::vector<uint8, 128> uint8x128 = int32x32.cast_to<uint8>();
    // 3) select even pixels, getting: [uint8-pixel-value, 0, uint8-pixel-value, 0 ...]
    aie::vector<uint8, 64> uint8x64 = aie::filter_even(uint8x128, 1);
    // 4) select again even pixels, getting: [uint8-pixel-value, uint8-pixel-value, ...]
    aie::vector<uint8, 32> uint8x32 = aie::filter_even(uint8x64, 1);

    return uint8x32;
}

constexpr int J_COEFF_INDEX = 2 + (1 - ID);

void
#if ID == 0
mac_top
#else
mac_bottom
#endif
(input_window_uint8* restrict pixels_in, input_window_float* restrict coeff_in, output_stream<uint8>* restrict float_interpolated) {

    // --- reading n_couples from coeff_in ---
    window_acquire(coeff_in);
    aie::vector<float, 32> c_tmp1 = window_readincr_v32(coeff_in);
    aie::vector<float, 32> c_tmp2 = window_readincr_v32(coeff_in);
    window_release(coeff_in);
    const int n_couples = aie::to_fixed(c_tmp1.get(3), 0);
    const int n_col = aie::to_fixed(c_tmp1.get(5), 0);  // TODO remove
    const int size = aie::to_fixed(c_tmp1.get(11), 0);
    const int loop_size_final = aie::to_fixed(c_tmp1.get(12), 0);

    // --- reading IPE_ID from pixels_in ---
    window_acquire(pixels_in);
    aie::vector<uint8, 32> p_tmp1 = window_readincr_v32(pixels_in);
    aie::vector<uint8, 32> p_tmp2 = window_readincr_v32(pixels_in);
    aie::vector<uint8, 32> p_tmp3 = window_readincr_v32(pixels_in);
    aie::vector<uint8, 32> p_tmp4 = window_readincr_v32(pixels_in);
    window_release(pixels_in);
    const int IPE_ID = p_tmp1.get(0);

    // --- sending n_couples to mergers (only if INT_PE > 64) ---
    // #if INT_PE > MAX_INT_PE_PLIOS / 2 && ID == 0
    if (IPE_ID % 2 == 0) {
        printf("[%d] sending loop_size=%d\n", IPE_ID, loop_size_final);
        writeincr(float_interpolated, (uint8)(loop_size_final & 0xFF));
        writeincr(float_interpolated, (uint8)((loop_size_final >> 8) & 0xFF));
        writeincr(float_interpolated, (uint8)((loop_size_final >> 16) & 0xFF));
        writeincr(float_interpolated, (uint8)((loop_size_final >> 24) & 0xFF));
    }
    // #endif

    // --- main interpolation loop ---

    const int depth = n_couples / 32;
    int local_chunk_index = IPE_ID;

    for (int i = 0; i < size; i++) chess_prepare_for_pipelining {
            const int coeff_index = i;

            if (coeff_index % 16 == 0) window_acquire(coeff_in);
            const aie::vector<float, 4> current_coeff =
                window_readincr_v4(coeff_in);  // read all 4 coefficients at once
            if (coeff_index % 16 == 15) window_release(coeff_in);

            // TODO in theory, all discards could be computed in a vector in the `if` above (at the beginning of
            // depth)

            // computing the range of chunks to which to apply the current coefficients
            // if the current chunk index is outside the range, we will discard the chunk
            const int chunk_start = coeff_index * depth;
            const int chunk_end = chunk_start + depth - 1;
            const int internal_loop_size =
                ((local_chunk_index < chunk_start || local_chunk_index > chunk_end)
                     ? 0
                     : 1 + ((chunk_end - local_chunk_index) / INT_PE));

            for (int j = 0; j < internal_loop_size; j += 1) chess_loop_range(0, ) chess_prepare_for_pipelining {
                    local_chunk_index += INT_PE;

                    aie::vector<float, 32> Ri = aie::broadcast<float, 32>(current_coeff.get(0));
                    aie::vector<float, 32> Ri_inv = aie::broadcast<float, 32>(current_coeff.get(1));
                    aie::vector<float, 32> Rj = aie::broadcast<float, 32>(current_coeff.get(3));
                    aie::vector<float, 32> Rj_inv = aie::broadcast<float, 32>(current_coeff.get(2));

                    window_acquire(pixels_in);
                    aie::vector<uint8, 32> pixel_top_left_uint8 = window_readincr_v32(pixels_in);
                    aie::vector<uint8, 32> pixel_top_right_uint8 = window_readincr_v32(pixels_in);
                    aie::vector<uint8, 32> pixel_bottom_left_uint8 = window_readincr_v32(pixels_in);
                    aie::vector<uint8, 32> pixel_bottom_right_uint8 = window_readincr_v32(pixels_in);
                    window_release(pixels_in);

                    aie::vector<float, 32> pixel_top_left = fromUint8ToFloat32(pixel_top_left_uint8);
                    aie::vector<float, 32> pixel_top_right = fromUint8ToFloat32(pixel_top_right_uint8);
                    aie::vector<float, 32> pixel_bottom_left = fromUint8ToFloat32(pixel_bottom_left_uint8);
                    aie::vector<float, 32> pixel_bottom_right = fromUint8ToFloat32(pixel_bottom_right_uint8);

                    auto prod_ab = aie::mac(aie::mul(pixel_top_left, Ri_inv), pixel_top_right, Ri);
                    aie::vector<float, 32> partial_ab = aie::mul(prod_ab.to_vector<float>(), Rj);

                    auto prod_cd = aie::mac(aie::mul(pixel_bottom_left, Ri_inv), pixel_bottom_right, Ri);
                    aie::vector<float, 32> partial_cd = aie::mul(prod_cd.to_vector<float>(), Rj_inv);

                    aie::vector<float, 32> pixel_interpolated_float = aie::add(partial_ab, partial_cd);

                    writeincr(float_interpolated, fromFloat32ToUint8(pixel_interpolated_float));
                }
        }
}
