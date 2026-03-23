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

#include "kernel_mac_bottom.h"

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"

#define ID 1
using namespace std;

const uint64_t aie_patterns[INT_PE][2] = AIE_PATTERNS;
const int aie_pattern_offsets[INT_PE][2] = AIE_PATTERN_OFFSETS;

inline aie::vector<float, 32> fromUint8ToFloat32(aie::vector<uint8, 32> vec) {
    aie::accum<acc32, 32> acc;
    acc.from_vector(vec, 0);
    aie::vector<int32, 32> vec_int32 = acc.to_vector<int32>();
    return aie::to_float(vec_int32, 0);
}

constexpr int J_COEFF_INDEX = 2 + (1 - ID);

void
#if ID == 0
mac_top
#else
mac_bottom
#endif
(input_window_uint8* restrict pixels_in, input_window_float* restrict coeff_in, output_window_float* restrict partial_interpol_out) {
    aie::vector<uint8, 32> pixel_left_uint8;
    aie::vector<uint8, 32> pixel_right_uint8;
    aie::vector<uint8, 128> pixel_left_uint8_BIG;
    aie::vector<uint8, 128> pixel_right_uint8_BIG;

    aie::vector<int16, 32> pixel_left_uint16;
    aie::vector<int16, 32> pixel_right_uint16;

    aie::vector<int32, 32> pixel_left_int32;
    aie::vector<int32, 32> pixel_right_int32;

    aie::vector<float, 32> pixel_left;
    aie::vector<float, 32> pixel_right;
    aie::vector<float, 32> pixel_leftright;
    aie::vector<float, 4> Rcoeff;

    aie::vector<uint8, 128> uint8zeros;

    window_acquire(coeff_in);
    aie::vector<float, 32> tmp = window_readincr_v32(coeff_in);
    aie::vector<float, 32> tmp2 = window_readincr_v32(coeff_in);
    window_release(coeff_in);

    const float n_couples_float = tmp.get(3);
    const int n_couples = aie::to_fixed(n_couples_float, 0);
    const float n_col_float = tmp.get(5);
    const int n_col = aie::to_fixed(n_col_float, 0);
    const float size_float = tmp.get(11);
    const int size = aie::to_fixed(size_float, 0);

#if ID == 0
    window_acquire(partial_interpol_out);
    window_writeincr(partial_interpol_out, tmp);
    window_writeincr(partial_interpol_out, tmp);
    window_release(partial_interpol_out);
#endif

    const int N = n_couples >> 5;
    const int N_class = (N - 1) % INT_PE;
    const int MULT = (N - 1) / INT_PE;

    window_acquire(pixels_in);
    pixel_left_uint8 = window_readincr_v32(pixels_in);
    pixel_right_uint8 = window_readincr_v32(pixels_in);
    window_release(pixels_in);

    const int PE = pixel_left_uint8.get(0);
    const int acutal_offset = aie_pattern_offsets[N_class][0] * (PE / aie_pattern_offsets[N_class][1]);
    int actual_pattern[INT_PE];
    for (int i = 0; i < INT_PE; i++) chess_loop_count(INT_PE) chess_flatten_loop {
            int skip_bit = (aie_patterns[N_class][i / 64] >> i) & 1UL;
            actual_pattern[i] = skip_bit + MULT;
        }

    constexpr int INT_PE_MASK = INT_PE - 1;

    for (int i = 0; i < size; i++)
    // chess_loop_range(DIMENSION * DIMENSION, ) chess_prepare_for_pipelining
    {
        if (i % 16 == 0) window_acquire(coeff_in);
        Rcoeff = window_readincr_v4(coeff_in);
        if (i % 16 == 15) window_release(coeff_in);

        aie::vector<float, 32> Ri = aie::broadcast<float, 32>(Rcoeff.get(0));
        aie::vector<float, 32> Ri_inv = aie::broadcast<float, 32>(Rcoeff.get(1));
        aie::vector<float, 32> j_coeff = aie::broadcast<float, 32>(Rcoeff.get(J_COEFF_INDEX));

        int internal_loop_size = actual_pattern[(i + acutal_offset) & INT_PE_MASK];

        for (int j = 0; j < internal_loop_size; j += 1) chess_loop_range(0, ) chess_prepare_for_pipelining {
                window_acquire(pixels_in);
                pixel_left_uint8 = window_readincr_v32(pixels_in);
                pixel_right_uint8 = window_readincr_v32(pixels_in);
                window_release(pixels_in);

                pixel_left = fromUint8ToFloat32(pixel_left_uint8);
                pixel_right = fromUint8ToFloat32(pixel_right_uint8);

                auto tmp = aie::mac(aie::mul(pixel_left, Ri_inv), pixel_right, Ri);
                aie::vector<float, 32> result = aie::mul(tmp.to_vector<float>(), j_coeff);

                window_acquire(partial_interpol_out);
                window_writeincr(partial_interpol_out, result);
                window_release(partial_interpol_out);
            }
    }
}
