/******************************************
 *MIT License
 *
 *Copyright (c) [2022] [Eleonora D'Arnese, Davide Conficconi, Emanuele Del
 *Sozzo, Luigi Fusco, Donatella Sciuto, Marco Domenico Santambrogio]
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in all
 *copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *SOFTWARE.
 */
/***************************************************************
 *
 * High-Level-Synthesis header file for Mutual Information computation
 *
 ****************************************************************/
#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "hls_stream.h"
#include "mutual_info.hpp"

#if ENTROPY_PE == 1
#define SLICE_BITWIDTH 0
#endif

#if ENTROPY_PE == 2
#define SLICE_BITWIDTH 1
#endif

#if ENTROPY_PE == 4
#define SLICE_BITWIDTH 2
#endif

#if ENTROPY_PE == 8
#define SLICE_BITWIDTH 3
#endif

#if ENTROPY_PE == 16
#define SLICE_BITWIDTH 4
#endif

#if ENTROPY_PE == 32
#define SLICE_BITWIDTH 5
#endif

// The hist_id parameter is necessary to make HLS instantiate different
// functions, each one with a different j_h array Otherwise, each instance would
// use the same j_h array (because it is static)
template <typename Tin, unsigned int hist_id, typename Thist, typename Tout, unsigned int bitsThist>
void joint_histogram(hls::stream<Tin> &index_stream, hls::stream<Tout> &j_h_stream, unsigned int dim) {
    static Thist j_h[ENTROPY_PE][J_HISTO_ROWS * J_HISTO_COLS / ENTROPY_PE] = {0};
#pragma HLS ARRAY_PARTITION variable = j_h complete dim = 1

    static ap_uint<HIST_INDEX_DATA_BITWIDTH - SLICE_BITWIDTH> prev_index[ENTROPY_PE] = {0};
#pragma HLS ARRAY_PARTITION variable = prev_index complete dim = 1

    static Thist acc[ENTROPY_PE] = {0};

// GIUSEPPE: Upon usage URAM -> TOO MUCH LUT -> TRYING TO REINTRODUCE SOME BRAM INSTEAD OF LUTS
#pragma HLS bind_storage variable = acc type = RAM_2P impl = bram
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1

    // GIUSEPPE: URAM for histograms - 2P = dual port
    // Every element is read and written multiple times, a single port URAM does not respect II=1.

#pragma HLS bind_storage variable = j_h type = RAM_2P impl = uram

#pragma HLS DEPENDENCE variable = j_h intra RAW false
HIST:
    for (unsigned int i = 0; i < dim;) {
#pragma HLS PIPELINE II = 1
        HIST_INDEX_DATA_TYPE input_index;
        bool ready = index_stream.read_nb(input_index);

#if ENTROPY_PE > 1
        ap_uint<SLICE_BITWIDTH> slice_idx = input_index.range(SLICE_BITWIDTH - 1, 0);
#endif
        ap_uint<HIST_INDEX_DATA_BITWIDTH - SLICE_BITWIDTH> curr_index =
            input_index.range(HIST_INDEX_DATA_BITWIDTH - 1, SLICE_BITWIDTH);

// This preprocessor directives are necessary because, if we used the commented
// code below, which covers all cases, Vivado HLS cannot unexpectedly achieve
// II=1 when ENTROPY_PE==2 (maybe a wrong automatic loop unrolling?) Maybe the
// commented code works properly with a different version of Vivado/Vitis HLS
#if ENTROPY_PE == 1
        if (ready) {
            bool check = curr_index == prev_index[0];
            j_h[0][prev_index[0]] = acc[0];
            Thist new_val = j_h[0][curr_index];
            acc[0] = check ? acc[0] + 1 : new_val + 1;
            prev_index[0] = curr_index;
        }
#endif

#if ENTROPY_PE == 2
        if (ready && slice_idx == 0) {
            bool check = curr_index == prev_index[0];
            j_h[0][prev_index[0]] = acc[0];
            Thist new_val = j_h[0][curr_index];
            acc[0] = check ? acc[0] + 1 : new_val + 1;
            prev_index[0] = curr_index;
        }
        if (ready && slice_idx == 1) {
            bool check = curr_index == prev_index[1];
            j_h[1][prev_index[1]] = acc[1];
            Thist new_val = j_h[1][curr_index];
            acc[1] = check ? acc[1] + 1 : new_val + 1;
            prev_index[1] = curr_index;
        }
#endif

#if ENTROPY_PE > 2
        for (unsigned int s = 0; s < ENTROPY_PE; s++) {
            if (ready && s == slice_idx) {
                bool check = curr_index == prev_index[s];
                j_h[s][prev_index[s]] = acc[s];
                Thist new_val = j_h[s][curr_index];
                acc[s] = check ? acc[s] + 1 : new_val + 1;
                prev_index[s] = curr_index;
            }
        }
#endif
        i += ready;
    }

    for (unsigned int s = 0; s < ENTROPY_PE; s++) {
#pragma HLS UNROLL
        j_h[s][prev_index[s]] = acc[s];
        prev_index[s] = 0;
        acc[s] = 0;
    }

WRITE_OUT:
    for (unsigned int i = 0; i < J_HISTO_ROWS * J_HISTO_COLS / ENTROPY_PE; i++) {
#pragma HLS PIPELINE
        Tout val = 0;
        for (unsigned int s = 0; s < ENTROPY_PE; s++) {
            val.range((s + 1) * bitsThist - 1, s * bitsThist) = j_h[s][i];
            j_h[s][i] = 0;
        }
        j_h_stream.write(val);
    }
}

template <
    typename Tin,
    unsigned int dim,
    typename Tout,
    unsigned int STREAM,
    typename TtmpIn,
    unsigned int bitsTtmpIn,
    typename TtmpOut,
    unsigned int bitsTtmpOut>
void sum_joint_histogram(
    hls::stream<Tin> in_stream[STREAM], hls::stream<Tout> &j_h_stream, unsigned int padding_pixels) {
    static TtmpOut tmp[ENTROPY_PE];
#pragma HLS ARRAY_PARTITION variable = tmp complete dim = 1

// GIUSEPPE: URAM for TMP - 2P = dual Port
// Every element is read and written multiple times, a single port URAM does not respect II=1.
#pragma HLS bind_storage variable = tmp type = RAM_2P impl = uram

SUM_JOINT_HIST_LOOP_0:
    for (int i = 0; i < dim; i++) {
#pragma HLS PIPELINE
        Tout out = 0;
    SUM_JOINT_HIST_LOOP_1:
        for (int j = 0; j < STREAM; j++) {
            Tin elem = in_stream[j].read();
            for (int k = 0; k < ENTROPY_PE; k++) {
                TtmpIn unpacked = elem.range((k + 1) * bitsTtmpIn - 1, k * bitsTtmpIn);
                tmp[k] += unpacked;
            }
        }
        if (i == 0) tmp[0] = tmp[0] - padding_pixels;
    SUM_JOINT_HIST_LOOP_2:
        for (int k = 0; k < ENTROPY_PE; k++) {
            out.range((k + 1) * bitsTtmpOut - 1, k * bitsTtmpOut) = tmp[k];
            tmp[k] = 0;
        }
        j_h_stream.write(out);
    }
}

template <
    typename Tin,
    unsigned int dim,
    typename Tout,
    typename TtmpIn,
    unsigned int bitsTtmpIn,
    typename TtmpOut,
    unsigned int bitsTtmpOut>
void convert(hls::stream<Tin> &in_stream, hls::stream<Tout> &out_stream) {
CONVERT_LOOP_0:
    for (int i = 0; i < dim; i++) {
#pragma HLS PIPELINE
        Tin in = in_stream.read();
        Tout out = 0;
    CONVERT_LOOP_1:
        for (int k = 0; k < ENTROPY_PE; k++) {
            TtmpIn unpackedIn = in.range((k + 1) * bitsTtmpIn - 1, k * bitsTtmpIn);
            unsigned int tmp = unpackedIn;
            out.range((k + 1) * bitsTtmpOut - 1, k * bitsTtmpOut) = tmp;
        }
        out_stream.write(out);
    }
}

template <typename Tin, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_1(hls::stream<Tin> index_stream[1], hls::stream<Tout> j_h_pe_stream[1], unsigned int dim) {
#pragma HLS INLINE

    joint_histogram<Tin, 0, Thist, Tout, bitsThist>(index_stream[0], j_h_pe_stream[0], dim);
}

template <typename Tin, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_2(hls::stream<Tin> index_stream[2], hls::stream<Tout> j_h_pe_stream[2], unsigned int dim) {
#pragma HLS INLINE

    joint_histogram<Tin, 0, Thist, Tout, bitsThist>(index_stream[0], j_h_pe_stream[0], dim);
    joint_histogram<Tin, 1, Thist, Tout, bitsThist>(index_stream[1], j_h_pe_stream[1], dim);
}

template <typename Tin, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_4(hls::stream<Tin> index_stream[4], hls::stream<Tout> j_h_pe_stream[4], unsigned int dim) {
#pragma HLS INLINE

    joint_histogram<Tin, 0, Thist, Tout, bitsThist>(index_stream[0], j_h_pe_stream[0], dim);
    joint_histogram<Tin, 1, Thist, Tout, bitsThist>(index_stream[1], j_h_pe_stream[1], dim);
    joint_histogram<Tin, 2, Thist, Tout, bitsThist>(index_stream[2], j_h_pe_stream[2], dim);
    joint_histogram<Tin, 3, Thist, Tout, bitsThist>(index_stream[3], j_h_pe_stream[3], dim);
}

template <typename Tin, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_8(hls::stream<Tin> index_stream[8], hls::stream<Tout> j_h_pe_stream[8], unsigned int dim) {
#pragma HLS INLINE

    joint_histogram<Tin, 0, Thist, Tout, bitsThist>(index_stream[0], j_h_pe_stream[0], dim);
    joint_histogram<Tin, 1, Thist, Tout, bitsThist>(index_stream[1], j_h_pe_stream[1], dim);
    joint_histogram<Tin, 2, Thist, Tout, bitsThist>(index_stream[2], j_h_pe_stream[2], dim);
    joint_histogram<Tin, 3, Thist, Tout, bitsThist>(index_stream[3], j_h_pe_stream[3], dim);
    joint_histogram<Tin, 4, Thist, Tout, bitsThist>(index_stream[4], j_h_pe_stream[4], dim);
    joint_histogram<Tin, 5, Thist, Tout, bitsThist>(index_stream[5], j_h_pe_stream[5], dim);
    joint_histogram<Tin, 6, Thist, Tout, bitsThist>(index_stream[6], j_h_pe_stream[6], dim);
    joint_histogram<Tin, 7, Thist, Tout, bitsThist>(index_stream[7], j_h_pe_stream[7], dim);
}

template <typename Tin, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_16(
    hls::stream<Tin> index_stream[16], hls::stream<Tout> j_h_pe_stream[16], unsigned int dim) {
#pragma HLS INLINE

    joint_histogram<Tin, 0, Thist, Tout, bitsThist>(index_stream[0], j_h_pe_stream[0], dim);
    joint_histogram<Tin, 1, Thist, Tout, bitsThist>(index_stream[1], j_h_pe_stream[1], dim);
    joint_histogram<Tin, 2, Thist, Tout, bitsThist>(index_stream[2], j_h_pe_stream[2], dim);
    joint_histogram<Tin, 3, Thist, Tout, bitsThist>(index_stream[3], j_h_pe_stream[3], dim);
    joint_histogram<Tin, 4, Thist, Tout, bitsThist>(index_stream[4], j_h_pe_stream[4], dim);
    joint_histogram<Tin, 5, Thist, Tout, bitsThist>(index_stream[5], j_h_pe_stream[5], dim);
    joint_histogram<Tin, 6, Thist, Tout, bitsThist>(index_stream[6], j_h_pe_stream[6], dim);
    joint_histogram<Tin, 7, Thist, Tout, bitsThist>(index_stream[7], j_h_pe_stream[7], dim);
    joint_histogram<Tin, 8, Thist, Tout, bitsThist>(index_stream[8], j_h_pe_stream[8], dim);
    joint_histogram<Tin, 9, Thist, Tout, bitsThist>(index_stream[9], j_h_pe_stream[9], dim);
    joint_histogram<Tin, 10, Thist, Tout, bitsThist>(index_stream[10], j_h_pe_stream[10], dim);
    joint_histogram<Tin, 11, Thist, Tout, bitsThist>(index_stream[11], j_h_pe_stream[11], dim);
    joint_histogram<Tin, 12, Thist, Tout, bitsThist>(index_stream[12], j_h_pe_stream[12], dim);
    joint_histogram<Tin, 13, Thist, Tout, bitsThist>(index_stream[13], j_h_pe_stream[13], dim);
    joint_histogram<Tin, 14, Thist, Tout, bitsThist>(index_stream[14], j_h_pe_stream[14], dim);
    joint_histogram<Tin, 15, Thist, Tout, bitsThist>(index_stream[15], j_h_pe_stream[15], dim);
}

template <typename Tin, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_32(
    hls::stream<Tin> index_stream[32], hls::stream<Tout> j_h_pe_stream[32], unsigned int dim) {
#pragma HLS INLINE

    joint_histogram<Tin, 0, Thist, Tout, bitsThist>(index_stream[0], j_h_pe_stream[0], dim);
    joint_histogram<Tin, 1, Thist, Tout, bitsThist>(index_stream[1], j_h_pe_stream[1], dim);
    joint_histogram<Tin, 2, Thist, Tout, bitsThist>(index_stream[2], j_h_pe_stream[2], dim);
    joint_histogram<Tin, 3, Thist, Tout, bitsThist>(index_stream[3], j_h_pe_stream[3], dim);
    joint_histogram<Tin, 4, Thist, Tout, bitsThist>(index_stream[4], j_h_pe_stream[4], dim);
    joint_histogram<Tin, 5, Thist, Tout, bitsThist>(index_stream[5], j_h_pe_stream[5], dim);
    joint_histogram<Tin, 6, Thist, Tout, bitsThist>(index_stream[6], j_h_pe_stream[6], dim);
    joint_histogram<Tin, 7, Thist, Tout, bitsThist>(index_stream[7], j_h_pe_stream[7], dim);
    joint_histogram<Tin, 8, Thist, Tout, bitsThist>(index_stream[8], j_h_pe_stream[8], dim);
    joint_histogram<Tin, 9, Thist, Tout, bitsThist>(index_stream[9], j_h_pe_stream[9], dim);
    joint_histogram<Tin, 10, Thist, Tout, bitsThist>(index_stream[10], j_h_pe_stream[10], dim);
    joint_histogram<Tin, 11, Thist, Tout, bitsThist>(index_stream[11], j_h_pe_stream[11], dim);
    joint_histogram<Tin, 12, Thist, Tout, bitsThist>(index_stream[12], j_h_pe_stream[12], dim);
    joint_histogram<Tin, 13, Thist, Tout, bitsThist>(index_stream[13], j_h_pe_stream[13], dim);
    joint_histogram<Tin, 14, Thist, Tout, bitsThist>(index_stream[14], j_h_pe_stream[14], dim);
    joint_histogram<Tin, 15, Thist, Tout, bitsThist>(index_stream[15], j_h_pe_stream[15], dim);
    joint_histogram<Tin, 16, Thist, Tout, bitsThist>(index_stream[16], j_h_pe_stream[16], dim);
    joint_histogram<Tin, 17, Thist, Tout, bitsThist>(index_stream[17], j_h_pe_stream[17], dim);
    joint_histogram<Tin, 18, Thist, Tout, bitsThist>(index_stream[18], j_h_pe_stream[18], dim);
    joint_histogram<Tin, 19, Thist, Tout, bitsThist>(index_stream[19], j_h_pe_stream[19], dim);
    joint_histogram<Tin, 20, Thist, Tout, bitsThist>(index_stream[20], j_h_pe_stream[20], dim);
    joint_histogram<Tin, 21, Thist, Tout, bitsThist>(index_stream[21], j_h_pe_stream[21], dim);
    joint_histogram<Tin, 22, Thist, Tout, bitsThist>(index_stream[22], j_h_pe_stream[22], dim);
    joint_histogram<Tin, 23, Thist, Tout, bitsThist>(index_stream[23], j_h_pe_stream[23], dim);
    joint_histogram<Tin, 24, Thist, Tout, bitsThist>(index_stream[24], j_h_pe_stream[24], dim);
    joint_histogram<Tin, 25, Thist, Tout, bitsThist>(index_stream[25], j_h_pe_stream[25], dim);
    joint_histogram<Tin, 26, Thist, Tout, bitsThist>(index_stream[26], j_h_pe_stream[26], dim);
    joint_histogram<Tin, 27, Thist, Tout, bitsThist>(index_stream[27], j_h_pe_stream[27], dim);
    joint_histogram<Tin, 28, Thist, Tout, bitsThist>(index_stream[28], j_h_pe_stream[28], dim);
    joint_histogram<Tin, 29, Thist, Tout, bitsThist>(index_stream[29], j_h_pe_stream[29], dim);
    joint_histogram<Tin, 30, Thist, Tout, bitsThist>(index_stream[30], j_h_pe_stream[30], dim);
    joint_histogram<Tin, 31, Thist, Tout, bitsThist>(index_stream[31], j_h_pe_stream[31], dim);
}

template <typename Tin, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_64(
    hls::stream<Tin> index_stream[64], hls::stream<Tout> j_h_pe_stream[64], unsigned int dim) {
#pragma HLS INLINE

    joint_histogram<Tin, 0, Thist, Tout, bitsThist>(index_stream[0], j_h_pe_stream[0], dim);
    joint_histogram<Tin, 1, Thist, Tout, bitsThist>(index_stream[1], j_h_pe_stream[1], dim);
    joint_histogram<Tin, 2, Thist, Tout, bitsThist>(index_stream[2], j_h_pe_stream[2], dim);
    joint_histogram<Tin, 3, Thist, Tout, bitsThist>(index_stream[3], j_h_pe_stream[3], dim);
    joint_histogram<Tin, 4, Thist, Tout, bitsThist>(index_stream[4], j_h_pe_stream[4], dim);
    joint_histogram<Tin, 5, Thist, Tout, bitsThist>(index_stream[5], j_h_pe_stream[5], dim);
    joint_histogram<Tin, 6, Thist, Tout, bitsThist>(index_stream[6], j_h_pe_stream[6], dim);
    joint_histogram<Tin, 7, Thist, Tout, bitsThist>(index_stream[7], j_h_pe_stream[7], dim);
    joint_histogram<Tin, 8, Thist, Tout, bitsThist>(index_stream[8], j_h_pe_stream[8], dim);
    joint_histogram<Tin, 9, Thist, Tout, bitsThist>(index_stream[9], j_h_pe_stream[9], dim);
    joint_histogram<Tin, 10, Thist, Tout, bitsThist>(index_stream[10], j_h_pe_stream[10], dim);
    joint_histogram<Tin, 11, Thist, Tout, bitsThist>(index_stream[11], j_h_pe_stream[11], dim);
    joint_histogram<Tin, 12, Thist, Tout, bitsThist>(index_stream[12], j_h_pe_stream[12], dim);
    joint_histogram<Tin, 13, Thist, Tout, bitsThist>(index_stream[13], j_h_pe_stream[13], dim);
    joint_histogram<Tin, 14, Thist, Tout, bitsThist>(index_stream[14], j_h_pe_stream[14], dim);
    joint_histogram<Tin, 15, Thist, Tout, bitsThist>(index_stream[15], j_h_pe_stream[15], dim);
    joint_histogram<Tin, 16, Thist, Tout, bitsThist>(index_stream[16], j_h_pe_stream[16], dim);
    joint_histogram<Tin, 17, Thist, Tout, bitsThist>(index_stream[17], j_h_pe_stream[17], dim);
    joint_histogram<Tin, 18, Thist, Tout, bitsThist>(index_stream[18], j_h_pe_stream[18], dim);
    joint_histogram<Tin, 19, Thist, Tout, bitsThist>(index_stream[19], j_h_pe_stream[19], dim);
    joint_histogram<Tin, 20, Thist, Tout, bitsThist>(index_stream[20], j_h_pe_stream[20], dim);
    joint_histogram<Tin, 21, Thist, Tout, bitsThist>(index_stream[21], j_h_pe_stream[21], dim);
    joint_histogram<Tin, 22, Thist, Tout, bitsThist>(index_stream[22], j_h_pe_stream[22], dim);
    joint_histogram<Tin, 23, Thist, Tout, bitsThist>(index_stream[23], j_h_pe_stream[23], dim);
    joint_histogram<Tin, 24, Thist, Tout, bitsThist>(index_stream[24], j_h_pe_stream[24], dim);
    joint_histogram<Tin, 25, Thist, Tout, bitsThist>(index_stream[25], j_h_pe_stream[25], dim);
    joint_histogram<Tin, 26, Thist, Tout, bitsThist>(index_stream[26], j_h_pe_stream[26], dim);
    joint_histogram<Tin, 27, Thist, Tout, bitsThist>(index_stream[27], j_h_pe_stream[27], dim);
    joint_histogram<Tin, 28, Thist, Tout, bitsThist>(index_stream[28], j_h_pe_stream[28], dim);
    joint_histogram<Tin, 29, Thist, Tout, bitsThist>(index_stream[29], j_h_pe_stream[29], dim);
    joint_histogram<Tin, 30, Thist, Tout, bitsThist>(index_stream[30], j_h_pe_stream[30], dim);
    joint_histogram<Tin, 31, Thist, Tout, bitsThist>(index_stream[31], j_h_pe_stream[31], dim);
    joint_histogram<Tin, 32, Thist, Tout, bitsThist>(index_stream[32], j_h_pe_stream[32], dim);
    joint_histogram<Tin, 33, Thist, Tout, bitsThist>(index_stream[33], j_h_pe_stream[33], dim);
    joint_histogram<Tin, 34, Thist, Tout, bitsThist>(index_stream[34], j_h_pe_stream[34], dim);
    joint_histogram<Tin, 35, Thist, Tout, bitsThist>(index_stream[35], j_h_pe_stream[35], dim);
    joint_histogram<Tin, 36, Thist, Tout, bitsThist>(index_stream[36], j_h_pe_stream[36], dim);
    joint_histogram<Tin, 37, Thist, Tout, bitsThist>(index_stream[37], j_h_pe_stream[37], dim);
    joint_histogram<Tin, 38, Thist, Tout, bitsThist>(index_stream[38], j_h_pe_stream[38], dim);
    joint_histogram<Tin, 39, Thist, Tout, bitsThist>(index_stream[39], j_h_pe_stream[39], dim);
    joint_histogram<Tin, 40, Thist, Tout, bitsThist>(index_stream[40], j_h_pe_stream[40], dim);
    joint_histogram<Tin, 41, Thist, Tout, bitsThist>(index_stream[41], j_h_pe_stream[41], dim);
    joint_histogram<Tin, 42, Thist, Tout, bitsThist>(index_stream[42], j_h_pe_stream[42], dim);
    joint_histogram<Tin, 43, Thist, Tout, bitsThist>(index_stream[43], j_h_pe_stream[43], dim);
    joint_histogram<Tin, 44, Thist, Tout, bitsThist>(index_stream[44], j_h_pe_stream[44], dim);
    joint_histogram<Tin, 45, Thist, Tout, bitsThist>(index_stream[45], j_h_pe_stream[45], dim);
    joint_histogram<Tin, 46, Thist, Tout, bitsThist>(index_stream[46], j_h_pe_stream[46], dim);
    joint_histogram<Tin, 47, Thist, Tout, bitsThist>(index_stream[47], j_h_pe_stream[47], dim);
    joint_histogram<Tin, 48, Thist, Tout, bitsThist>(index_stream[48], j_h_pe_stream[48], dim);
    joint_histogram<Tin, 49, Thist, Tout, bitsThist>(index_stream[49], j_h_pe_stream[49], dim);
    joint_histogram<Tin, 50, Thist, Tout, bitsThist>(index_stream[50], j_h_pe_stream[50], dim);
    joint_histogram<Tin, 51, Thist, Tout, bitsThist>(index_stream[51], j_h_pe_stream[51], dim);
    joint_histogram<Tin, 52, Thist, Tout, bitsThist>(index_stream[52], j_h_pe_stream[52], dim);
    joint_histogram<Tin, 53, Thist, Tout, bitsThist>(index_stream[53], j_h_pe_stream[53], dim);
    joint_histogram<Tin, 54, Thist, Tout, bitsThist>(index_stream[54], j_h_pe_stream[54], dim);
    joint_histogram<Tin, 55, Thist, Tout, bitsThist>(index_stream[55], j_h_pe_stream[55], dim);
    joint_histogram<Tin, 56, Thist, Tout, bitsThist>(index_stream[56], j_h_pe_stream[56], dim);
    joint_histogram<Tin, 57, Thist, Tout, bitsThist>(index_stream[57], j_h_pe_stream[57], dim);
    joint_histogram<Tin, 58, Thist, Tout, bitsThist>(index_stream[58], j_h_pe_stream[58], dim);
    joint_histogram<Tin, 59, Thist, Tout, bitsThist>(index_stream[59], j_h_pe_stream[59], dim);
    joint_histogram<Tin, 60, Thist, Tout, bitsThist>(index_stream[60], j_h_pe_stream[60], dim);
    joint_histogram<Tin, 61, Thist, Tout, bitsThist>(index_stream[61], j_h_pe_stream[61], dim);
    joint_histogram<Tin, 62, Thist, Tout, bitsThist>(index_stream[62], j_h_pe_stream[62], dim);
    joint_histogram<Tin, 63, Thist, Tout, bitsThist>(index_stream[63], j_h_pe_stream[63], dim);
}

#endif  // HISTOGRAM_H