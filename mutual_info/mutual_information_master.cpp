/******************************************
 *MIT License
 *
 *Copyright (c) [2019] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del
 *Sozzo, Marco Domenico Santambrogio]
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
 * High-Level-Synthesis implementation file for Mutual Information computation
 *
 ****************************************************************/
#ifndef __MUTUAL_INFO_CPP__
#define __MUTUAL_INFO_CPP__
#include <stdio.h>
#include <string.h>

#include "./include/hw/mutualInfo/entropy.h"
#include "./include/hw/mutualInfo/histogram.h"
#include "./include/hw/mutualInfo/mutual_info.hpp"
#include "./include/hw/mutualInfo/utils.hpp"
#include "assert.h"
#include "hls_math.h"
#include "stdlib.h"

void compute_metric(
    hls::stream<INPUT_DATA_TYPE> &flt_stream,
    INPUT_DATA_TYPE *input_ref,
    data_t *mutual_info,
    unsigned int packed_input_size,
    unsigned int input_size,
    unsigned int padding_pixels) {
#pragma HLS DATAFLOW

    static hls::stream<INPUT_DATA_TYPE> ref_stream("ref_stream");
#pragma HLS STREAM variable = ref_stream depth = 2

    static hls::stream<unsigned int> input_size_stream("input_size_stream");
#pragma HLS STREAM variable = input_size_stream depth = 2

    static hls::stream<HIST_INDEX_DATA_TYPE> index_stream[HIST_PE];
#pragma HLS STREAM variable = index_stream depth = 2

    static hls::stream<PACKED_HIST_PE_DATA_TYPE> j_h_pe_stream[HIST_PE];
#pragma HLS STREAM variable = j_h_pe_stream depth = 2

    static hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream("joint_j_h_stream");
#pragma HLS STREAM variable = joint_j_h_stream depth = 2
    static hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream_0("joint_j_h_stream_0");
#pragma HLS STREAM variable = joint_j_h_stream_0 depth = 2
    static hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream_1("joint_j_h_stream_1");
#pragma HLS STREAM variable = joint_j_h_stream_1 depth = 2
    static hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream_2("joint_j_h_stream_2");
#pragma HLS STREAM variable = joint_j_h_stream_2 depth = 2

    static hls::stream<PACKED_HIST_DATA_TYPE> row_hist_stream("row_hist_stream");
#pragma HLS STREAM variable = row_hist_stream depth = 2
    static hls::stream<PACKED_HIST_DATA_TYPE> col_hist_stream("col_hist_stream");
#pragma HLS STREAM variable = col_hist_stream depth = 2

    static hls::stream<HIST_TYPE> full_hist_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable = full_hist_split_stream depth = 2
    static hls::stream<HIST_TYPE> row_hist_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable = row_hist_split_stream depth = 2
    static hls::stream<HIST_TYPE> col_hist_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable = col_hist_split_stream depth = 2

    static hls::stream<OUT_ENTROPY_TYPE> full_entropy_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable = full_entropy_split_stream depth = 2
    static hls::stream<OUT_ENTROPY_TYPE> row_entropy_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable = row_entropy_split_stream depth = 2
    static hls::stream<OUT_ENTROPY_TYPE> col_entropy_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable = col_entropy_split_stream depth = 2

    static hls::stream<OUT_ENTROPY_TYPE> full_entropy_stream("full_entropy_stream");
#pragma HLS STREAM variable = full_entropy_stream depth = 2
    static hls::stream<OUT_ENTROPY_TYPE> row_entropy_stream("row_entropy_stream");
#pragma HLS STREAM variable = row_entropy_stream depth = 2
    static hls::stream<OUT_ENTROPY_TYPE> col_entropy_stream("col_entropy_stream");
#pragma HLS STREAM variable = col_entropy_stream depth = 2

    static hls::stream<data_t> mutual_information_stream("mutual_information_stream");
#pragma HLS STREAM variable = mutual_information_stream depth = 2

    // Step 1: read data from DDR and split them, write input size value to its
    // stream

    input_size_stream.write(input_size);

    // axi2stream<INPUT_DATA_TYPE, 0>(flt_stream, input_img, packed_input_size);
#ifndef CACHING
    axi2stream<INPUT_DATA_TYPE, 1>(ref_stream, input_ref, packed_input_size);
#else
    bram2stream<INPUT_DATA_TYPE>(ref_stream, input_ref, packed_input_size);
#endif

    join_and_split_stream<INPUT_DATA_TYPE, HIST_INDEX_DATA_TYPE, UNPACK_DATA_BITWIDTH, HIST_PE>(
        flt_stream, ref_stream, index_stream, packed_input_size);
    // End Step 1

    // Step 2: Compute two histograms in parallel
    WRAPPER_HIST(HIST_PE)<HIST_INDEX_DATA_TYPE, HIST_PE_TYPE, PACKED_HIST_PE_DATA_TYPE, MIN_HIST_PE_BITS>(
        index_stream, j_h_pe_stream, packed_input_size);
    sum_joint_histogram<
        PACKED_HIST_PE_DATA_TYPE,
        J_HISTO_ROWS * J_HISTO_COLS / ENTROPY_PE,
        PACKED_HIST_DATA_TYPE,
        HIST_PE,
        HIST_PE_TYPE,
        MIN_HIST_PE_BITS,
        HIST_TYPE,
        MIN_HIST_BITS>(j_h_pe_stream, joint_j_h_stream, padding_pixels);
    // End Step 2

    // Step 3: Compute histograms per row and column
    tri_stream<PACKED_HIST_DATA_TYPE, J_HISTO_ROWS * J_HISTO_COLS / ENTROPY_PE>(
        joint_j_h_stream, joint_j_h_stream_0, joint_j_h_stream_1, joint_j_h_stream_2);

    hist_row<PACKED_HIST_DATA_TYPE, J_HISTO_ROWS, J_HISTO_COLS / ENTROPY_PE, HIST_TYPE, MIN_HIST_BITS>(
        joint_j_h_stream_0, row_hist_stream);
    hist_col<PACKED_HIST_DATA_TYPE, J_HISTO_ROWS, J_HISTO_COLS / ENTROPY_PE>(joint_j_h_stream_1, col_hist_stream);
    // End Step 3

    // Step 4: Compute Entropies
    WRAPPER_ENTROPY(ENTROPY_PE)<
        PACKED_HIST_DATA_TYPE,
        HIST_TYPE,
        MIN_HIST_BITS,
        OUT_ENTROPY_TYPE,
        J_HISTO_ROWS * J_HISTO_COLS / ENTROPY_PE>(
        joint_j_h_stream_2, full_hist_split_stream, full_entropy_split_stream, full_entropy_stream);
    WRAPPER_ENTROPY(
        ENTROPY_PE)<PACKED_HIST_DATA_TYPE, HIST_TYPE, MIN_HIST_BITS, OUT_ENTROPY_TYPE, J_HISTO_ROWS / ENTROPY_PE>(
        row_hist_stream, row_hist_split_stream, row_entropy_split_stream, row_entropy_stream);
    WRAPPER_ENTROPY(
        ENTROPY_PE)<PACKED_HIST_DATA_TYPE, HIST_TYPE, MIN_HIST_BITS, OUT_ENTROPY_TYPE, J_HISTO_COLS / ENTROPY_PE>(
        col_hist_stream, col_hist_split_stream, col_entropy_split_stream, col_entropy_stream);
    // End Step 4

    // Step 5: Mutual information
    compute_mutual_information<OUT_ENTROPY_TYPE, data_t>(
        row_entropy_stream,
        col_entropy_stream,
        full_entropy_stream,
        mutual_information_stream,
        input_size_stream,
        padding_pixels);
    // End Step 5

    // Step 6: Write result back to DDR
    stream2axi<data_t>(mutual_info, mutual_information_stream);
    // End Step 6
}

#ifndef CACHING

#ifdef KERNEL_NAME
extern "C" {
void KERNEL_NAME
#else
void mutual_information_master
#endif  // KERNEL_NAME
    (hls::stream<INPUT_DATA_TYPE> &stream_input_img,
     INPUT_DATA_TYPE *input_ref,
     data_t *mutual_info,
     unsigned int input_size,
     unsigned int padding_pixels) {
#pragma HLS INTERFACE m_axi port = input_ref depth = fifo_in_depth num_write_outstanding = 1 max_write_burst_length = \
    2 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = mutual_info depth = 1 num_read_outstanding = 1 max_read_burst_length = 2 offset = \
    slave bundle = gmem2

//#pragma HLS INTERFACE s_axilite register port = input_img bundle = control
#pragma HLS INTERFACE s_axilite register port = input_ref bundle = control
#pragma HLS INTERFACE s_axilite register port = mutual_info bundle = control
#pragma HLS INTERFACE s_axilite register port = input_size bundle = control
#pragma HLS INTERFACE s_axilite register port = padding_pixels bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static unsigned int local_input_size = 0;
    local_input_size = input_size;
    unsigned int packed_input_size = input_size / HIST_PE;

    compute_metric(stream_input_img, input_ref, mutual_info, packed_input_size, local_input_size, padding_pixels);
}

#else  // CACHING

#ifdef KERNEL_NAME
extern "C" {
void KERNEL_NAME
#else
void mutual_information_master
#endif  // KERNEL_NAME
    (INPUT_DATA_TYPE *input_img,
     data_t *mutual_info,
     unsigned int functionality,
     int *status,
     unsigned int input_size,
     unsigned int padding_pixels) {
#pragma HLS INTERFACE m_axi port = input_img depth = fifo_in_depth num_write_outstanding = 1 max_write_burst_length = \
    2 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = mutual_info depth = 1 num_read_outstanding = 1 max_read_burst_length = 2 offset = \
    slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = status depth = 1 num_read_outstanding = 1 max_read_burst_length = 2 offset = \
    slave bundle = gmem2

#pragma HLS INTERFACE s_axilite register port = input_img bundle = control
#pragma HLS INTERFACE s_axilite register port = mutual_info bundle = control
#pragma HLS INTERFACE s_axilite register port = functionality bundle = control
#pragma HLS INTERFACE s_axilite register port = status bundle = control
#pragma HLS INTERFACE s_axilite register port = input_size bundle = control
#pragma HLS INTERFACE s_axilite register port = padding_pixels bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static INPUT_DATA_TYPE ref_img[NUM_INPUT_DATA] = {0};

#pragma HLS RESOURCE variable = ref_img core = RAM_1P_URAM

    static unsigned int local_input_size = 0;
    local_input_size = input_size;
    unsigned int packed_input_size = input_size / HIST_PE;

    switch (functionality) {
        case LOAD_IMG:
            copyData<INPUT_DATA_TYPE>(input_img, ref_img, packed_input_size);
            *status = 1;
            *mutual_info = 0.0;
            break;
        case COMPUTE:
            compute_metric(input_img, ref_img, mutual_info, packed_input_size, local_input_size, padding_pixels);
            *status = 1;
            break;
        default:
            *status = -1;
            *mutual_info = 0.0;
            break;
    }
}

#endif  // CACHING

#ifdef KERNEL_NAME

}  // extern "C"

#endif  // KERNEL_NAME
#endif  //__MUTUAL_INFO_CPP__