#include "kernel_in_pixels_splitter.h"

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"
#include "common.h"

void in_pixels_splitter(
    input_window_uint8* restrict in, output_window_uint8* restrict out0, output_window_uint8* restrict out1) {
    // --- reading header for first IPE
    window_acquire(in);
    aie::vector<uint8, 32> metadata = window_readincr_v32(in);
    aie::vector<uint8, 32> temp1 = window_readincr_v32(in);
    aie::vector<uint8, 32> temp2 = window_readincr_v32(in);
    aie::vector<uint8, 32> temp3 = window_readincr_v32(in);
    window_release(in);

    // --- rebuilding n_couples from header
    // haeder structure:
    //  metadata[0]       : ID of the IPE
    //  metadata[1,2,3]   : unused
    //  metadata[4,5,6,7] : n_couples

    uint8 ID = metadata.get(0);
    if (ID == 0) {
        for (int i = 0; i < 32; i++) {
            printf("--- splitter 0: metadata[%d]=%d\n", i, metadata.get(i));
        }
    }

    uint8 n_couples_bytes[4];
    n_couples_bytes[0] = metadata.get(4);
    n_couples_bytes[1] = metadata.get(5);
    n_couples_bytes[2] = metadata.get(6);
    n_couples_bytes[3] = metadata.get(7);
    int n_couples = ((uint32)n_couples_bytes[0]) + (((uint32)n_couples_bytes[1]) << 8) +
                    (((uint32)n_couples_bytes[2]) << 16) + (((uint32)n_couples_bytes[3]) << 24);

    printf("splitter: n_couples=%d\n", n_couples);

    // --- passing header to first IPE
    window_acquire(out0);
    window_writeincr(out0, metadata);
    window_writeincr(out0, temp1);
    window_writeincr(out0, temp2);
    window_writeincr(out0, temp3);
    window_release(out0);

    // --- reading header for second IPE
    window_acquire(in);
    aie::vector<uint8, 32> w0 = window_readincr_v32(in);
    aie::vector<uint8, 32> w1 = window_readincr_v32(in);
    aie::vector<uint8, 32> w2 = window_readincr_v32(in);
    aie::vector<uint8, 32> w3 = window_readincr_v32(in);
    window_release(in);

    // --- passing header to second IPE
    window_acquire(out1);
    window_writeincr(out1, w0);
    window_writeincr(out1, w1);
    window_writeincr(out1, w2);
    window_writeincr(out1, w3);
    window_release(out1);

    // loop range description:
    // - volume: DIMENSION * DIMENSION * n_couples
    // - divided by size of chunk: 32 pixels per chunk
    // - divided by INT_PE: one in_pixels_splitter per pair of IPEs (INT_PE/2), times two workloads per IPE
    // - divided by 4: one workload is 4 chunks of pixels (top-left, top-right, bottom-left, bottom-right)

    printf(
        "I will read %lu pixels\n", 32 * 4 * 2 * (((uint64_t)(DIMENSION * DIMENSION * n_couples / 1) / INT_PE) / 32));

    for (int i = 0; i < ((DIMENSION * DIMENSION * n_couples / 1) / INT_PE) / 32; i++)
        chess_loop_range(
            DIMENSION * DIMENSION / (INT_PE * 4),
        )  // TODO check again if it's correct (is it times 32 or not?)
            chess_prepare_for_pipelining {
            window_acquire(in);
            aie::vector<uint8, 32> vector_0_0 = window_readincr_v32(in);
            aie::vector<uint8, 32> vector_0_1 = window_readincr_v32(in);
            aie::vector<uint8, 32> vector_0_2 = window_readincr_v32(in);
            aie::vector<uint8, 32> vector_0_3 = window_readincr_v32(in);
            window_release(in);

            window_acquire(in);
            aie::vector<uint8, 32> vector_1_0 = window_readincr_v32(in);
            aie::vector<uint8, 32> vector_1_1 = window_readincr_v32(in);
            aie::vector<uint8, 32> vector_1_2 = window_readincr_v32(in);
            aie::vector<uint8, 32> vector_1_3 = window_readincr_v32(in);
            window_release(in);

            window_acquire(out0);
            window_writeincr(out0, vector_0_0);
            window_writeincr(out0, vector_0_1);
            window_writeincr(out0, vector_0_2);
            window_writeincr(out0, vector_0_3);
            window_release(out0);

            window_acquire(out1);
            window_writeincr(out1, vector_1_0);
            window_writeincr(out1, vector_1_1);
            window_writeincr(out1, vector_1_2);
            window_writeincr(out1, vector_1_3);
            window_release(out1);
        }
}
