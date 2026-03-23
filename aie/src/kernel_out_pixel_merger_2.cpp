#include "kernel_out_pixel_merger_2.h"

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"
#include "common.h"

template <int LEVEL, int VEC_SIZE>
void out_pixels_merger(
    input_stream<uint8>* restrict in0, input_stream<uint8>* restrict in1, output_stream<uint8>* restrict out) {
    // reading loop_size (4 bytes)
    uint8 loop_size[4];
    if (LEVEL == 1) {
        loop_size[0] = readincr(in0);
        loop_size[1] = readincr(in0);
        loop_size[2] = readincr(in0);
        loop_size[3] = readincr(in0);
    } else {
        uint8 fake_loop_size[4];
        loop_size[0] = readincr(in0);
        loop_size[1] = readincr(in0);
        loop_size[2] = readincr(in0);
        loop_size[3] = readincr(in0);
        fake_loop_size[0] = readincr(in1);
        fake_loop_size[1] = readincr(in1);
        fake_loop_size[2] = readincr(in1);
        fake_loop_size[3] = readincr(in1);
    }

    int loop_size_final = ((uint32)loop_size[0]) | (((uint32)loop_size[1]) << 8) | (((uint32)loop_size[2]) << 16) |
                          (((uint32)loop_size[3]) << 24);

    if (LEVEL == 1 && INT_PE > 64) {
        writeincr(out, (uint8)(loop_size_final & 0xFF));
        writeincr(out, (uint8)((loop_size_final >> 8) & 0xFF));
        writeincr(out, (uint8)((loop_size_final >> 16) & 0xFF));
        writeincr(out, (uint8)((loop_size_final >> 24) & 0xFF));
    }

    printf("loop_size_final: %d (LEVEL=%d)\n", loop_size_final, LEVEL);

    for (int i = 0; i < loop_size_final; i++) chess_prepare_for_pipelining {
            aie::vector<uint8, VEC_SIZE> vector_0 = readincr_v<VEC_SIZE>(in0);
            aie::vector<uint8, VEC_SIZE> vector_1 = readincr_v<VEC_SIZE>(in1);

            writeincr(out, vector_0);
            writeincr(out, vector_1);
        }
}
