#include "kernel_out_pixels_merger.h"

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"
#include "common.h"

void out_pixels_merger(
    input_stream<uint8>* restrict in0, input_stream<uint8>* restrict in1, output_stream<uint8>* restrict out) {
    // reading loop_size
    uint8 loop_size[4];
    loop_size[0] = readincr(in0);
    loop_size[1] = readincr(in0);
    loop_size[2] = readincr(in0);
    loop_size[3] = readincr(in0);

    int loop_size_final = ((uint32)loop_size[0]) + (((uint32)loop_size[1]) << 8) + (((uint32)loop_size[2]) << 16) +
                          (((uint32)loop_size[3]) << 24);

    for (int i = 0; i < loop_size_final; i++) chess_prepare_for_pipelining {
            aie::vector<uint8, 32> vector_0 = readincr_v<32>(in0);
            aie::vector<uint8, 32> vector_1 = readincr_v<32>(in1);

            writeincr(out, vector_0);
            writeincr(out, vector_1);
        }
}
