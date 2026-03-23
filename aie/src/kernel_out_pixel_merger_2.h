#pragma once
#include <adf.h>

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"

// ===========================================================
// Template kernel: out_pixels_merger
// Reads blocks of 32 * LEVEL bytes from two input streams and writes them to output.
// ===========================================================

template <int LEVEL, int VEC_SIZE>
void out_pixels_merger(
    input_stream<uint8>* restrict in0, input_stream<uint8>* restrict in1, output_stream<uint8>* restrict out);
