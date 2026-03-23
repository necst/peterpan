#pragma once
#include <adf.h>

void out_pixels_merger(
    input_stream<uint8>* restrict in0, input_stream<uint8>* restrict in1, output_stream<uint8>* restrict out);
