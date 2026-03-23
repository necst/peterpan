#pragma once

#include "../common/common.h"
#include "../mutual_info/include/hw/mutualInfo/mutual_info.hpp"
#include "hls_burst_maxi.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

constexpr int TRIP_VARIABLE_LAYERS_MIN = 1;
constexpr int TRIP_VARIABLE_LAYERS_MAX = 512;
constexpr int TRIP_VARIABLE_LAYERS_AVG = 256;

constexpr int TRIP_VARIABLE_VOLUME_MIN =
    DIMENSION * DIMENSION * TRIP_VARIABLE_LAYERS_MIN;
constexpr int TRIP_VARIABLE_VOLUME_MAX =
    DIMENSION * DIMENSION * TRIP_VARIABLE_LAYERS_MAX;
constexpr int TRIP_VARIABLE_VOLUME_AVG =
    DIMENSION * DIMENSION * TRIP_VARIABLE_LAYERS_AVG;

constexpr int fifo_in_depth_smi = N_COUPLES_MAX * DIMENSION * DIMENSION;
constexpr int fifo_out_depth_smi = N_COUPLES_MAX * DIMENSION * DIMENSION;

void read_original_whole(
    hls::stream<ap_axis<COORD_BITWIDTH, 0, 0, 0>> &coords_in,
    hls::burst_maxi<ap_uint<INPUT_DATA_BITWIDTH_FETCHER>> &float_original,
    hls::stream<ap_uint<INPUT_DATA_BITWIDTH_FETCHER>> &float_out, int n_couples,
    int n_row, int n_col) {
  const int VOLUME = n_couples * n_row * n_col;

  int32_t coord;

loop_read_original_whole:
  for (int q = 0, slice = 0; q < VOLUME; q++, slice++) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = TRIP_VARIABLE_VOLUME_MIN max =                \
    TRIP_VARIABLE_VOLUME_MAX avg = TRIP_VARIABLE_VOLUME_AVG

    if (slice == n_couples || slice == 0) {
      coord = coords_in.read().data;
      if (coord != -1) {
        coord >>= NUM_PIXELS_PER_READ_EXPO;
        float_original.read_request(coord, n_couples);
      }
      slice = 0;
    }

    ap_uint<INPUT_DATA_BITWIDTH_FETCHER> pixel;
    if (coord != -1)
      pixel = float_original.read();
    else
      pixel = 0;

    float_out.write(pixel);
  }
}
