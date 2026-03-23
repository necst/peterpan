/*
MIT License

Copyright (c) 2023 Paolo Salvatore Galfano, Giuseppe Sorrentino

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

/***************************************************************
*
* Configuration header file for Versal 3D Image Registration
*
****************************************************************/
#ifndef CONSTANTS_H
#define CONSTANTS_H

typedef float data_t;
#define DATA_T_BITWIDTH 32

#define DIMENSION 512

#define HIST_PE 16
#define HIST_PE_EXPO 4
#define UNPACK_DATA_BITWIDTH 8
#define INPUT_DATA_BITWIDTH (HIST_PE*UNPACK_DATA_BITWIDTH)
// #define INPUT_DATA_BITWIDTH_INTERP 8 // TODO remove (use INPUT_DATA_BITWIDTH_FETCHER instead)
#define INPUT_DATA_BITWIDTH_FETCHER 256
#define INPUT_DATA_BITWIDTH_FETCHER_MIN 256 // DEPRECATED, use CHUNK_SIZE instead
#define CHUNK_SIZE 256 // same as INPUT_DATA_BITWIDTH_FETCHER_MIN
#define NUM_PIXELS_PER_READ 32
#define NUM_PIXELS_PER_READ_EXPO 5
#define NUM_INPUT_DATA (DIMENSION*DIMENSION/(HIST_PE))
#define N_COUPLES_MAX 512
#define J_HISTO_ROWS 256
#define J_HISTO_COLS J_HISTO_ROWS
#define HISTO_ROWS J_HISTO_ROWS
#define INTERVAL_NUMBER 256 // L, amount of levels we want for the binning process, thus at the output

#define SIZE_ROWS 512 // how many rows per aie tile
#define SIZE_COLS 512 // how many columns per aie tile

#define INIT_COLS {-64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33} // the initial columns for the aie tiles

#define ENTROPY_PE 8

#define MAX_INT_PE_PLIOS 128 // max number of PLIOS towards and from AIE interpolator PEs
#define INT_PE 128
#define INT_PE_EXPO 7
#define DIV_EXPO 7
#define INT_PE_SATURATED 128
#define INT_PE_EXPO_SATURATED 7
#define DIV_EXPO_SATURATED 7

#define AIE_PATTERNS 0
#define AIE_PATTERN_OFFSETS 0

#define DS_PE 8 // number of data-scheduler PEs (scheduler_IPE)
#define DS_PE_MASK (DS_PE - 1)
#define INT_PE_PER_DS 16 // number of IPEs per data-scheduler PE
#define INT_PE_PER_DS_EXPO 4

#define COORD_BITWIDTH 32

#define AIE_TOP_LEFT 0
#define AIE_TOP_RIGHT 1
#define AIE_BOTTOM_LEFT 2
#define AIE_BOTTOM_RIGHT 3

#endif
