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

#pragma once
#include <iostream>
#include <unordered_map>

#include "opencv2/opencv.hpp"

// CONFIGURATION: (UN)COMMENT THE DEFINES AS NEEDED

// #define USE_OLD_FORMAT           // use old format for storing input and output volumes
// #define USE_FLOOR                   // truncate transformed coordinates instead of rounding them
// #define USE_FLOAT_INDEX          // compute transformed buffer index without rounding the coordinates first (inaccurate results) 
// #define DEBUG_ACCESSED_INDEXES   // print the index for each read of the input volume
#define TRACK_READS                 // count the ratio of sequential reads over total reads (slower)
#define COMPILE_WITHOUT_DCMTK       // compile without DICOM support (to avoid linking dcmtk libs)
// #define KEEP_BORDER_PADDING         // keep the border padding when storing the output volume

#ifndef COMPILE_WITHOUT_DCMTK
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimage/diregist.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#endif

#define MODE_NEAREST false
#define MODE_BILINEAR true

// POSSIBLE FORMATS
enum ImageFormat { DICOM, PNG };

int read_volume_from_file(uint8_t *volume, const int n_row, const int n_col,
                          const int N_COUPLES, const int row_padding,
                          const int col_padding, const int DEPTH_PADDING,
                          const std::string &path,
                          const ImageFormat imageFormat = ImageFormat::PNG);
// int read_volume_from_file_nopad(
//     uint8_t *volume, const int n_row, const int n_col, const int N_COUPLES,
//     const std::string &path, const ImageFormat imageFormat =
//     ImageFormat::PNG);

void write_volume_to_file(uint8_t *volume, const int N_ROW, const int N_COL,
                          const int N_COUPLES, const int row_padding,
                          const int col_padding, const int DEPTH_PADDING,
                          const std::string &path);
void write_volume_to_file(uint8_t *volume, const int N_ROW, const int N_COL,
                          const int N_COUPLES, const int BOARDER_PADDING,
                          const int DEPTH_PADDING, const std::string &path);

uint8_t track_reads(uint8_t *mem, const int index, float *ratio = NULL);
void get_cache_stats(int &hits, int &misses, int &size);
void reset_cache_stats();

void transform_volume(uint8_t *volume_src, uint8_t *volume_dest, const float TX,
                      const float TY, const float ANG, const int n_row,
                      int n_col, const int LAYERS,
                      const bool bilinear_interpolation = MODE_NEAREST);