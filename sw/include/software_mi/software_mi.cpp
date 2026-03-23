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

#include "software_mi.hpp"

#define J_HISTO_ROWS 256
#define J_HISTO_COLS 256
#define ANOTHER_DIMENSION 256

double software_mi(
    int n_couples,
    int n_row,
    int n_col,
    int row_padding,
    int col_padding,
    int depth_padding,
    const int TX,
    const int TY,
    const float ANG,
    const std::string& input_path,
    double* duration_sec) {
    const int N_COUPLES_TOTAL = n_couples + depth_padding;
    const int IMG_SIZE = (n_row + row_padding) * (n_col + col_padding);

    uint8_t* input_ref = new uint8_t[IMG_SIZE * N_COUPLES_TOTAL];
    uint8_t* input_flt = new uint8_t[IMG_SIZE * N_COUPLES_TOTAL];
    uint8_t* output_flt = new uint8_t[IMG_SIZE * N_COUPLES_TOTAL];

    if (read_volume_from_file(
            input_ref, n_row, n_col, n_couples, row_padding, col_padding, depth_padding, input_path) == -1) {
        std::cerr << "Could not open file" << std::endl;
        return 1;
    }
    if (read_volume_from_file(
            input_flt, n_row, n_col, n_couples, row_padding, col_padding, depth_padding, input_path) == -1) {
        std::cerr << "Could not open file" << std::endl;
        return 1;
    }

    Timer timer_sw;
    timer_sw.start();
    transform_volume(
        input_flt, output_flt, TX, TY, ANG, n_row + row_padding, n_col + col_padding, N_COUPLES_TOTAL, MODE_BILINEAR);
    int n_row_orig = n_row;
    int n_col_orig = n_col;
    n_row = n_row + row_padding;
    n_col = n_col + col_padding;
    // --------------------------------- Joint histogram
    // ---------------------------------
    double j_h[J_HISTO_ROWS][J_HISTO_COLS];
    for (int i = 0; i < J_HISTO_ROWS; i++)
        for (int j = 0; j < J_HISTO_COLS; j++) j_h[i][j] = 0.0;

    for (int k = 0; k < N_COUPLES_TOTAL - depth_padding; k++) {
        for (int i = 0; i < n_row; i++) {
            for (int j = 0; j < n_col; j++) {
                unsigned int a = input_ref[i * n_col * N_COUPLES_TOTAL + j * N_COUPLES_TOTAL + k];
                unsigned int b = output_flt[i * n_col * N_COUPLES_TOTAL + j * N_COUPLES_TOTAL + k];
                j_h[a][b] += 1.0;
            }
        }
    }

    // Normalizza
    for (int i = 0; i < J_HISTO_ROWS; i++) {
        for (int j = 0; j < J_HISTO_COLS; j++) {
            j_h[i][j] /= ((N_COUPLES_TOTAL - depth_padding) * IMG_SIZE);
        }
    }

    // Entropia congiunta
    float entropy = 0.0;
    for (int i = 0; i < J_HISTO_ROWS; i++) {
        for (int j = 0; j < J_HISTO_COLS; j++) {
            float v = j_h[j][i];
            if (v > 1e-15) {
                entropy += v * log2(v);
            }
        }
    }
    entropy *= -1;

    // Marginali
    double href[ANOTHER_DIMENSION] = {0.0};
    double hflt[ANOTHER_DIMENSION] = {0.0};

    for (int i = 0; i < ANOTHER_DIMENSION; i++) {
        for (int j = 0; j < ANOTHER_DIMENSION; j++) {
            href[i] += j_h[i][j];
            hflt[i] += j_h[j][i];
        }
    }

    double eref = 0.0;
    for (int i = 0; i < ANOTHER_DIMENSION; i++) {
        if (href[i] > 1e-15) {
            eref += href[i] * log2(href[i]);
        }
    }
    eref *= -1;

    double eflt = 0.0;
    for (int i = 0; i < ANOTHER_DIMENSION; i++) {
        if (hflt[i] > 1e-15) {
            eflt += hflt[i] * log2(hflt[i]);
        }
    }
    eflt *= -1;

    double mutualinfo = eref + eflt - entropy;

    if (duration_sec != NULL) *duration_sec = timer_sw.getElapsedSeconds();

    write_volume_to_file(
        output_flt, n_row_orig, n_col_orig, n_couples, row_padding, col_padding, depth_padding, "dataset_sw_output/");

    delete[] input_flt;
    delete[] input_ref;
    delete[] output_flt;

    return mutualinfo;
}
