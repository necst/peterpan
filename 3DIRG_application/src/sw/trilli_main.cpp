/******************************************
* MIT License
*
* Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide
Conficconi, Eleonora D'Arnese
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.

*MIT License
*
*Copyright (c) [2019] [Davide Conficconi, Eleonora D'Arnese, Marco Domenico
Santambrogio]
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
 * main of the whole application
 * credits goes also to the author of this repo:
 *https://github.com/mariusherzog/ImageRegistration
 *
 ****************************************************************/

// #define HW_REG 1

#ifdef HW_REG
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"
#else
#include "./include/software_mi/software_mi.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "../../../common/constants.h"
#include "app/imagefusion.hpp"
#include "core/fusion_algorithms.hpp"
#include "core/register_algorithms.hpp"
#define DEVICE_ID 0

using namespace cv;
using namespace std;
using namespace std::placeholders;

bool get_xclbin_path(std::string& xclbin_file);
std::ostream& bold_on(std::ostream& os);
std::ostream& bold_off(std::ostream& os);
void print_header(int n_couples, float TX, float TY, float ANG_DEG);

void getBackwardSplit(size_t size, char* string,
                      char* dst) {  //, char * prefix){
    int i;
    int j;
    for (i = size; i >= 0; i--) {
        if (string[i] == '.') j = i;
        if (string[i] == '/') break;
    }
    strncpy(dst, string + i + 1, (size - j - 1));
    dst[size - j - 1] = '\0';
}

void getFinalName(char* im1name, char* im2name, char* finalname) {
    size_t im1size = strlen(im1name);
    size_t im2size = strlen(im2name);
    strcat(finalname, im1name);
    strcat(finalname, im2name);
    const char* format = ".jpeg";
#ifdef HW_REG
    const char* currplat = "-hw";

#else
    const char* currplat = "-sw";

#endif

    strcat(finalname, currplat);
    strcat(finalname, format);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cout << "[WARNING] Iron needs more arguments: " << argc << "are given" << std::endl << std::endl;
        std::cout << "Example usage:" << std::endl;
        std::cout << "\t <iron executable> <reference img> <floating img> <img "
                     "result folder> <n_couples> <n_row> <n_col> [<rangeX>], "
                     "[<rangeY>], [<rangeAngZ>]"
                  << std::endl;
        return -1;
    }
    std::cout << "Starting Trilli" << std::endl;
    int n_couples = atoi(argv[4]);
    int n_row = atoi(argv[5]);
    int n_col = atoi(argv[6]);
    int rangeX = argc >= 6 ? atoi(argv[7]) : 256;
    int rangeY = argc >= 7 ? atoi(argv[8]) : 256;
    float rangeAngZ = argc >= 8 ? atof(argv[9]) : 1.0;

    std::cout << "Number of couples: " << n_couples << std::endl;
    std::cout << "Number of rows: " << n_row << std::endl;
    std::cout << "Number of columns: " << n_col << std::endl;
    std::cout << "RangeX: " << rangeX << std::endl;
    std::cout << "RangeY: " << rangeY << std::endl;
    std::cout << "RangeAngZ: " << rangeAngZ << std::endl;

    std::cout << "Number of couples: " << n_couples << std::endl;
    int padding = (NUM_PIXELS_PER_READ - (n_couples % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ;
    std::cout << "Padding: " << padding << std::endl;

    auto available_fusion_names = imagefusion::fusion_strategies();
    auto available_register_names = imagefusion::register_strategies();
    file_repository files(argv[1], argv[2]);
    std::vector<cv::Mat> reference_image = files.reference_image_3d(n_couples);
    std::vector<cv::Mat> floating_image = files.floating_image_3d(n_couples);
    uint8_t* buffer_ref = new uint8_t[n_row * n_col * (n_couples + padding)];
    uint8_t* buffer_flt = new uint8_t[n_row * n_col * (n_couples + padding)];
    cast_mats_to_vector(buffer_ref, reference_image, n_row, n_col, n_couples, 0, padding);
    cast_mats_to_vector(buffer_flt, floating_image, n_row, n_col, n_couples, 0, padding);
    std::vector<uint8_t> ref_vol = std::vector<uint8_t>(buffer_ref, buffer_ref + n_row * n_col * (n_couples + padding));
    std::vector<uint8_t> flt_vol = std::vector<uint8_t>(buffer_flt, buffer_flt + n_row * n_col * (n_couples + padding));
#ifdef HW_REG

    std::cout << "HW_REG" << std::endl;
    //------------------------------------------------LOADING
    // XCLBIN------------------------------------------
    std::string xclbin_file;
    if (!get_xclbin_path(xclbin_file)) return EXIT_FAILURE;

    // Load xclbin
    std::cout << "1. Loading bitstream (" << xclbin_file << ")... ";
    xrt::device device = xrt::device(DEVICE_ID);
    xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
    std::cout << "Done" << std::endl;
    //----------------------------------------------INITIALIZING THE
    // BOARD------------------------------------------
    Versal3DIR board = Versal3DIR(device, xclbin_uuid, n_couples, n_row, n_col);
    std::vector<uint8_t> output_volume(board.buffer_size);
    board.output_flt = output_volume.data();

    printf("Padding: %d\n", board.padding);
    auto ct_path = std::string(argv[1]);
    auto pet_pat = std::string(argv[2]);
    std::ofstream out_file;
    out_file.open("TRILLI_pow.csv", std::ios::out);
    out_file << "execTime,withPCIE_time" << std::endl;
    if (board.read_volumes_from_file(ct_path, pet_pat) == -1) return -1;
    auto start_pcie = std::chrono::high_resolution_clock::now();
    double exec_time = imagefusion::perform_fusion_from_files_3d(
        reference_image, floating_image, "mutualinformation", "alphablend", board, rangeX, rangeY, rangeAngZ);
    auto end_pcie = std::chrono::high_resolution_clock::now();

    write_volume_to_file(board.output_flt, DIMENSION, n_couples, 0, padding, argv[3]);
    std::cout << "Saving Volumes" << std::endl;
    out_file << exec_time << "," << std::chrono::duration<double>(end_pcie - start_pcie).count() << std::endl;
    out_file.close();

#else
    uint8_t* registered_volume = new uint8_t[n_row * n_col * (n_couples + padding)];
    imagefusion::perform_fusion_from_files_3d(
        reference_image,
        floating_image,
        ref_vol,
        flt_vol,
        "mutualinformation",
        "alphablend",
        n_couples,
        n_row,
        n_col,
        padding,
        rangeX,
        rangeY,
        rangeAngZ,
        registered_volume);
    std::cout << "Saving Volumes" << std::endl;
    write_volume_to_file(registered_volume, n_row, n_col, n_couples, 0, padding, argv[3]);
    delete[] registered_volume;
#endif
}

bool get_xclbin_path(std::string& xclbin_file) {
    // Judge emulation mode accoring to env variable
    char* env_emu;
    if (env_emu = getenv("XCL_EMULATION_MODE")) {
        std::string mode(env_emu);
        if (mode == "hw_emu") {
            std::cout << "Program running in hardware emulation mode" << std::endl;
            xclbin_file = "overlay_hw_emu.xclbin";
        } else {
            std::cout << "[ERROR] Unsupported Emulation Mode: " << mode << std::endl;
            return false;
        }
    } else {
        std::cout << bold_on << "Program running in hardware mode" << bold_off << std::endl;
        xclbin_file = "overlay_hw.xclbin";
    }

    std::cout << std::endl << std::endl;
    return true;
}

std::ostream& bold_on(std::ostream& os) { return os << "\e[1m"; }

std::ostream& bold_off(std::ostream& os) { return os << "\e[0m"; }

void print_header(int n_couples, float TX, float TY, float ANG_DEG) {
    std::cout << std::endl << std::endl;
    std::cout << "   __      __                 _   ____  _____ _____ _____  " << std::endl;
    std::cout << "   \\ \\    / /                | | |___ \\|  __ \\_   _|  __ \\ " << std::endl;
    std::cout << "    \\ \\  / /__ _ __ ___  __ _| |   __) | |  | || | | |__) |" << std::endl;
    std::cout << "     \\ \\/ / _ \\ '__/ __|/ _` | |  |__ <| |  | || | |  _  / " << std::endl;
    std::cout << "      \\  /  __/ |  \\__ \\ (_| | |  ___) | |__| || |_| | \\ \\ " << std::endl;
    std::cout << "       \\/ \\___|_|  |___/\\__,_|_| |____/|_____/_____|_|  \\_\\" << std::endl;
    std::cout << std::endl << std::endl;

    std::cout << "+-----------------------+" << std::endl;
    std::cout << "| DATASET:              |" << std::endl;
    std::cout << "|   Resolution\t" << bold_on << DIMENSION << 'x' << DIMENSION << bold_off << " | " << std::endl;
    std::cout << "|   Depth\t" << bold_on << n_couples << bold_off << "     |" << std::endl;
    std::cout << "+-----------------------+" << std::endl;
    std::cout << "| TRANSFORMATION:       |" << std::endl;
    std::cout << "|   TX\t\t" << bold_on << TX << bold_off << " px   |" << std::endl;
    std::cout << "|   TY\t\t" << bold_on << TY << bold_off << " px   |" << std::endl;
    std::cout << "|   ANG\t\t" << bold_on << ANG_DEG << bold_off << " deg  |" << std::endl;
    std::cout << "+-----------------------+" << std::endl;
    std::cout << std::endl;
}
