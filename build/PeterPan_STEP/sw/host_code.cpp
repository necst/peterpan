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

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "../common/common.h" // TODO non dovrebbe servire
#include "./include/software_mi/software_mi.hpp"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"
#include "opencv2/opencv.hpp"

// For hw emulation, run in sw directory: source ./setup_emu.sh -s on

#define DEVICE_ID 1

#ifndef ONLYTX
#include "./include/versal_3dir/Versal3DIR.cpp"
#else
#include "./include/versal_3dir/Versal3DIR_onlyTX.cpp"
#endif

bool get_xclbin_path(std::string &xclbin_file);
std::ostream &bold_on(std::ostream &os);
std::ostream &bold_off(std::ostream &os);
void print_header(int n_couples, int n_row, int n_col, float TX, float TY,
                  float ANG_DEG);

float hw_transform(Versal3DIR &board, float TX, float TY, float ANG,
                   double *duration_exec = NULL) {
  board.set_transform_params(TX, TY, ANG);
  board.run(duration_exec);
  return board.read_mutual_information();
}

int main(int argc, char *argv[]) {
#ifndef ONLYTX
  printf("Testing registration STEP");
#else
  printf("Testing only transformation");
#endif

  int n_couples = 512;
  if (argc >= 2) {
    n_couples = atoi(argv[1]);
    if (n_couples > N_COUPLES_MAX)
      n_couples = N_COUPLES_MAX;
  }

  int n_row = 512;
  if (argc >= 3) {
    n_row = atoi(argv[2]);
    if (n_row > DIMENSION)
      n_row = DIMENSION;
  }

  int n_col = 512;
  if (argc >= 4) {
    n_col = atoi(argv[3]);
    if (n_col > DIMENSION)
      n_col = DIMENSION;
  }

  float TX = 30;
  if (argc >= 5) {
    TX = atof(argv[4]);
  }

  float TY = 30;
  if (argc >= 6) {
    TY = atof(argv[5]);
  }

  float ANG_DEG = 25;
  if (argc >= 7) {
    ANG_DEG = atof(argv[6]);
  }
  const float ANG = (ANG_DEG * M_PI) / 180.f; // radians

  int N_RUNS = 1;
  if (argc >= 8) {
    N_RUNS = atoi(argv[7]);
  }

  int device_id = 1;
  if (argc >= 9) {
    device_id = atoi(argv[8]);
  }

  print_header(n_couples, n_row, n_col, TX, TY, ANG_DEG);
  //------------------------------------------------LOADING
  // XCLBIN------------------------------------------
  std::string xclbin_file;
  if (!get_xclbin_path(xclbin_file))
    return EXIT_FAILURE;

  // Load xclbin
  std::cout << "1. Loading bitstream (" << xclbin_file << ") on device id "
            << device_id << "... ";
  xrt::device device = xrt::device(device_id);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Done" << std::endl;

  //----------------------------------------------INITIALIZING THE
  // BOARD------------------------------------------
  Versal3DIR board = Versal3DIR(device, xclbin_uuid, n_couples, n_row, n_col);

  printf("Original depth: %d\n", board.n_couples);
  printf("Original rows: %d\n", board.n_row);
  printf("Original cols: %d\n", board.n_col);
  printf("Padded depth: %d\n", n_couples + board.depth_padding);
  printf("Padded rows: %d\n", n_row + board.row_padding);
  printf("Padded cols: %d\n", n_col + board.col_padding);
  // Read input volumes from disk
  std::cout << "2. Reading input volumes from disk... ";
  if (board.read_volumes_from_file("dataset/", "dataset/") == -1)
    return -1;
  std::cout << "Done" << std::endl;

  //------------------------------------------------RUNNING
  // KERNELS------------------------------------------
  float output_data = -1.0f;

  if (N_RUNS > 1)
    std::cout << "4. Running " << N_RUNS << " times" << std::endl;
  else
    std::cout << "4. Running 1 time" << std::endl;

  // files contaning the execution times
  std::ofstream outfile;
  // set the filename as time_IPEX_DYY_NXX.csv where X is the Number of IPE, YY
  // is the image resolution (DIMENSION) and XX is the given n_couples
  std::string filename = "time_IPE" + std::to_string(INT_PE) + "_D" +
                         std::to_string(n_row) + "_R" + std::to_string(n_col) +
                         "_C" + std::to_string(n_couples) + "_peterpan.csv";
  outfile.open(filename, std::ios_base::out); // append instead of overwrite
  outfile << "exec_time,write_time,read_time" << std::endl;
  std::ofstream outfile_sw;
  outfile_sw.open("time_sw.csv",
                  std::ios_base::out); // append instead of overwrite
  outfile_sw << "time," << std::endl;

  double duration_sw_sec = 0;
  double duration_execution_sec = 0;
  double duration_read_flt_sec = 0;
  double duration_write_flt_sec = 0;
  double mutualinfo;

  for (int i = 0; i < N_RUNS; i++) {
    duration_execution_sec = 0;
    duration_read_flt_sec = 0;
    duration_sw_sec = 0;

    // Write input volumes to board's main memory
    duration_write_flt_sec = 0;
    std::cout << "3. Transfering volumes to board's main memory... ";
    board.write_floating_volume(&duration_write_flt_sec);
    board.write_reference_volume(&duration_write_flt_sec);
    std::cout << "Done\t\t[" << duration_write_flt_sec << " s]" << std::endl
              << std::flush;

    // Running on the board
    std::cout << bold_on << "5. Running 3DIR step on hardware... " << bold_off
              << std::flush;
    output_data = hw_transform(board, TX, TY, ANG, &duration_execution_sec);
    std::cout << bold_on << "Done\t\t\t[" << duration_execution_sec << " s]"
              << bold_off << std::flush << std::endl;

    //------------------------------------------------SAVING
    // RESULTS------------------------------------------
    std::cout << "6. Transfering output volume from board's main memory... ";
    board.read_flt_transformed(&duration_read_flt_sec);
    std::cout << "Done\t[" << duration_read_flt_sec << " s]" << std::endl;
    outfile << duration_execution_sec << "," << duration_write_flt_sec << ","
            << duration_read_flt_sec << std::endl;

    std::cout << "7. Writing output volume on disk... ";
    write_volume_to_file(board.output_flt, n_row, n_col, n_couples,
                         board.row_padding, board.col_padding,
                         board.depth_padding, "dataset_output/");
    std::cout << "Done" << std::endl;

    // ---------------------------------CONFRONTO PER VERIFICARE
    // L'ERRORE--------------------------------------
    if (i == N_RUNS - 1) {
      std::cout << bold_on << "8. Running software implementation... "
                << std::flush;
      int row_padding = board.row_padding;
      int col_padding = board.col_padding;
      int depth_padding = board.depth_padding;
      mutualinfo =
          software_mi(n_couples, n_row, n_col, row_padding, col_padding,
                      depth_padding, TX, TY, ANG, "dataset/", &duration_sw_sec);
      std::cout << "Done\t\t\t[" << duration_sw_sec << " s]" << bold_off
                << std::endl;
      std::cout << std::endl;
      outfile_sw << duration_sw_sec << std::endl;
    } else {
      std::cout << bold_on << "8. Skip software implementation run" << bold_off
                << std::endl;
    }
  }

  double tot_time =
      duration_write_flt_sec + duration_execution_sec + duration_read_flt_sec;

  std::cout << "--- RESULTS "
               "---------------------------------------------------------------"
               "-----\n\n";
  std::cout << "Mutual information:" << std::endl;
  std::cout << bold_on << "  HW:\t\t" << output_data << bold_off << std::endl;
  std::cout << "  SW:\t\t" << mutualinfo << std::endl;
  std::cout << "  Error:\t" << abs(mutualinfo - output_data) << std::endl;
  std::cout << std::endl;

  float speedup = duration_sw_sec / duration_execution_sec; // sw time / hw time
  std::cout << "Execution time:" << std::endl;
  std::cout << bold_on << "  HW:\t\t" << duration_execution_sec << bold_off
            << std::endl;
  std::cout << "  SW:\t\t" << duration_sw_sec << std::endl << std::endl;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << bold_on << "Speedup: " << speedup << bold_off << std::endl
            << std::endl;

#ifndef ONLYTX
  float absolute_error = abs((float)mutualinfo - output_data);
  float relative_error = 100.f * absolute_error / (float)mutualinfo;
  std::cout << "Error: " << std::scientific << absolute_error << " ("
            << std::scientific << relative_error << "%)" << std::endl;
#endif

  return 0;
}

bool get_xclbin_path(std::string &xclbin_file) {
  // Judge emulation mode accoring to env variable
  char *env_emu;
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
    std::cout << bold_on << "Program running in hardware mode" << bold_off
              << std::endl;
    xclbin_file = "PeterPan.xclbin";
  }

  std::cout << std::endl << std::endl;
  return true;
}

std::ostream &bold_on(std::ostream &os) { return os << "\e[1m"; }

std::ostream &bold_off(std::ostream &os) { return os << "\e[0m"; }

void print_header(int n_couples, int n_row, int n_col, float TX, float TY,
                  float ANG_DEG) {
  std::cout << std::endl << std::endl;
  std::cout << "   __      __                 _   ____  _____ _____ _____  "
            << std::endl;
  std::cout
      << "   \\ \\    / /                | | |___ \\|  __ \\_   _|  __ \\ "
      << std::endl;
  std::cout << "    \\ \\  / /__ _ __ ___  __ _| |   __) | |  | || | | |__) |"
            << std::endl;
  std::cout << "     \\ \\/ / _ \\ '__/ __|/ _` | |  |__ <| |  | || | |  _  / "
            << std::endl;
  std::cout
      << "      \\  /  __/ |  \\__ \\ (_| | |  ___) | |__| || |_| | \\ \\ "
      << std::endl;
  std::cout
      << "       \\/ \\___|_|  |___/\\__,_|_| |____/|_____/_____|_|  \\_\\"
      << std::endl;
  std::cout << std::endl << std::endl;

  std::cout << "+-----------------------+" << std::endl;
  std::cout << "| DATASET:              |" << std::endl;
  std::cout << "|   Resolution\t" << bold_on << n_row << 'x' << n_col
            << bold_off << " | " << std::endl;
  std::cout << "|   Depth\t" << bold_on << n_couples << bold_off << "     |"
            << std::endl;
  std::cout << "|   Rows\t" << bold_on << n_row << bold_off << "        |"
            << std::endl;
  std::cout << "|   Cols\t" << bold_on << n_col << bold_off << "        |"
            << std::endl;
  std::cout << "+-----------------------+" << std::endl;
  std::cout << "| TRANSFORMATION:       |" << std::endl;
  std::cout << "|   TX\t\t" << bold_on << TX << bold_off << " px   |"
            << std::endl;
  std::cout << "|   TY\t\t" << bold_on << TY << bold_off << " px   |"
            << std::endl;
  std::cout << "|   ANG\t\t" << bold_on << ANG_DEG << bold_off << " deg  |"
            << std::endl;
  std::cout << "+-----------------------+" << std::endl;
  std::cout << std::endl;
  std::cout << std::fixed << std::setprecision(6);
}
