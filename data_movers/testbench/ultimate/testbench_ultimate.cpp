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

#include "../../../mutual_info/include/hw/mutualInfo/mutual_info.hpp"
#include "../../../mutual_info/mutual_information_master.cpp"
#include "../../../sw/include/image_utils/image_utils.cpp"
#include "../../../sw/include/image_utils/image_utils.hpp"
#include "../../../sw/include/software_mi/software_mi.hpp"
#include "../../fetcher_A.cpp"
#include "../../fetcher_B.cpp"
#include "../../fetcher_C.cpp"
#include "../../fetcher_D.cpp"
#include "../../scheduler_IPE.cpp"
// #include "../../setup_interpolator.cpp"
// #include "../../pixels_merger.cog.cpp"
// #include "../../IPEs_merger.cog.cpp"
#include <ap_axi_sdata.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <string>

#include "../../setup_aie.cpp"
#include "../../setup_mi.cpp"
#include "../../writer.cpp"
#include "../utils.hpp"

#define DEFAULT_TX 18.54458648
#define DEFAULT_TY -12.30391042
#define DEFAULT_ANG_DEG 20

#define AIE_PATH "../../../aie/"
#define SW_PATH "../../../sw/"
#define TEST_PATH "../"

#define AIE_FOLDER(x) (AIE_PATH x)
#define SW_FOLDER(x) (SW_PATH x)
#define TEST_FOLDER(x) (TEST_PATH x)

#define COORD_AXIS_W COORD_BITWIDTH, 0, 0, 0

typedef ap_uint<8> ORIGINAL_PIXEL_TYPE;
typedef ap_uint<INPUT_DATA_BITWIDTH_FETCHER> WIDE_PIXEL_TYPE;
typedef ap_uint<INPUT_DATA_BITWIDTH_FETCHER_MIN> AIE_PIXEL_TYPE;
typedef ap_uint<INPUT_DATA_BITWIDTH> MI_PIXEL_TYPE;
typedef ap_axis<COORD_AXIS_W> COORDS_TYPE;

// saturating num of output plios to MAX_INT_PE_PLIOS
#if INT_PE == 2 * MAX_INT_PE_PLIOS  // mapping 1 plio to 2 IPEs
#if INT_PE_SATURATED != INT_PE / 2
#error "Only 1:2 mapping (PLIOs to IPEs) is supported, set INT_PE_SATURATED to INT_PE/2, or INT_PE <= MAX_INT_PE_PLIOS"
#endif
#define NUM_OUTPUT_PLIOS MAX_INT_PE_PLIOS
#define NUM_INPUT_PLIOS MAX_INT_PE_PLIOS
#elif INT_PE < 2 * MAX_INT_PE_PLIOS
#define NUM_OUTPUT_PLIOS INT_PE / 2
#define NUM_INPUT_PLIOS INT_PE
#else
#error "INT_PE > 2 * MAX_INT_PE_PLIOS not supported"
#endif

void run_aie() {
    std::string command = std::string("make -C ") + AIE_PATH + " aie_simulate_x86";
    std::cout << std::endl << std::flush;
    int r = system(command.c_str());
    if (r != 0) {
        std::cerr << "Error running AIE simulation." << std::endl;
        exit(-1);
    }
    std::cout << "Done" << std::endl << std::endl;
}

void create_folder(const std::string& folder) {
    std::string command = std::string("mkdir -p ") + folder;
    int r = system(command.c_str());
    if (r != 0) {
        std::cerr << "Error creating folder: " << folder << std::endl;
        exit(-1);
    }
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 7) {
        std::cerr << "usage: ./testbench_ultimate <n_couples> <n_row> <n_col> "
                     "[<TX> <TY> <ANG_DEG>]"
                  << std::endl;
        std::cerr << "value of argc: " << argc << std::endl;
        return -1;
    }

    const int n_couples = atoi(argv[1]);
    const int depth_padding = (NUM_PIXELS_PER_READ - (n_couples % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ;
    const int row_padding = (NUM_PIXELS_PER_READ - (atoi(argv[2]) % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ;
    const int col_padding = (NUM_PIXELS_PER_READ - (atoi(argv[3]) % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ;
    const int n_row = atoi(argv[2]);
    const int n_col = atoi(argv[3]);
    const float TX = (argc == 7) ? atof(argv[4]) : DEFAULT_TX;
    const float TY = (argc == 7) ? atof(argv[5]) : DEFAULT_TY;
    const float ANG_DEG = (argc == 7) ? atof(argv[6]) : DEFAULT_ANG_DEG;
    const float ANG = (ANG_DEG * M_PI) / 180.f;  // radians

    // if (n_col % 32 != 0) {
    //   std::cerr << "Error: n_row and n_col must be multiple of 32" <<
    //   std::endl; return -1;
    // }

    std::cout << "TESTBENCH PARAMETERS:\n";
    std::cout << "n_couples: " << n_couples << std::endl;
    std::cout << "n_row: " << n_row << std::endl;
    std::cout << "n_col: " << n_col << std::endl;
    std::cout << "depth_padding: " << depth_padding << std::endl;
    std::cout << "row_padding: " << row_padding << std::endl;
    std::cout << "col_padding: " << col_padding << std::endl;
    std::cout << "TX: " << TX << std::endl;
    std::cout << "TY: " << TY << std::endl;
    std::cout << "ANG_DEG: " << ANG_DEG << std::endl;
    std::cout << "ANG: " << ANG << std::endl;
    std::cout << "------------------------------\n";

    uint8_t* input_volume = new uint8_t[(n_row + row_padding) * (n_col + col_padding) * (n_couples + depth_padding)];
    uint8_t* output_volume_hw =
        new uint8_t[(n_row + row_padding) * (n_col + col_padding) * (n_couples + depth_padding)];
    uint8_t* output_volume_sw =
        new uint8_t[(n_row + row_padding) * (n_col + col_padding) * (n_couples + depth_padding)];

    read_volume_from_file(
        input_volume, n_row, n_col, n_couples, row_padding, col_padding, depth_padding, SW_FOLDER("dataset/"));

    //
    // ---------- (1) setup aie ----------
    //
    std::printf("-> Running setup_aie\n");
    hls::stream<float> out_setup_aie("out_setup_aie");
    setup_aie(
        TX,
        TY,
        ANG,
        n_couples + depth_padding,
        (float)(n_row + row_padding),
        (float)(n_col + col_padding),
        out_setup_aie);
    write_stream_to_file(out_setup_aie, AIE_FOLDER("data/params_in.txt"), PLIO_32);

    // write fake streams to file
    {
        std::cout << "-> Writing fake streams to file ..." << std::endl;
        hls::stream<ORIGINAL_PIXEL_TYPE> fake_ab[NUM_INPUT_PLIOS];

        for (int i = 0; i < NUM_INPUT_PLIOS; i++) {
            fake_ab[i].write(i);
            for (int j = 0; j < 3; j++) {
                fake_ab[i].write(0);
            }
            for (int j = 0; j < 4; j++) {
                fake_ab[i].write((n_couples >> 8 * j) & 0xFF);  // write n_couples as 4 bytes
            }
            for (int j = 0; j < 64 - 8; j++) {
                fake_ab[i].write(0);
            }

            // second n_couples chunks, replacing mac_bottom
            fake_ab[i].write(i);
            for (int j = 0; j < 64 - 1; j++) {
                fake_ab[i].write(0);
            }

            int twice = (INT_PE <= 64 ? 1 : 2);
            for (int j = 0; j < twice * 2 * 2 * (n_row + row_padding) * (n_col + col_padding) *
                                    (n_couples + depth_padding) / INT_PE;
                 j++) {
                fake_ab[i].write(0);
            }
        }

        for (int i = 0; i < NUM_INPUT_PLIOS; i++) {
            write_stream_to_file(fake_ab[i], AIE_FOLDER("data/p_ab_" + std::to_string(i + 1) + ".txt"), PLIO_128);
        }
    }

    //
    // ---------- (2) AIE ----------
    //
    std::printf("-> Running AIE (indexes & coefficients) . . .\n");
    run_aie();

    // printf("FIRST AIE RUN SKIPPED: READING BACKUP !!!\n");
    // // return 0;

    hls::stream<COORDS_TYPE> out_aie_A("out_aie_A");
    hls::stream<COORDS_TYPE> out_aie_B("out_aie_B");
    hls::stream<COORDS_TYPE> out_aie_C("out_aie_C");
    hls::stream<COORDS_TYPE> out_aie_D("out_aie_D");
    read_stream_from_file<COORD_AXIS_W>(out_aie_A, AIE_FOLDER("x86simulator_output/data/TL_out.txt"));
    read_stream_from_file<COORD_AXIS_W>(out_aie_B, AIE_FOLDER("x86simulator_output/data/TR_out.txt"));
    read_stream_from_file<COORD_AXIS_W>(out_aie_C, AIE_FOLDER("x86simulator_output/data/BL_out.txt"));
    read_stream_from_file<COORD_AXIS_W>(out_aie_D, AIE_FOLDER("x86simulator_output/data/BR_out.txt"));

    //
    // ---------- (3) SETUP MUTUAL INFO ----------
    //
    std::printf("-> Running fetchers\n");
    hls::stream<WIDE_PIXEL_TYPE> out_fetcher_A("out_setup_mutualInfo_A");
    hls::stream<WIDE_PIXEL_TYPE> out_fetcher_B("out_setup_mutualInfo_B");
    hls::stream<WIDE_PIXEL_TYPE> out_fetcher_C("out_setup_mutualInfo_C");
    hls::stream<WIDE_PIXEL_TYPE> out_fetcher_D("out_setup_mutualInfo_D");
    fetcher_A(
        out_aie_A,
        out_fetcher_A,
        (WIDE_PIXEL_TYPE*)input_volume,
        n_couples + depth_padding,
        n_row + row_padding,
        n_col + col_padding);
    fetcher_B(
        out_aie_B,
        out_fetcher_B,
        (WIDE_PIXEL_TYPE*)input_volume,
        n_couples + depth_padding,
        n_row + row_padding,
        n_col + col_padding);
    fetcher_C(
        out_aie_C,
        out_fetcher_C,
        (WIDE_PIXEL_TYPE*)input_volume,
        n_couples + depth_padding,
        n_row + row_padding,
        n_col + col_padding);
    fetcher_D(
        out_aie_D,
        out_fetcher_D,
        (WIDE_PIXEL_TYPE*)input_volume,
        n_couples + depth_padding,
        n_row + row_padding,
        n_col + col_padding);

    // print size of each stream
    std::printf("Size of out_fetcher_A: %ld\n", out_fetcher_A.size());
    std::printf("Size of out_fetcher_B: %ld\n", out_fetcher_B.size());
    std::printf("Size of out_fetcher_C: %ld\n", out_fetcher_C.size());
    std::printf("Size of out_fetcher_D: %ld\n", out_fetcher_D.size());

    // substituting pixel colors
    // int size_fetcher = out_fetcher_A.size();
    // // empty all streams
    // for (int i = 0; i < size_fetcher; i++) {
    //     WIDE_PIXEL_TYPE pixel_A = out_fetcher_A.read();
    //     WIDE_PIXEL_TYPE pixel_B = out_fetcher_B.read();
    //     WIDE_PIXEL_TYPE pixel_C = out_fetcher_C.read();
    //     WIDE_PIXEL_TYPE pixel_D = out_fetcher_D.read();

    //     // substituting colors
    //     for (int j = 0; j < NUM_PIXELS_PER_READ; j++) {
    //         pixel_A.range(8*j + 7, 8*j) = ORIGINAL_PIXEL_TYPE(10);
    //         pixel_B.range(8*j + 7, 8*j) = ORIGINAL_PIXEL_TYPE(110);
    //         pixel_C.range(8*j + 7, 8*j) = ORIGINAL_PIXEL_TYPE(190);
    //         pixel_D.range(8*j + 7, 8*j) = ORIGINAL_PIXEL_TYPE(255);
    //     }

    //     out_fetcher_A.write(pixel_A);
    //     out_fetcher_B.write(pixel_B);
    //     out_fetcher_C.write(pixel_C);
    //     out_fetcher_D.write(pixel_D);
    // }

    // write the four streams in data/test_fetcher_A.txt, B.txt, C.txt, D.txt
    // write_stream_to_file_unpack<WIDE_PIXEL_TYPE, ORIGINAL_PIXEL_TYPE>(out_fetcher_A,
    // AIE_FOLDER("data/test_fetcher_A.txt"), PLIO_128); write_stream_to_file_unpack<WIDE_PIXEL_TYPE,
    // ORIGINAL_PIXEL_TYPE>(out_fetcher_B, AIE_FOLDER("data/test_fetcher_B.txt"), PLIO_128);
    // write_stream_to_file_unpack<WIDE_PIXEL_TYPE, ORIGINAL_PIXEL_TYPE>(out_fetcher_C,
    // AIE_FOLDER("data/test_fetcher_C.txt"), PLIO_128); write_stream_to_file_unpack<WIDE_PIXEL_TYPE,
    // ORIGINAL_PIXEL_TYPE>(out_fetcher_D, AIE_FOLDER("data/test_fetcher_D.txt"), PLIO_128); return 0;

    std::printf("-> Running scheduler_IPE\n");
    hls::stream<ap_uint<INPUT_DATA_BITWIDTH_FETCHER_MIN / 2>> out_scheduler_IPE[NUM_INPUT_PLIOS];
    scheduler_IPE(
        out_fetcher_A,
        out_fetcher_B,
        out_fetcher_C,
        out_fetcher_D,
        n_couples + depth_padding,
        n_row + row_padding,
        n_col + col_padding,
        out_scheduler_IPE);

    // print remaining data in out_fetcher_A, B, C, D
    std::printf("Remaining data in fetcher_A: %ld\n", out_fetcher_A.size());
    std::printf("Remaining data in fetcher_B: %ld\n", out_fetcher_B.size());
    std::printf("Remaining data in fetcher_C: %ld\n", out_fetcher_C.size());
    std::printf("Remaining data in fetcher_D: %ld\n", out_fetcher_D.size());

    // return 0;
    // print size of each stream in out_scheduler_IPE
    for (int i = 0; i < NUM_INPUT_PLIOS; i++) {
        std::printf("Size of out_scheduler_IPE[%d]: %ld\n", i, out_scheduler_IPE[i].size());
    }

    // return 0;

    for (int i = 0; i < NUM_INPUT_PLIOS; i++) {
        write_stream_to_file_unpack<ap_uint<INPUT_DATA_BITWIDTH_FETCHER_MIN / 2>, ORIGINAL_PIXEL_TYPE>(
            out_scheduler_IPE[i], AIE_FOLDER("data/p_ab_" + std::to_string(i + 1) + ".txt"), PLIO_128);
    }

    // return 0;

    //
    // ---------- (5) AIE interpolator ----------
    //
    std::printf("-> Running AIE (interpolator) . . .\n");
    run_aie();
    // se int_pe <= 64, cosi - altrimenti considera 32 stream
    hls::stream<AIE_PIXEL_TYPE>* out_aie_interpolated;

    if (INT_PE <= 64) {
        // hls::stream<AIE_PIXEL_TYPE> out_aie_interpolated[NUM_OUTPUT_PLIOS];
        out_aie_interpolated = new hls::stream<AIE_PIXEL_TYPE>[NUM_OUTPUT_PLIOS];
        for (int i = 0; i < NUM_OUTPUT_PLIOS; i++) {
            read_stream_from_file_pack<ORIGINAL_PIXEL_TYPE, AIE_PIXEL_TYPE>(
                out_aie_interpolated[i],
                AIE_FOLDER("x86simulator_output/data/result_" + std::to_string(i + 1) + ".txt"));
        }
    } else {
        out_aie_interpolated = new hls::stream<AIE_PIXEL_TYPE>[32];
        for (int i = 0; i < 32; i++) {
            read_stream_from_file_pack<ORIGINAL_PIXEL_TYPE, AIE_PIXEL_TYPE>(
                out_aie_interpolated[i],
                AIE_FOLDER("x86simulator_output/data/result_" + std::to_string(i + 1) + ".txt"));
        }
    }

    // // print size of each stream
    // for (int i = 0; i < INT_PE_SATURATED; i++) {
    //     std::printf("Size of out_aie_interpolated[%d]: %ld\n", i, out_aie_interpolated[i].size());
    // }

    // return 0;

    //
    // ---------- (6) writer ----------
    //
    // std::printf("-> Running writer\n");
    // writer(
    //     (WIDE_PIXEL_TYPE*)output_volume_hw, n_couples + depth_padding, n_row + row_padding, n_col + col_padding,
    //     WRITER_TESTBENCH_CALL(out_aie_interpolated));

    //
    // ---------- (6) setup_mi ----------
    std::printf("-> Running setup_mi\n");
    hls::stream<MI_PIXEL_TYPE> out_setup_mi("out_setup_mi");
    setup_mi(
        out_setup_mi,
        (WIDE_PIXEL_TYPE*)output_volume_hw,
        n_couples + depth_padding,
        n_row + row_padding,
        n_col + col_padding,
        SETUP_MI_TESTBENCH_CALL(out_aie_interpolated));

    // // Salva out_setup_mi in numeri da 1 byte su file
    // // write_stream_to_file_unpack<MI_PIXEL_TYPE, ORIGINAL_PIXEL_TYPE>(out_setup_mi, "mi_in.txt" , PLIO_32);

    // //write_stream_to_file_(out_setup_mi, "mi_in.txt", PLIO_32);
    // // ---------- (7) mutual_info ----------
    // //
    // // std::cout << "SIZE: " << out_setup_mi.size() << std::endl;
    // // return 0;
    // printf("Size of stream before: %ld\n", out_setup_mi.size());

    // Salva out_setup_mi in numeri da 1 byte su file
    // write_stream_to_file_unpack<MI_PIXEL_TYPE,
    // ORIGINAL_PIXEL_TYPE>(out_setup_mi, "mi_in.txt" , PLIO_32);

    // write_stream_to_file_(out_setup_mi, "mi_in.txt", PLIO_32);
    //  ---------- (7) mutual_info ----------
    //
    //  std::cout << "SIZE: " << out_setup_mi.size() << std::endl;
    //  return 0;
    //   printf("Size of stream before: %d\n", out_setup_mi.size());

    std::printf("-> Running mutual_info\n");
    printf("Size of stream before: %d\n", out_setup_mi.size());
    float hw_mi = 0;

    unsigned int input_size = (n_couples + depth_padding) * (n_row + row_padding) * (n_col + col_padding);
    mutual_information_master(
        out_setup_mi,
        (MI_PIXEL_TYPE*)input_volume,
        &hw_mi,
        input_size,
        depth_padding * (n_row + row_padding) * (n_col + col_padding));
    printf("Size of stream after: %d\n", out_setup_mi.size());

    //   // write_stream_to_file_unpack<MI_PIXEL_TYPE,
    //   // ORIGINAL_PIXEL_TYPE>(out_setup_mi, "leftover.txt" , PLIO_32);

    // write volume to file
    std::printf("-> Writing volume to file\n");
    create_folder(TEST_FOLDER("dataset_output_new/"));
    write_volume_to_file(
        output_volume_hw,
        n_row,
        n_col,
        n_couples,
        row_padding,
        col_padding,
        depth_padding,
        TEST_FOLDER("dataset_output_new/"));

    std::cout << "--- TESTBENCH HW COMPLETED ---" << std::endl;
    // ###############################################################################################################################

    //
    // ---------- (7) SW transform ----------
    //
    std::printf("-> Running SW transform\n");
    transform_volume(
        input_volume,
        output_volume_sw,
        TX,
        TY,
        ANG,
        n_row + row_padding,
        n_col + col_padding,
        n_couples + depth_padding,
        true);
    std::printf("Writing volume to file ...\n");
    create_folder(TEST_FOLDER("dataset_output_sw/"));
    write_volume_to_file(
        output_volume_sw,
        n_row,
        n_col,
        n_couples,
        row_padding,
        col_padding,
        depth_padding,
        TEST_FOLDER("dataset_output_sw/"));

    printf("-> Running SW mutual_info\n");
    float sw_mi = software_mi(
        n_couples, n_row, n_col, row_padding, col_padding, depth_padding, TX, TY, ANG, SW_FOLDER("dataset/"), nullptr);

    std::cout << "--- TESTBENCH SW COMPLETED ---" << std::endl;

    std::cout << "remaining pixels: " << out_aie_interpolated[7].size() << std::endl;

    float error = std::abs(hw_mi - sw_mi);
    std::cout << "HW MI: " << hw_mi << std::endl;
    std::cout << "SW MI: " << sw_mi << std::endl;
    std::cout << "Error: " << error << std::endl;

    return 0;
}
