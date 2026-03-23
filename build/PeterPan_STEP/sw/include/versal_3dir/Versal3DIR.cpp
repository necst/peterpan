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

#include "../../../common/common.h"
#include "../image_utils/image_utils.hpp"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"

// args indexes for setup_aie kernel
#define arg_setup_aie_in_tx 0
#define arg_setup_aie_in_ty 1
#define arg_setup_aie_in_ang 2
#define arg_setup_aie_in_n_couples 3
#define arg_setup_aie_in_n_row 4
#define arg_setup_aie_in_n_col 5

// args indexes for fetcher kernel
#define arg_fetcher_in_flt_original_ptr 2
#define arg_fetcher_in_n_couples 3
#define arg_fetcher_in_n_row 4
#define arg_fetcher_in_n_col 5

// args indexes for scheduler_IPE kernel
#define arg_scheduler_IPE_in_n_couples 4
#define arg_scheduler_IPE_in_n_row 5
#define arg_scheduler_IPE_in_n_col 6

// args indexes for setup_mi
#define arg_setup_mi_pixel_out 1
#define arg_setup_mi_n_couples 2
#define arg_setup_mi_n_row 3
#define arg_setup_mi_n_col 4

// args mutual info
#define arg_mutual_info_reference 1
#define arg_mutual_info_mi 2
#define arg_mutual_info_input_size 3

class Versal3DIR {
   public:
    xrt::device& device;
    xrt::uuid& xclbin_uuid;
    int n_couples;
    int n_row;
    int n_col;
    int row_padding;
    int col_padding;
    int depth_padding;
    size_t buffer_size;

    uint8_t* input_ref = NULL;
    uint8_t* input_flt = NULL;
    uint8_t* output_flt = NULL;

    xrt::kernel krnl_setup_aie;
    xrt::kernel krnl_fetcher_A;
    xrt::kernel krnl_fetcher_B;
    xrt::kernel krnl_fetcher_C;
    xrt::kernel krnl_fetcher_D;
    xrt::kernel krnl_scheduler_IPE;
    xrt::kernel krnl_setup_mi;
    xrt::kernel krnl_mutual_info;

    xrtMemoryGroup bank_fetcher_A_flt_in;
    xrtMemoryGroup bank_fetcher_B_flt_in;
    xrtMemoryGroup bank_fetcher_C_flt_in;
    xrtMemoryGroup bank_fetcher_D_flt_in;
    xrtMemoryGroup bank_setup_mi;
    xrtMemoryGroup bank_mutual_info;
    xrtMemoryGroup bank_mutual_info_output;

    xrt::bo buffer_fetcher_A_flt_in;
    xrt::bo buffer_fetcher_B_flt_in;
    xrt::bo buffer_fetcher_C_flt_in;
    xrt::bo buffer_fetcher_D_flt_in;
    xrt::bo buffer_mutual_info_reference;
    xrt::bo buffer_mutual_info_output;
    xrt::bo buffer_setup_mi_flt_transformed;

    xrt::run run_setup_aie;
    xrt::run run_fetcher_A;
    xrt::run run_fetcher_B;
    xrt::run run_fetcher_C;
    xrt::run run_fetcher_D;
    xrt::run run_scheduler_IPE;
    xrt::run run_setup_mi;
    xrt::run run_mutual_info;

    //
    // Initialize the board configuring it for a specific volume-depth
    //
    Versal3DIR(xrt::device& device, xrt::uuid& xclbin_uuid, int n_couples, int n_row, int n_col)
        : device(device),
          xclbin_uuid(xclbin_uuid),
          n_couples(n_couples),
          n_row(n_row),
          n_col(n_col),
          row_padding((NUM_PIXELS_PER_READ - (n_row % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ),
          col_padding((NUM_PIXELS_PER_READ - (n_col % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ),
          depth_padding((NUM_PIXELS_PER_READ - (n_couples % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ),
          buffer_size((n_row + row_padding) * (n_col + col_padding) * (n_couples + depth_padding)) {
        // create kernel objects
        std::cout << "Creating kernels..." << std::endl;
        krnl_setup_aie = xrt::kernel(device, xclbin_uuid, "setup_aie");
        krnl_fetcher_A = xrt::kernel(device, xclbin_uuid, "fetcher_A");
        krnl_fetcher_B = xrt::kernel(device, xclbin_uuid, "fetcher_B");
        krnl_fetcher_C = xrt::kernel(device, xclbin_uuid, "fetcher_C");
        krnl_fetcher_D = xrt::kernel(device, xclbin_uuid, "fetcher_D");
        krnl_scheduler_IPE = xrt::kernel(device, xclbin_uuid, "scheduler_IPE");
        krnl_setup_mi = xrt::kernel(device, xclbin_uuid, "setup_mi");
        krnl_mutual_info = xrt::kernel(device, xclbin_uuid, "mutual_information_master");

        std::cout << "Kernels created" << std::endl;
        // create memory banks for kernels
        std::cout << "Creating memory banks..." << std::endl;
        // get memory bank groups for device buffer
        bank_fetcher_A_flt_in = krnl_fetcher_A.group_id(arg_fetcher_in_flt_original_ptr);
        bank_fetcher_B_flt_in = krnl_fetcher_B.group_id(arg_fetcher_in_flt_original_ptr);
        bank_fetcher_C_flt_in = krnl_fetcher_C.group_id(arg_fetcher_in_flt_original_ptr);
        bank_fetcher_D_flt_in = krnl_fetcher_D.group_id(arg_fetcher_in_flt_original_ptr);
        bank_setup_mi = krnl_setup_mi.group_id(arg_setup_mi_pixel_out);
        bank_mutual_info = krnl_mutual_info.group_id(arg_mutual_info_reference);
        bank_mutual_info_output = krnl_mutual_info.group_id(arg_mutual_info_mi);
        std::cout << "Memory banks created" << std::endl;
        // bank_suppmi_coord_out =
        // krnl_suppmi.group_id(arg_support_mi_out_coord_ptr); bank_suppmi_coeff_out
        // = krnl_suppmi.group_id(arg_support_mi_out_coeff_ptr);

        // create device buffers
        std::cout << "Creating device buffers with buffer size: " << buffer_size << std::endl;
        buffer_fetcher_A_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_A_flt_in);
        buffer_fetcher_B_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_B_flt_in);
        buffer_fetcher_C_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_C_flt_in);
        buffer_fetcher_D_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_D_flt_in);
        buffer_setup_mi_flt_transformed = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_setup_mi);
        buffer_mutual_info_reference = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_mutual_info);
        buffer_mutual_info_output = xrt::bo(device, sizeof(float), xrt::bo::flags::normal, bank_mutual_info_output);
        std::cout << "Device buffers created" << std::endl;
        std::cout << "Buffer size: " << buffer_size << std::endl;
        std::cout << "create runner instances..." << std::endl;
        // create kernel runner instances
        run_setup_aie = xrt::run(krnl_setup_aie);
        run_fetcher_A = xrt::run(krnl_fetcher_A);
        run_fetcher_B = xrt::run(krnl_fetcher_B);
        run_fetcher_C = xrt::run(krnl_fetcher_C);
        run_fetcher_D = xrt::run(krnl_fetcher_D);
        run_setup_mi = xrt::run(krnl_setup_mi);
        run_scheduler_IPE = xrt::run(krnl_scheduler_IPE);
        run_mutual_info = xrt::run(krnl_mutual_info);

        // run_suppmi = xrt::run(krnl_suppmi);
        // run_mover_T1B = xrt::run(krnl_mover_T1B);
        std::cout << "Run created" << std::endl;
        // set setup_setminfo kernel arguments
        run_fetcher_A.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_A_flt_in);
        run_fetcher_A.set_arg(arg_fetcher_in_n_couples, n_couples + depth_padding);
        run_fetcher_A.set_arg(arg_fetcher_in_n_row, n_row + row_padding);
        run_fetcher_A.set_arg(arg_fetcher_in_n_col, n_col + col_padding);
        run_fetcher_B.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_B_flt_in);
        run_fetcher_B.set_arg(arg_fetcher_in_n_couples, n_couples + depth_padding);
        run_fetcher_B.set_arg(arg_fetcher_in_n_row, n_row + row_padding);
        run_fetcher_B.set_arg(arg_fetcher_in_n_col, n_col + col_padding);
        run_fetcher_C.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_C_flt_in);
        run_fetcher_C.set_arg(arg_fetcher_in_n_couples, n_couples + depth_padding);
        run_fetcher_C.set_arg(arg_fetcher_in_n_row, n_row + row_padding);
        run_fetcher_C.set_arg(arg_fetcher_in_n_col, n_col + col_padding);
        run_fetcher_D.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_D_flt_in);
        run_fetcher_D.set_arg(arg_fetcher_in_n_couples, n_couples + depth_padding);
        run_fetcher_D.set_arg(arg_fetcher_in_n_row, n_row + row_padding);
        run_fetcher_D.set_arg(arg_fetcher_in_n_col, n_col + col_padding);

        // set scheduler_IPE kernel arguments
        run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_couples, n_couples + depth_padding);
        run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_row, n_row + row_padding);
        run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_col, n_col + col_padding);

        // set setup mi kernel arguments
        run_setup_mi.set_arg(arg_setup_mi_pixel_out, buffer_setup_mi_flt_transformed);
        run_setup_mi.set_arg(arg_setup_mi_n_couples, n_couples + depth_padding);
        run_setup_mi.set_arg(arg_setup_mi_n_row, n_row + row_padding);
        run_setup_mi.set_arg(arg_setup_mi_n_col, n_col + col_padding);

        // set mutual info kernel arguments
        run_mutual_info.set_arg(arg_mutual_info_reference, buffer_mutual_info_reference);
        run_mutual_info.set_arg(arg_mutual_info_mi, buffer_mutual_info_output);
        run_mutual_info.set_arg(arg_mutual_info_input_size, (unsigned int)buffer_size);
    }

    //
    // Read volumes from file
    //
    int read_volumes_from_file(
        const std::string& path_ref, const std::string& path_flt, const ImageFormat imageFormat = ImageFormat::PNG) {
        input_flt = new uint8_t[buffer_size];
        output_flt = new uint8_t[buffer_size];
        input_ref = new uint8_t[buffer_size];
        if (read_volume_from_file(
                input_flt, n_row, n_col, n_couples, row_padding, col_padding, depth_padding, path_flt, imageFormat) ==
            -1) {
            std::cerr << "Error: Could not open floating volume. Some file in path \"" << path_flt
                      << "\" might not exist" << std::endl;
            return -1;
        }
        if (read_volume_from_file(
                input_ref, n_row, n_col, n_couples, row_padding, col_padding, depth_padding, path_ref, imageFormat) ==
            -1) {
            std::cerr << "Error: Could not open reference volume. Some file in path \"" << path_ref
                      << "\" might not exist" << std::endl;
            return -1;
        }

        return 0;
    }

    //
    // Set the transformation parameters
    //
    void set_transform_params(float TX, float TY, float ANG) {
        // set setup_aie kernel arguments
        run_setup_aie.set_arg(arg_setup_aie_in_tx, TX);
        run_setup_aie.set_arg(arg_setup_aie_in_ty, TY);
        run_setup_aie.set_arg(arg_setup_aie_in_ang, ANG);
        run_setup_aie.set_arg(arg_setup_aie_in_n_couples, n_couples + depth_padding);
        run_setup_aie.set_arg(arg_setup_aie_in_n_row, n_row + row_padding);
        run_setup_aie.set_arg(arg_setup_aie_in_n_col, n_col + col_padding);
    }

    //
    // Transform the floating volume to the board
    //
    void write_floating_volume(double* duration = NULL) {
        Timer timer_transfer_flt_write;
        if (duration != NULL) timer_transfer_flt_write.start();
        buffer_fetcher_A_flt_in.write(input_flt);
        buffer_fetcher_A_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        buffer_fetcher_B_flt_in.write(input_flt);
        buffer_fetcher_B_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        buffer_fetcher_C_flt_in.write(input_flt);
        buffer_fetcher_C_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        buffer_fetcher_D_flt_in.write(input_flt);
        buffer_fetcher_D_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (duration != NULL) *duration += timer_transfer_flt_write.getElapsedSeconds();
    }

    //
    // Transform the reference volume to the board
    //
    void write_reference_volume(double* duration = NULL) {
        // std::printf("no reference volume needs to be passed to the board\n");
        Timer timer_transfer_ref_write;
        if (duration != NULL) timer_transfer_ref_write.start();
        buffer_mutual_info_reference.write(input_ref);
        buffer_mutual_info_reference.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (duration != NULL) *duration += 0;  // timer_transfer_ref_write.getElapsedSeconds();
    }

    //
    // Run the kernels
    //
    void run(double* duration = NULL) {
        // run the pl kernels
        Timer timer_execution;
        if (duration != NULL) timer_execution.start();
        run_setup_aie.start();
        run_fetcher_A.start();
        run_fetcher_B.start();
        run_fetcher_C.start();
        run_fetcher_D.start();
        run_scheduler_IPE.start();
        run_setup_mi.start();
        run_mutual_info.start();

        run_setup_aie.wait();
        run_fetcher_A.wait();
        run_fetcher_B.wait();
        run_fetcher_C.wait();
        run_fetcher_D.wait();
        run_scheduler_IPE.wait();
        run_setup_mi.wait();
        run_mutual_info.wait();

        if (duration != NULL) *duration += timer_execution.getElapsedSeconds();
    }

    //
    // Read the transformed floating volume from the board
    //
    void read_flt_transformed(double* duration = NULL) {
        Timer timer_transfer_read_flt;
        if (duration != NULL) timer_transfer_read_flt.start();
        buffer_setup_mi_flt_transformed.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        buffer_setup_mi_flt_transformed.read(output_flt);
        if (duration != NULL) *duration += timer_transfer_read_flt.getElapsedSeconds();
    }

    //
    // Read the mutual information from the board
    //
    float read_mutual_information() {
        // std::printf("no mutual information needs to be read from the board\n");
        float output_data = -1;
        buffer_mutual_info_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        buffer_mutual_info_output.read(&output_data);
        return output_data;
    }

    ~Versal3DIR() {
        delete[] input_ref;
        delete[] input_flt;
        delete[] output_flt;
    }
};
