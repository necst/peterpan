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

// args indexes for writer kernel
#define arg_writer_out_interpolated_ptr 0
#define arg_writer_in_n_couples 1
#define arg_writer_in_n_row 2
#define arg_writer_in_n_col 3

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
    xrt::kernel krnl_writer;

    xrtMemoryGroup bank_fetcher_A_flt_in;
    xrtMemoryGroup bank_fetcher_B_flt_in;
    xrtMemoryGroup bank_fetcher_C_flt_in;
    xrtMemoryGroup bank_fetcher_D_flt_in;
    xrtMemoryGroup bank_writer_flt_transformed;

    xrt::bo buffer_fetcher_A_flt_in;
    xrt::bo buffer_fetcher_B_flt_in;
    xrt::bo buffer_fetcher_C_flt_in;
    xrt::bo buffer_fetcher_D_flt_in;
    xrt::bo buffer_writer_flt_transformed;

    xrt::run run_setup_aie;
    xrt::run run_fetcher_A;
    xrt::run run_fetcher_B;
    xrt::run run_fetcher_C;
    xrt::run run_fetcher_D;
    xrt::run run_scheduler_IPE;
    xrt::run run_writer;

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
        krnl_setup_aie = xrt::kernel(device, xclbin_uuid, "setup_aie");
        krnl_fetcher_A = xrt::kernel(device, xclbin_uuid, "fetcher_A");
        krnl_fetcher_B = xrt::kernel(device, xclbin_uuid, "fetcher_B");
        krnl_fetcher_C = xrt::kernel(device, xclbin_uuid, "fetcher_C");
        krnl_fetcher_D = xrt::kernel(device, xclbin_uuid, "fetcher_D");
        krnl_scheduler_IPE = xrt::kernel(device, xclbin_uuid, "scheduler_IPE");
        krnl_writer = xrt::kernel(device, xclbin_uuid, "writer");

        std::cout << "Kernels created" << std::endl;
        // create memory banks for kernels
        std::cout << "Creating memory banks..." << std::endl;
        // get memory bank groups for device buffer
        bank_fetcher_A_flt_in = krnl_fetcher_A.group_id(arg_fetcher_in_flt_original_ptr);
        bank_fetcher_B_flt_in = krnl_fetcher_B.group_id(arg_fetcher_in_flt_original_ptr);
        bank_fetcher_C_flt_in = krnl_fetcher_C.group_id(arg_fetcher_in_flt_original_ptr);
        bank_fetcher_D_flt_in = krnl_fetcher_D.group_id(arg_fetcher_in_flt_original_ptr);
        bank_writer_flt_transformed = krnl_writer.group_id(arg_writer_out_interpolated_ptr);

        // create device buffers
        buffer_fetcher_A_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_A_flt_in);
        buffer_fetcher_B_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_B_flt_in);
        buffer_fetcher_C_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_C_flt_in);
        buffer_fetcher_D_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_D_flt_in);
        buffer_writer_flt_transformed =
            xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_writer_flt_transformed);

        // create kernel runner instances
        run_setup_aie = xrt::run(krnl_setup_aie);
        run_fetcher_A = xrt::run(krnl_fetcher_A);
        run_fetcher_B = xrt::run(krnl_fetcher_B);
        run_fetcher_C = xrt::run(krnl_fetcher_C);
        run_fetcher_D = xrt::run(krnl_fetcher_D);
        run_scheduler_IPE = xrt::run(krnl_scheduler_IPE);
        run_writer = xrt::run(krnl_writer);

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

        // set writer kernel arguments
        run_writer.set_arg(arg_writer_out_interpolated_ptr, buffer_writer_flt_transformed);
        run_writer.set_arg(arg_writer_in_n_couples, n_couples + depth_padding);
        run_writer.set_arg(arg_writer_in_n_row, n_row + row_padding);
        run_writer.set_arg(arg_writer_in_n_col, n_col + col_padding);
    }

    //
    // Read volumes from file
    //
    int read_volumes_from_file(
        const std::string& path_ref, const std::string& path_flt, const ImageFormat imageFormat = ImageFormat::PNG) {
        input_flt = new uint8_t[buffer_size];
        output_flt = new uint8_t[buffer_size];

        if (read_volume_from_file(
                input_flt, n_row, n_col, n_couples, row_padding, col_padding, depth_padding, path_flt, imageFormat) ==
            -1) {
            std::cerr << "Error: Could not open floating volume. Some file in path \"" << path_flt
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
        std::printf("no reference volume needs to be passed to the board\n");
        // Timer timer_transfer_ref_write;
        // if (duration != NULL) timer_transfer_ref_write.start();
        // buffer_minfo_ref.write(input_ref);
        // buffer_minfo_ref.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
        run_writer.start();

        // waiting for kernels to finish
        run_setup_aie.wait();
        run_fetcher_A.wait();
        run_fetcher_B.wait();
        run_fetcher_C.wait();
        run_fetcher_D.wait();
        run_scheduler_IPE.wait();
        run_writer.wait();

        if (duration != NULL) *duration += timer_execution.getElapsedSeconds();
    }

    //
    // Read the transformed floating volume from the board
    //
    void read_flt_transformed(double* duration = NULL) {
        Timer timer_transfer_read_flt;
        if (duration != NULL) timer_transfer_read_flt.start();
        buffer_writer_flt_transformed.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        buffer_writer_flt_transformed.read(output_flt);
        if (duration != NULL) *duration += timer_transfer_read_flt.getElapsedSeconds();
    }

    //
    // Read the mutual information from the board
    //
    float read_mutual_information() {
        std::printf("no mutual information needs to be read from the board\n");
        // float output_data;
        // buffer_minfo_rlst.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        // buffer_minfo_rlst.read(&output_data);
        // return output_data;
        return -1;
    }

    ~Versal3DIR() {
        delete[] input_ref;
        delete[] input_flt;
        delete[] output_flt;
    }
};