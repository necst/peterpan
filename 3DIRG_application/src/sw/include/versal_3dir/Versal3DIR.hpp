#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include "../../../common/common.h"
#include "../image_utils/image_utils.hpp"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"

// ------------------------------------------------------
// Kernel argument indexes
// ------------------------------------------------------
#define arg_setup_aie_in_tx 0
#define arg_setup_aie_in_ty 1
#define arg_setup_aie_in_ang 2
#define arg_setup_aie_in_n_couples 3
#define arg_setup_aie_in_n_row 4
#define arg_setup_aie_in_n_col 5

#define arg_fetcher_in_flt_original_ptr 2
#define arg_fetcher_in_n_couples 3
#define arg_fetcher_in_n_row 4
#define arg_fetcher_in_n_col 5

#define arg_scheduler_IPE_in_n_couples 4
#define arg_scheduler_IPE_in_n_row 5
#define arg_scheduler_IPE_in_n_col 6

#define arg_setup_mi_pixel_out 1
#define arg_setup_mi_n_couples 2
#define arg_setup_mi_n_row 3
#define arg_setup_mi_n_col 4

#define arg_mutual_info_reference 1
#define arg_mutual_info_mi 2
#define arg_mutual_info_input_size 3

class Versal3DIR {
   public:
    // Basic config
    xrt::device& device;
    xrt::uuid& xclbin_uuid;
    int n_couples;
    int n_row;
    int n_col;
    int row_padding;
    int col_padding;
    int depth_padding;
    size_t buffer_size;
    size_t curr_size;

    // Pointers to image buffers
    uint8_t* input_ref = nullptr;
    uint8_t* input_flt = nullptr;
    uint8_t* output_flt = nullptr;

    bool owns_ref = false;
    bool owns_flt = false;
    bool owns_out = false;

    // Kernels
    xrt::kernel krnl_setup_aie;
    xrt::kernel krnl_fetcher_A;
    xrt::kernel krnl_fetcher_B;
    xrt::kernel krnl_fetcher_C;
    xrt::kernel krnl_fetcher_D;
    xrt::kernel krnl_scheduler_IPE;
    xrt::kernel krnl_setup_mi;
    xrt::kernel krnl_mutual_info;

    // Banks
    xrtMemoryGroup bank_fetcher_A_flt_in;
    xrtMemoryGroup bank_fetcher_B_flt_in;
    xrtMemoryGroup bank_fetcher_C_flt_in;
    xrtMemoryGroup bank_fetcher_D_flt_in;
    xrtMemoryGroup bank_setup_mi;
    xrtMemoryGroup bank_mutual_info;
    xrtMemoryGroup bank_mutual_info_output;

    // Buffers
    xrt::bo buffer_fetcher_A_flt_in;
    xrt::bo buffer_fetcher_B_flt_in;
    xrt::bo buffer_fetcher_C_flt_in;
    xrt::bo buffer_fetcher_D_flt_in;
    xrt::bo buffer_mutual_info_reference;
    xrt::bo buffer_mutual_info_output;
    xrt::bo buffer_setup_mi_flt_transformed;

    // Runners
    xrt::run run_setup_aie;
    xrt::run run_fetcher_A;
    xrt::run run_fetcher_B;
    xrt::run run_fetcher_C;
    xrt::run run_fetcher_D;
    xrt::run run_scheduler_IPE;
    xrt::run run_setup_mi;
    xrt::run run_mutual_info;

    // Constructor
    Versal3DIR(xrt::device& device, xrt::uuid& xclbin_uuid, int n_couples, int n_row, int n_col);

    // Load from FILES (allocates memory → class OWNS buffers)
    int read_volumes_from_file(
        const std::string& path_ref, const std::string& path_flt, const ImageFormat imageFormat = ImageFormat::PNG);

    // Load from DATA (does NOT allocate → class does NOT own)
    int load_volumes_from_data(const std::vector<uint8_t>& ref_volume, const std::vector<uint8_t>& float_volume);

    // Transform parameters
    void set_transform_params(float TX, float TY, float ANG);

    // Data transfer
    void write_floating_volume(double* duration = nullptr);
    void write_reference_volume(double* duration = nullptr);

    // Kernel execution
    void run(double* duration = nullptr);

    // Readback
    void read_flt_transformed(double* duration = nullptr);
    float read_mutual_information();

    // One-shot execution
    float hw_exec(float TX, float TY, float ANG, double* duration_exec = nullptr);
    float hw_exec_tx(float TX, float TY, float ANG, double* duration_exec = nullptr, bool save = false);
    void zero_buffer();

    void update_dimensions(int new_n_couples, int new_n_row, int new_n_col);
    void rebuild_runners();
    ~Versal3DIR();
};
