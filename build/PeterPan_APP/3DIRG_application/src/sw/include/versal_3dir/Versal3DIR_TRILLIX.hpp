#pragma once
/*
 * Versal3DIR_TRILLIX.hpp
 *
 * Ad-hoc header for the Versal3DIR class used in TRILLI_X / TRILLIX
 * (version that uses fixed DIMENSION in the bitstream and accepts only n_couples).
 *
 * NOTE:
 *  - This class is intended for HW (XRT). Compile it only with HW_REG.
 *  - It also exposes load_volumes_from_data(...) to use pyramid volumes
 *    passed by the host (NO file read).
 */

#ifdef HW_REG

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"
#include "xrt/xrt_device.h"

// common.h must define at least: DIMENSION, NUM_PIXELS_PER_READ, NUM_OUTPUT_PLIOS
#include "../../../../../common/common.h"

// required for ImageFormat and read_volume_from_file(...)
#include "../image_utils/image_utils.hpp"

#define NUM_INPUT_PLIOS 128
#define NUM_OUTPUT_PLIOS 64

// setup_aie
#define arg_setup_aie_in_tx 0
#define arg_setup_aie_in_ty 1
#define arg_setup_aie_in_ang 2
#define arg_setup_aie_in_n_couples 3

// fetcher
#define arg_fetcher_in_flt_original_ptr 2
#define arg_fetcher_in_n_couples 3

// scheduler_IPE
#define arg_scheduler_IPE_in_n_couples 4

// setup_mi
#define arg_setup_mi_pixel_out (NUM_OUTPUT_PLIOS + 1)
#define arg_setup_mi_n_couples (NUM_OUTPUT_PLIOS + 2)

// mutual info
#define arg_mutual_info_reference 1
#define arg_mutual_info_mi 2
#define arg_mutual_info_n_couples 3
#define arg_mutual_info_paddin 4

// buffer sizes (bitstream fixed DIMENSION x DIMENSION)
#define FLOAT_INPUT_BUFFER_SIZE (DIMENSION * DIMENSION * sizeof(uint8_t))
#define MINFO_INPUT_BUFFER_SIZE FLOAT_INPUT_BUFFER_SIZE
#define MINFO_OUTPUT_BUFFER_SIZE (sizeof(float))

class Versal3DIR {
   public:
    // --- XRT context ---
    xrt::device& device;
    xrt::uuid& xclbin_uuid;

    // --- geometry (TRILLIX: XY fixed in bitstream, only Z is managed here) ---
    int n_couples = 0;
    int padding = 0;
    size_t buffer_size = 0;  // BYTES: FLOAT_INPUT_BUFFER_SIZE * (n_couples + padding)

    // --- host buffers ---
    uint8_t* input_ref = nullptr;
    uint8_t* input_flt = nullptr;
    uint8_t* output_flt = nullptr;
    bool owns_ref = false;
    bool owns_flt = false;
    bool owns_out = false;
    // --- kernels ---
    xrt::kernel krnl_setup_aie;
    xrt::kernel krnl_fetcher_A;
    xrt::kernel krnl_fetcher_B;
    xrt::kernel krnl_fetcher_C;
    xrt::kernel krnl_fetcher_D;
    xrt::kernel krnl_scheduler_IPE;
    xrt::kernel krnl_setup_mi;
    xrt::kernel krnl_mutual_info;

    // --- bank groups ---
    xrtMemoryGroup bank_fetcher_A_flt_in;
    xrtMemoryGroup bank_fetcher_B_flt_in;
    xrtMemoryGroup bank_fetcher_C_flt_in;
    xrtMemoryGroup bank_fetcher_D_flt_in;
    xrtMemoryGroup bank_setup_mi;
    xrtMemoryGroup bank_mutual_info;
    xrtMemoryGroup bank_mutual_info_output;

    // --- device buffers ---
    xrt::bo buffer_fetcher_A_flt_in;
    xrt::bo buffer_fetcher_B_flt_in;
    xrt::bo buffer_fetcher_C_flt_in;
    xrt::bo buffer_fetcher_D_flt_in;
    xrt::bo buffer_mutual_info_reference;
    xrt::bo buffer_mutual_info_output;
    xrt::bo buffer_setup_mi_flt_transformed;

    // --- runs ---
    xrt::run run_setup_aie;
    xrt::run run_fetcher_A;
    xrt::run run_fetcher_B;
    xrt::run run_fetcher_C;
    xrt::run run_fetcher_D;
    xrt::run run_scheduler_IPE;
    xrt::run run_setup_mi;
    xrt::run run_mutual_info;

   public:
    // TRILLIX constructor: XY dimensions implicit in the bitstream (DIMENSION)
    Versal3DIR(xrt::device& device, xrt::uuid& xclbin_uuid, int n_couples);

    // --- Volume I/O ---
    // Reads from file (as in your original TRILLIX code)
    int read_volumes_from_file(
        const std::string& path_ref, const std::string& path_flt, const ImageFormat imageFormat = ImageFormat::PNG);

    // Load from vectors (pyramidal usage: NO file I/O)
    int load_volumes_from_data(const std::vector<uint8_t>& ref_volume, const std::vector<uint8_t>& float_volume);

    // --- execution ---
    void set_transform_params(float TX, float TY, float ANG);
    void write_floating_volume(double* duration = nullptr);
    void write_reference_volume(double* duration = nullptr);
    void run(double* duration = nullptr);

    void read_flt_transformed(double* duration = nullptr);
    float read_mutual_information();

    float hw_exec(float TX, float TY, float ANG, double* duration_exec = nullptr);
    float hw_exec_tx(float TX, float TY, float ANG, double* duration_exec = nullptr, bool save = false);

    ~Versal3DIR();
};

#endif  // HW_REG
