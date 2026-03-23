#pragma once

#include <cstdint>
#include <string>
#include <vector>
#ifdef HW_REG
#ifdef TRILLI_X
#include "versal_3dir/Versal3DIR_TRILLIX.hpp"
#else
#include "versal_3dir/Versal3DIR.hpp"
#endif
#endif
struct RegistrationParams {
    float tx, ty, theta;
};

struct RegistrationTime {
    double exec_time = 0;
    double load_time = 0;
};

struct RegistrationResult {
    std::vector<uint8_t> registered_volume;
    RegistrationParams final_params;
    RegistrationTime times;
};

RegistrationResult run_pyramidal_registration_trilli(
    const std::vector<uint8_t>& ref_lvl1,
    const std::vector<uint8_t>& mov_lvl1,
    const std::vector<uint8_t>& ref_lvl2,
    const std::vector<uint8_t>& mov_lvl2,
    const std::vector<uint8_t>& ref_lvl3,
    const std::vector<uint8_t>& mov_lvl3,
    int Dz,
    int n_row,
    int n_col,
    const std::string& xclbin_path);

RegistrationResult run_registration_level(
#ifdef HW_REG
    Versal3DIR& board,
#endif
    std::vector<uint8_t>& ref_volume,
    std::vector<uint8_t>& float_volume,
    const std::string& output_folder,
    int n_couples,
    int n_row,
    int n_col,
    int rangeX,
    int rangeY,
    float rangeAngZ,
    const RegistrationParams& init_params,
    double& level_total_time,
    double& fpga_load_time,
    double& fpga_kernel_time,
    int num_iter);
