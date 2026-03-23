#ifndef RIGID_REGISTRATION_RUNNER_HPP
#define RIGID_REGISTRATION_RUNNER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

void run_rigid_registration_trilli(const std::string &ref_img_path,
                                   const std::string &float_img_path,
                                   const std::string &output_folder,
                                   int n_couples, int n_row, int n_col,
                                   int rangeX = 256, int rangeY = 256,
                                   float rangeAngZ = 1.0f,
                                   const std::string &xclbin_path = "");

std::vector<uint8_t> run_rigid_registration_trilli_from_data(
    std::vector<uint8_t> &ref_volume, std::vector<uint8_t> &float_volume,
    const std::string &output_folder, int n_couples, int n_row, int n_col,
    int rangeX = 256, int rangeY = 256, float rangeAngZ = 1.0f, const std::string &xclbin_path = "");

#endif // RIGID_REGISTRATION_RUNNER_HPP