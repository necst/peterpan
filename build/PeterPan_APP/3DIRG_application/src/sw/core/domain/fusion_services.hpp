#ifndef DOMAIN_FUSION_HPP
#define DOMAIN_FUSION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../fusion_algorithms.hpp"
#include "../register_algorithms.hpp"

/**
 * @brief available_fusion_algorithms
 * @return available strategies for fusion
 */
inline std::vector<std::string> available_fusion_algorithms() { return fusion_algorithms::available(); }

/**
 * @brief available_registration_algorithms
 * @return available strategies for registration
 */
inline std::vector<std::string> available_registration_algorithms() { return register_algorithms::available(); }

#ifndef HW_REG
// =====================================================================
// SOFTWARE VERSION (ESISTENTE)
// =====================================================================
inline void fuse_images_3d(
    std::vector<cv::Mat>& ref,
    std::vector<cv::Mat>& flt,
    std::vector<uint8_t>& ref_vol,
    std::vector<uint8_t>& flt_vol,
    std::string register_strategy,
    std::string fusion_strategy,
    int n_couples,
    int n_row,
    int n_col,
    int padding,
    int rangeX,
    int rangeY,
    float rangeAngZ,
    uint8_t* registered_volume) {
    std::unique_ptr<fusion> fusion_algorithm = fusion_algorithms::pick(fusion_strategy);
    std::unique_ptr<registration> registration_algorithm = register_algorithms::pick(register_strategy);

    registration_algorithm->register_images_3d(
        ref, flt, ref_vol, flt_vol, n_couples, n_row, n_col, padding, rangeX, rangeY, rangeAngZ, registered_volume);
}

// =====================================================================
// SOFTWARE VERSION (NEW overload with by-reference parameters)
// =====================================================================
inline void fuse_images_3d(
    std::vector<uint8_t>& ref_vol,
    std::vector<uint8_t>& flt_vol,
    std::string register_strategy,
    std::string fusion_strategy,
    int n_couples,
    int n_row,
    int n_col,
    int padding,
    int rangeX,
    int rangeY,
    float rangeAngZ,
    float& tx,
    float& ty,
    float& theta,
    uint8_t* registered_volume) {
    std::unique_ptr<fusion> fusion_algorithm = fusion_algorithms::pick(fusion_strategy);
    std::unique_ptr<registration> registration_algorithm = register_algorithms::pick(register_strategy);

    registration_algorithm->register_images_3d(
        ref_vol,
        flt_vol,
        n_couples,
        n_row,
        n_col,
        padding,
        rangeX,
        rangeY,
        rangeAngZ,
        tx,
        ty,
        theta,
        registered_volume);
}

#else
// =====================================================================
// HARDWARE VERSION (ESISTENTE)
// =====================================================================
inline void fuse_images_3d(
    std::vector<cv::Mat>& ref,
    std::vector<cv::Mat>& flt,
    std::string register_strategy,
    std::string fusion_strategy,
    Versal3DIR& board,
    int rangeX,
    int rangeY,
    float rangeAngZ) {
    using namespace cv;
    std::unique_ptr<fusion> fusion_algorithm = fusion_algorithms::pick(fusion_strategy);
    std::unique_ptr<registration> registration_algorithm = register_algorithms::pick(register_strategy);

    registration_algorithm->register_images_3d(ref, flt, board, rangeX, rangeY, rangeAngZ);
}

// =====================================================================
// HARDWARE VERSION (NEW overload with by-reference parameters)
// =====================================================================
inline void fuse_images_3d(
    std::string register_strategy,
    std::string fusion_strategy,
    Versal3DIR& board,
    int rangeX,
    int rangeY,
    float rangeAngZ,
    float& tx,
    float& ty,
    float& theta,
    int num_iterations = 100000) {
    using namespace cv;
    std::unique_ptr<fusion> fusion_algorithm = fusion_algorithms::pick(fusion_strategy);
    std::unique_ptr<registration> registration_algorithm = register_algorithms::pick(register_strategy);
    registration_algorithm->register_images_3d(board, rangeX, rangeY, rangeAngZ, tx, ty, theta, num_iterations);
}
#endif  // HW_REG

#endif  // DOMAIN_FUSION_HPP
