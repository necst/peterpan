#ifndef IMAGEFUSION_HPP
#define IMAGEFUSION_HPP

#include <future>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../core/domain/fusion_services.hpp"
#include "../infastructure/file_repository.hpp"
#include "../interfaces/image_repository.hpp"

#ifdef HW_REG
#ifdef TRILLI_X
    #include "versal_3dir/Versal3DIR_TRILLIX.hpp"

#else
    #include "versal_3dir/Versal3DIR.hpp"
#endif
#endif

class imagefusion {
   public:
#ifndef HW_REG
    // ===============================================================
    //  SOFTWARE VERSION (EXISTING)
    // ===============================================================
    static void perform_fusion_from_files_3d(
        std::vector<cv::Mat>& reference_image,
        std::vector<cv::Mat>& floating_image,
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
        fuse_images_3d(
            reference_image,
            floating_image,
            ref_vol,
            flt_vol,
            register_strategy,
            fusion_strategy,
            n_couples,
            n_row,
            n_col,
            padding,
            rangeX,
            rangeY,
            rangeAngZ,
            registered_volume);
    }

    // ===============================================================
    //  SOFTWARE VERSION (OVERLOAD con parametri by-ref)
    // ===============================================================
    static double perform_fusion_from_files_3d(
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
        auto start = std::chrono::high_resolution_clock::now();

        fuse_images_3d(
            ref_vol,
            flt_vol,
            register_strategy,
            fusion_strategy,
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

        // placeholder update (fino a quando fuse_images_3d non restituisce i valori reali)
        tx += 0.0f;
        ty += 0.0f;
        theta += 0.0f;

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

#else
    // ===============================================================
    //  HARDWARE VERSION (EXISTING)
    // ===============================================================
    static double perform_fusion_from_files_3d(
        std::vector<cv::Mat>& reference_image,
        std::vector<cv::Mat>& floating_image,
        std::string register_strategy,
        std::string fusion_strategy,
        Versal3DIR& board,
        int rangeX,
        int rangeY,
        float rangeAngZ) {
        double duration_write_flt_sec = 0;
        double duration_write_ref_sec = 0;

        auto start = std::chrono::high_resolution_clock::now();

        board.write_floating_volume(&duration_write_flt_sec);
        board.write_reference_volume(&duration_write_ref_sec);

        fuse_images_3d(
            reference_image, floating_image, register_strategy, fusion_strategy, board, rangeX, rangeY, rangeAngZ);

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

    // ===============================================================
    //  HARDWARE VERSION (OVERLOAD con parametri by-ref)
    // ===============================================================
    static double perform_fusion_from_files_3d(
        std::vector<uint8_t>& reference_image,
        std::vector<uint8_t>& floating_image,
        std::string register_strategy,
        std::string fusion_strategy,
        Versal3DIR& board,
        int rangeX,
        int rangeY,
        float rangeAngZ,
        float& tx,
        float& ty,
        float& theta,
        int num_iterations = 1000000) {
        double duration_write_flt_sec = 0;
        double duration_write_ref_sec = 0;

        std::cout << "Running with ranges [" << rangeX << ", " << rangeY << ", " << rangeAngZ << "] and max iterations "
                  << num_iterations << std::endl;

        board.write_floating_volume(&duration_write_flt_sec);
        board.write_reference_volume(&duration_write_ref_sec);

        auto start = std::chrono::high_resolution_clock::now();

        fuse_images_3d(
            register_strategy, fusion_strategy, board, rangeX, rangeY, rangeAngZ, tx, ty, theta, num_iterations);

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
#endif

    // ===============================================================
    //  UTILITY METHODS (unchanged)
    // ===============================================================
    static std::vector<std::string> fusion_strategies() { return available_fusion_algorithms(); }

    static std::vector<std::string> register_strategies() { return available_registration_algorithms(); }
};

#endif  // IMAGEFUSION_HPP
