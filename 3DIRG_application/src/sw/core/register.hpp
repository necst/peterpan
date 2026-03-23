/******************************************
 * MIT License
 *
 * Copyright (c) 2025 Giuseppe Sorrentino,
 * Paolo Salvatore Galfano, Davide Conficconi,
 * Eleonora D'Arnese
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/***************************************************************
 * registration class of the whole app
 * credits goes also to the author of this repo:
 * https://github.com/mariusherzog/ImageRegistration
 ****************************************************************/
#ifndef REGISTER_HPP
#define REGISTER_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../include/image_utils/image_utils.hpp"
#ifdef HW_REG
#ifdef TRILLI_X
#include "../include/versal_3dir/Versal3DIR_TRILLIX.hpp"
#else
#include "../include/versal_3dir/Versal3DIR.hpp"
#endif
#endif
#include "../include/software_mi/software_mi.cpp"
#include "optimize.hpp"

/**
 * @brief The registration interface defines the signatures of a registration
 *        operation of a floating image to a reference image.
 */
class registration {
   public:
#ifndef HW_REG
    virtual void register_images_3d(
        std::vector<cv::Mat>& ref,
        std::vector<cv::Mat>& flt,
        std::vector<uint8_t>& ref_vol,
        std::vector<uint8_t>& flt_vol,
        int n_couples,
        int n_row,
        int n_col,
        int padding,
        int rangeX,
        int rangeY,
        float AngZ,
        uint8_t* registered_volume) = 0;

    // 🔹 Overload for pyramidal usage
    virtual void register_images_3d(
        std::vector<uint8_t>& ref_vol,
        std::vector<uint8_t>& flt_vol,
        int n_couples,
        int n_row,
        int n_col,
        int padding,
        int rangeX,
        int rangeY,
        float AngZ,
        float& tx,
        float& ty,
        float& theta,
        uint8_t* registered_volume) = 0;
#else
    virtual void register_images_3d(
        std::vector<cv::Mat>& ref,
        std::vector<cv::Mat>& flt,
        Versal3DIR& board,
        int rangeX,
        int rangeY,
        float rangeAngZ) = 0;

    // 🔹 Overload for pyramidal usage
    virtual void register_images_3d(
        Versal3DIR& board,
        int rangeX,
        int rangeY,
        float rangeAngZ,
        float& tx,
        float& ty,
        float& theta,
        int num_iterations = 100000) = 0;
#endif
    virtual ~registration() = 0;
};

registration::~registration() {}

/**
 * @brief The mutual information strategy uses mutual information as a
 *        similarity metric for registration.
 */
class mutualinformation : public registration {
   public:
    static inline std::size_t counter;  // C++17

#ifndef HW_REG
    // ===========================================================
    // STANDARD SOFTWARE VERSION (as before)
    // ===========================================================
    void register_images_3d(
        std::vector<cv::Mat>& ref,
        std::vector<cv::Mat>& flt,
        std::vector<uint8_t>& ref_vol,
        std::vector<uint8_t>& flt_vol,
        int n_couples,
        int n_row,
        int n_col,
        int padding,
        int rangeX,
        int rangeY,
        float AngZ,
        uint8_t* registered_volume) override {
        double tx, ty, a11, a12, a21, a22;
        double avg_tx = 0.0, avg_ty = 0.0;
        double avg_a11 = 0.0, avg_a12 = 0.0, avg_a21 = 0.0, avg_a22 = 0.0;

        for (int i = 0; i < ref.size(); i++) {
            estimate_initial(ref[i], flt[i], tx, ty, a11, a12, a21, a22);
            avg_tx += tx;
            avg_ty += ty;
            avg_a11 += a11;
            avg_a12 += a12;
            avg_a21 += a21;
            avg_a22 += a22;
        }

        avg_tx /= ref.size();
        avg_ty /= ref.size();
        avg_a11 /= ref.size();
        avg_a21 /= ref.size();
        float ang_rad = atan2(avg_a21, avg_a11);

        if (std::isnan(avg_tx)) avg_tx = 0.0;
        if (std::isnan(avg_ty)) avg_ty = 0.0;
        if (std::isnan(ang_rad)) ang_rad = 0.0;

        std::vector<double> init{avg_tx, avg_ty, ang_rad};
        std::vector<double> rng{(double)rangeX, (double)rangeY, (double)AngZ};
        std::pair<std::vector<double>::iterator, std::vector<double>::iterator> o{init.begin(), init.end()};

        uint8_t* buffer_ref = new uint8_t[n_row * n_col * (n_couples + padding)];
        uint8_t* buffer_flt = new uint8_t[n_row * n_col * (n_couples + padding)];
        std::memcpy(buffer_ref, ref_vol.data(), ref_vol.size());
        std::memcpy(buffer_flt, flt_vol.data(), flt_vol.size());

        optimize_powell(
            o,
            {rng.begin(), rng.end()},
            std::bind(
                cost_function_3d, buffer_ref, buffer_flt, n_couples, n_row, n_col, padding, std::placeholders::_1));

        tx = init[0];
        ty = init[1];
        ang_rad = init[2];
        sw_registration_step_3d(
            buffer_ref, buffer_flt, registered_volume, n_couples, n_row, n_col, padding, tx, ty, ang_rad);

        delete[] buffer_ref;
        delete[] buffer_flt;
    }

    // ===========================================================
    // VERSIONE SOFTWARE PIRAMIDALE
    // ===========================================================
    void register_images_3d(
        std::vector<uint8_t>& ref_vol,
        std::vector<uint8_t>& flt_vol,
        int n_couples,
        int n_row,
        int n_col,
        int padding,
        int rangeX,
        int rangeY,
        float AngZ,
        float& tx,
        float& ty,
        float& theta,
        uint8_t* registered_volume) override {
        std::cout << "[SW] Starting registration with init tx=" << tx << ", ty=" << ty << ", th=" << theta << "\n";

        std::vector<double> init{tx, ty, theta};
        std::vector<double> rng{(double)rangeX, (double)rangeY, (double)AngZ};
        std::pair<std::vector<double>::iterator, std::vector<double>::iterator> o{init.begin(), init.end()};

        uint8_t* buffer_ref = new uint8_t[n_row * n_col * (n_couples + padding)];
        uint8_t* buffer_flt = new uint8_t[n_row * n_col * (n_couples + padding)];
        std::memcpy(buffer_ref, ref_vol.data(), ref_vol.size());
        std::memcpy(buffer_flt, flt_vol.data(), flt_vol.size());

        optimize_powell(
            o,
            {rng.begin(), rng.end()},
            std::bind(
                cost_function_3d, buffer_ref, buffer_flt, n_couples, n_row, n_col, padding, std::placeholders::_1));

        tx = static_cast<float>(init[0]);
        ty = static_cast<float>(init[1]);
        theta = static_cast<float>(init[2]);

        std::cout << "[SW] Updated params → tx=" << tx << ", ty=" << ty << ", th=" << theta << "\n";

        sw_registration_step_3d(
            buffer_ref, buffer_flt, registered_volume, n_couples, n_row, n_col, padding, tx, ty, theta);

        delete[] buffer_ref;
        delete[] buffer_flt;
    }

#else
    // ===========================================================
    // VERSIONE HARDWARE STANDARD
    // ===========================================================
    void register_images_3d(
        std::vector<cv::Mat>& ref, std::vector<cv::Mat>& flt, Versal3DIR& board, int rangeX, int rangeY, float AngZ)
        override {
        double tx, ty, a11, a12, a21, a22;
        double avg_tx = 0.0, avg_ty = 0.0, avg_a11 = 0.0, avg_a21 = 0.0;

        for (int i = 0; i < ref.size(); i++) {
            estimate_initial(ref[i], flt[i], tx, ty, a11, a12, a21, a22);
            avg_tx += tx;
            avg_ty += ty;
            avg_a11 += a11;
            avg_a21 += a21;
        }

        avg_tx /= ref.size();
        avg_ty /= ref.size();
        float ang_rad = atan2(avg_a21, avg_a11);

        if (std::isnan(avg_tx)) avg_tx = 0.0;
        if (std::isnan(avg_ty)) avg_ty = 0.0;
        if (std::isnan(ang_rad)) ang_rad = 0.0;

        std::vector<double> init{avg_tx, avg_ty, ang_rad};
        std::vector<double> rng{(double)rangeX, (double)rangeY, (double)AngZ};
        std::pair<std::vector<double>::iterator, std::vector<double>::iterator> o{init.begin(), init.end()};

        optimize_powell(
            o, {rng.begin(), rng.end()}, std::bind(cost_function_3d, std::ref(board), std::placeholders::_1));

        tx = init[0];
        ty = init[1];
        ang_rad = init[2];
        board.hw_exec_tx(tx, ty, ang_rad, NULL, true);
    }

    // ===========================================================
    // VERSIONE HARDWARE PIRAMIDALE
    // ===========================================================
    void register_images_3d(
        Versal3DIR& board,
        int rangeX,
        int rangeY,
        float AngZ,
        float& tx,
        float& ty,
        float& theta,
        int num_iterations = 100000) override {
        std::cout << "[HW] Starting registration with init tx=" << tx << ", ty=" << ty << ", th=" << theta << "\n";

        counter = 0;
        std::vector<double> init{tx, ty, theta};
        std::vector<double> rng{(double)rangeX, (double)rangeY, (double)AngZ};
        std::pair<std::vector<double>::iterator, std::vector<double>::iterator> o{init.begin(), init.end()};

        optimize_powell(
            o,
            {rng.begin(), rng.end()},
            std::bind(cost_function_3d, std::ref(board), std::placeholders::_1),
            num_iterations);

        tx = static_cast<float>(init[0]);
        ty = static_cast<float>(init[1]);
        theta = static_cast<float>(init[2]);

        std::cout << "[HW] Updated params → tx=" << tx << ", ty=" << ty << ", th=" << theta << "\n";
        std::cout << "Value of global counter of iterations: " << counter << std::endl;
        board.hw_exec_tx(tx, ty, theta, NULL, true);
    }
#endif

   private:
#ifdef HW_REG
    static double cost_function_3d(Versal3DIR& board, std::vector<double>::iterator affine_params) {
        ++counter;
        const double tx = affine_params[0];
        const double ty = affine_params[1];
        const double ang_rad = affine_params[2];
        double val = exp(-board.hw_exec(tx, ty, ang_rad));
        return val;
    }
#else
    static double cost_function_3d(
        uint8_t* ref,
        uint8_t* flt,
        int depth,
        int n_row,
        int n_col,
        int padding,
        std::vector<double>::iterator affine_params) {
        const double tx = affine_params[0];
        const double ty = affine_params[1];
        const double ang = affine_params[2];
        double partial_mi = exp(-sw_registration_step_3d(ref, flt, depth, n_row, n_col, padding, tx, ty, ang));
        return partial_mi;
    }
#endif

    static void estimate_initial(
        cv::Mat ref, cv::Mat flt, double& tx, double& ty, double& a11, double& a12, double& a21, double& a22) {
        cv::Moments im_mom = moments(ref);
        cv::Moments pt_mom = moments(flt);
        tx = im_mom.m10 / im_mom.m00 - pt_mom.m10 / pt_mom.m00;
        ty = im_mom.m01 / im_mom.m00 - pt_mom.m01 / pt_mom.m00;
        a11 = 1.0;
        a12 = 0.0;
        a21 = 0.0;
        a22 = 1.0;
    }
};

#endif  // REGISTER_HPP
