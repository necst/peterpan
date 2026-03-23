#include "peterpan_runner.hpp"

#include <itkImage.h>
#include <itkMultiThreaderBase.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include "analyze/select_subvolume_lib.hpp"
#include "analyze/utils.hpp"
#include "app/imagefusion.hpp"
#include "core/optimize.hpp"
#include "core/register.hpp"
#include "include/image_utils/image_utils.hpp"
#include "pyramidal_main.hpp"

#ifdef HW_REG
#include "xrt/xrt_device.h"
struct FpgaContext {
    xrt::device device;
    xrt::uuid uuid;
};
#endif

struct PyramidalTiming {
    std::vector<double> lvl_total;
    std::vector<double> lvl_fpga_load;
    std::vector<double> lvl_fpga_kernel;
    std::vector<double> lvl_load_only;
    std::vector<double> lvl_exec_only;
};

struct LevelRange {
    int rangeX;
    int rangeY;
    float rangeAngZ;
};

using Clock = std::chrono::high_resolution_clock;
using DurationD = std::chrono::duration<double>;

std::vector<uint8_t> convert_itk_to_trilli_layout(itk::Image<unsigned char, 3>::Pointer img, int& nz_padded_out) {
    using ImageU8 = itk::Image<unsigned char, 3>;
    const auto region = img->GetLargestPossibleRegion();
    const auto sz = region.GetSize();

    const int nx = static_cast<int>(sz[0]);
    const int ny = static_cast<int>(sz[1]);
    const int nz = static_cast<int>(sz[2]);

    int nz_padded = ((nz + 31) / 32) * 32;
    nz_padded_out = nz_padded;

    const uint8_t* src = img->GetBufferPointer();

    std::vector<uint8_t> src_padded(static_cast<size_t>(nx) * ny * nz_padded, 0);
    std::memcpy(src_padded.data(), src, static_cast<size_t>(nx) * ny * nz);

    src = src_padded.data();

    std::vector<uint8_t> dst(static_cast<size_t>(nx) * ny * nz_padded);

#pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < nz_padded; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const size_t src_idx =
                    static_cast<size_t>(x) + static_cast<size_t>(y) * nx + static_cast<size_t>(z) * nx * ny;

                const size_t dst_idx =
                    (static_cast<size_t>(y) * nx + static_cast<size_t>(x)) * nz_padded + static_cast<size_t>(z);

                dst[dst_idx] = src[src_idx];
            }
        }
    }
    return dst;
}

#if defined(HW_REG) && defined(TRILLI_X)
RegistrationResult run_pyramidal_registration_trilli_reprogram_each_level(
    const std::vector<std::pair<const std::vector<uint8_t>*, const std::vector<uint8_t>*>>& pyramid,
    const std::vector<LevelRange>& ranges,
    int Dz,
    xrt::device& device,
    const std::vector<std::string>& xclbin_paths,
    const xrt::uuid& first_uuid,
    PyramidalTiming& pyr_timing,
    int num_iter) {
    if (xclbin_paths.size() != pyramid.size()) {
        throw std::runtime_error(
            "run_pyramidal_registration_trilli_reprogram_each_level: "
            "xclbin_paths.size() must match num_levels (pyramid.size()).");
    }

    RegistrationParams init{0.f, 0.f, 0.f};
    RegistrationResult last_res;

    for (size_t level = 0; level < pyramid.size(); ++level) {
        const auto& ref = *pyramid[level].first;
        const auto& mov = *pyramid[level].second;

        const size_t total_voxels = ref.size();
        const int rows = static_cast<int>(std::sqrt(static_cast<double>(total_voxels) / Dz));
        const int cols = rows;

        const LevelRange& R = ranges[level];

        xrt::uuid uuid;
        if (level == 0) {
            uuid = first_uuid;
        } else {
            xrt::xclbin xb(xclbin_paths[level]);
            uuid = device.load_xclbin(xb);
        }

        {
            Versal3DIR board(device, uuid, Dz);

            std::vector<uint8_t> output(board.buffer_size);
            board.output_flt = output.data();
            board.owns_out = false;

            last_res = run_registration_level(
                board,
                const_cast<std::vector<uint8_t>&>(ref),
                const_cast<std::vector<uint8_t>&>(mov),
                "output/level" + std::to_string(level + 1),
                Dz,
                rows,
                cols,
                R.rangeX,
                R.rangeY,
                R.rangeAngZ,
                init,
                pyr_timing.lvl_total[level],
                pyr_timing.lvl_fpga_load[level],
                pyr_timing.lvl_fpga_kernel[level],
                num_iter);
        }

        pyr_timing.lvl_load_only[level] = last_res.times.load_time;
        pyr_timing.lvl_exec_only[level] = last_res.times.exec_time;

        init = last_res.final_params;
        if (level + 1 < pyramid.size()) {
            const int resCurr = rows;
            const size_t next_voxels = pyramid[level + 1].first->size();
            const double resNext = std::sqrt(static_cast<double>(next_voxels) / Dz);
            const float scale = float(resNext) / float(resCurr);
            init.tx *= scale;
            init.ty *= scale;
        }
    }

    return last_res;
}
#endif

RegistrationResult run_registration_level(
#ifdef HW_REG
    Versal3DIR& board,
#endif
    std::vector<uint8_t>& ref_volume,
    std::vector<uint8_t>& float_volume,
    const std::string& /*output_folder*/,
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
    int num_iter) {
    RegistrationResult result;
    result.final_params = init_params;
    fpga_load_time = 0.0;
    fpga_kernel_time = 0.0;

    auto t_level_start = Clock::now();

#ifdef HW_REG
    if (board.load_volumes_from_data(ref_volume, float_volume) == -1)
        throw std::runtime_error("Error loading volumes to FPGA memory");

    auto t_exec_plus_load = Clock::now();
    double exec_time = imagefusion::perform_fusion_from_files_3d(
        ref_volume,
        float_volume,
        "mutualinformation",
        "alphablend",
        board,
        rangeX,
        rangeY,
        rangeAngZ,
        result.final_params.tx,
        result.final_params.ty,
        result.final_params.theta,
        num_iter);
    auto t_exec_plus_load_end = Clock::now();

    fpga_kernel_time = exec_time;
    fpga_load_time = DurationD(t_exec_plus_load_end - t_exec_plus_load).count() - exec_time;

#else
    int padding = (NUM_PIXELS_PER_READ - (n_couples % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ;

    uint8_t* registered_volume = new uint8_t[n_row * n_col * (n_couples + padding)];

    double exec_time = imagefusion::perform_fusion_from_files_3d(
        ref_volume,
        float_volume,
        "mutualinformation",
        "alphablend",
        n_couples,
        n_row,
        n_col,
        padding,
        rangeX,
        rangeY,
        rangeAngZ,
        result.final_params.tx,
        result.final_params.ty,
        result.final_params.theta,
        registered_volume);

    fpga_load_time = 0.0;
    fpga_kernel_time = exec_time;

    result.registered_volume.assign(registered_volume, registered_volume + n_row * n_col * (n_couples + padding));

    delete[] registered_volume;
#endif

    auto t_level_end = Clock::now();

    level_total_time = DurationD(t_level_end - t_level_start).count();
    result.times.load_time = fpga_load_time;
    result.times.exec_time = fpga_kernel_time;

    return result;
}

RegistrationResult run_pyramidal_registration_trilli(
    const std::vector<std::pair<const std::vector<uint8_t>*, const std::vector<uint8_t>*>>& pyramid,
    const std::vector<LevelRange>& ranges,
    int Dz,
#ifdef HW_REG
    Versal3DIR& board,
#endif
    PyramidalTiming& pyr_timing,
    int num_iter) {
    RegistrationParams init{0.f, 0.f, 0.f};
    RegistrationResult last_res;

    for (size_t level = 0; level < pyramid.size(); ++level) {
        const auto& ref = *pyramid[level].first;
        const auto& mov = *pyramid[level].second;

        size_t total_voxels = ref.size();
        int rows = static_cast<int>(std::sqrt(static_cast<double>(total_voxels) / Dz));
        int cols = rows;

#if defined(HW_REG) && !defined(TRILLI_X)
        board.update_dimensions(Dz, rows, cols);
#endif

        const LevelRange& R = ranges[level];

        last_res = run_registration_level(
#ifdef HW_REG
            board,
#endif
            const_cast<std::vector<uint8_t>&>(ref),
            const_cast<std::vector<uint8_t>&>(mov),
            "output/level" + std::to_string(level + 1),
            Dz,
            rows,
            cols,
            R.rangeX,
            R.rangeY,
            R.rangeAngZ,
            init,
            pyr_timing.lvl_total[level],
            pyr_timing.lvl_fpga_load[level],
            pyr_timing.lvl_fpga_kernel[level],
            num_iter);

        pyr_timing.lvl_load_only[level] = last_res.times.load_time;
        pyr_timing.lvl_exec_only[level] = last_res.times.exec_time;

        init = last_res.final_params;

        if (level + 1 < pyramid.size()) {
            int resCurr = rows;
            size_t next_voxels = pyramid[level + 1].first->size();
            double resNext = std::sqrt(static_cast<double>(next_voxels) / Dz);
            float scale = float(resNext) / float(resCurr);
            init.tx *= scale;
            init.ty *= scale;
        }
    }

    return last_res;
}

std::vector<LevelRange> compute_ranges(int num_levels, int /*original_res*/) {
    std::vector<LevelRange> ranges(num_levels);

    if (num_levels <= 0) return ranges;

    const double base_range = 80.0;
    const double ref_res = 512.0;

    if (num_levels == 1) {
        ranges[0].rangeX = static_cast<int>(std::round(base_range));
        ranges[0].rangeY = static_cast<int>(std::round(base_range));
        ranges[0].rangeAngZ = 1.0f;
        return ranges;
    }

    for (int lvl = 0; lvl < num_levels; ++lvl) {
        int level_res = 64 * (lvl + 1);
        if (level_res > static_cast<int>(ref_res)) level_res = static_cast<int>(ref_res);

        double r = base_range * (static_cast<double>(level_res) / ref_res);

        ranges[lvl].rangeX = static_cast<int>(std::round(r));
        ranges[lvl].rangeY = static_cast<int>(std::round(r));

        if (lvl == 0)
            ranges[lvl].rangeAngZ = 1.0f;
        else
            ranges[lvl].rangeAngZ = 0.0f;
    }

    ranges[num_levels - 1].rangeX = static_cast<int>(std::round(base_range));
    ranges[num_levels - 1].rangeY = static_cast<int>(std::round(base_range));
    ranges[num_levels - 1].rangeAngZ = 0.0f;

    return ranges;
}

int peterpan_runner(const std::vector<std::string>& args) {
    using namespace std;
    namespace fs = std::filesystem;

    const int argc = static_cast<int>(args.size());
    std::vector<char*> argv;
    argv.reserve(static_cast<size_t>(argc));
    for (const auto& s : args) {
        argv.push_back(const_cast<char*>(s.c_str()));
    }

#ifdef HW_REG
    cout << "RUNNING ON HARDWARE" << endl;
#else
    cout << "RUNNING ON SOFTWARE" << endl;
#endif

    if (argc < 3) {
        cerr << "Usage:\n"
             << "  <fixed_dir_or_file> <moving_dir_or_file> <Dz> "
                "[--mask=<otsu|zero|pct>] [--type=<png|nii|png8|uint8|nii8>] "
                "[--gradient=<itk|manual>] "
                "[--xclbin=<path>] (PETERPAN)  OR  [--xclbin1=... --xclbin2=... ...] (TRILLI_X)\n"
                "[--num_levels=<levels>] [--num_iter=<num_iter>] "
                "[--zmode=<auto|fixed>] [--zstart=<z0>] [--zend=<z1>]"
#ifdef HW_REG
                " [--device_id=<id>]"
#endif
             << "\n";
        return 1;
    }

    std::string fixedPath, movingPath;
    std::string maskType = "otsu";
    std::string type = "png";
    std::string gradientMode = "itk";
    std::string xclbin_path = "path/to/bitstream.xclbin";
    std::vector<std::string> xclbin_paths;
    int Dz = 0;
    int num_iter = 100000;
    std::string zMode = "auto";
    int zStart = 0;
    int zEnd = 0;
    int num_levels = 4;
    bool zStartProvided = false;
    bool zEndProvided = false;

#ifdef HW_REG
    int device_id = 0;
#endif

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[static_cast<size_t>(i)];

        if (arg.rfind("--type=", 0) == 0) {
            type = arg.substr(7);
        } else if (arg.rfind("--mask=", 0) == 0) {
            maskType = arg.substr(7);
        } else if (arg.rfind("--gradient=", 0) == 0) {
            gradientMode = arg.substr(11);
        } else if (arg.rfind("--xclbin=", 0) == 0) {
            xclbin_path = arg.substr(9);
        } else if (arg.rfind("--xclbin", 0) == 0) {
            const std::string prefix = "--xclbin";
            if (arg.size() > prefix.size() && std::isdigit(static_cast<unsigned char>(arg[prefix.size()]))) {
                auto eq = arg.find('=');
                if (eq == std::string::npos) {
                    throw std::runtime_error("Invalid xclbin flag (missing '='): " + arg);
                }

                std::string idx_str = arg.substr(prefix.size(), eq - prefix.size());
                int idx = std::stoi(idx_str);
                if (idx <= 0) {
                    throw std::runtime_error("Invalid xclbin index (must be >=1): " + arg);
                }

                std::string path = arg.substr(eq + 1);
                if (path.empty()) {
                    throw std::runtime_error("Invalid xclbin flag (empty path): " + arg);
                }

                if (static_cast<int>(xclbin_paths.size()) < idx) {
                    xclbin_paths.resize(static_cast<size_t>(idx));
                }
                xclbin_paths[static_cast<size_t>(idx - 1)] = path;
            }
        } else if (arg.rfind("--num_levels=", 0) == 0) {
            num_levels = std::stoi(arg.substr(13));
        } else if (arg.rfind("--num_iter=", 0) == 0) {
            num_iter = std::stoi(arg.substr(11));
        } else if (arg.rfind("--zmode=", 0) == 0) {
            zMode = arg.substr(8);
        } else if (arg.rfind("--zstart=", 0) == 0) {
            zStart = std::stoi(arg.substr(9));
            zStartProvided = true;
        } else if (arg.rfind("--zend=", 0) == 0) {
            zEnd = std::stoi(arg.substr(7));
            zEndProvided = true;
#ifdef HW_REG
        } else if (arg.rfind("--device_id=", 0) == 0) {
            device_id = std::stoi(arg.substr(12));
#endif
        } else if (fixedPath.empty()) {
            fixedPath = arg;
        } else if (movingPath.empty()) {
            movingPath = arg;
        } else if (Dz == 0) {
            Dz = std::stoi(arg);
        }
    }

#ifdef HW_REG
#ifdef TRILLI_X
    if (static_cast<int>(xclbin_paths.size()) != num_levels) {
        std::cerr << "ERROR: with TRILLI-X you must pass --xclbin1..--xclbin" << num_levels << " (one per level)\n";
        return 1;
    }
    for (int i = 0; i < num_levels; ++i) {
        if (xclbin_paths[static_cast<size_t>(i)].empty()) {
            std::cerr << "ERROR: missing --xclbin" << (i + 1) << " (required when using TRILLI-X)\n";
            return 1;
        }
    }
#else
    if (xclbin_path.empty()) {
        std::cerr << "ERROR: with PETERPAN you must pass --xclbin=<path>\n";
        return 1;
    }
#endif
#endif

    if (num_levels == 1) num_iter = 100000;

    if (fixedPath.empty() || movingPath.empty() || Dz <= 0) {
        cerr << "Missing required arguments.\n";
        return 1;
    }

    if (type != "png" && type != "nii" && type != "png8" && type != "uint8" && type != "nii8") {
        cerr << "Unsupported type: " << type << " (use png, nii, png8, uint8, or nii8)\n";
        return 1;
    }

    if (gradientMode != "itk" && gradientMode != "manual") {
        cerr << "Unsupported gradient mode: " << gradientMode << " (use itk or manual)\n";
        return 1;
    }

    if (zMode != "auto" && zMode != "fixed") {
        cerr << "Unsupported zmode: " << zMode << " (use auto or fixed)\n";
        return 1;
    }

    PyramidalTiming pyr_timing;
    pyr_timing.lvl_total.resize(static_cast<size_t>(num_levels), 0.0);
    pyr_timing.lvl_fpga_load.resize(static_cast<size_t>(num_levels), 0.0);
    pyr_timing.lvl_fpga_kernel.resize(static_cast<size_t>(num_levels), 0.0);
    pyr_timing.lvl_load_only.resize(static_cast<size_t>(num_levels), 0.0);
    pyr_timing.lvl_exec_only.resize(static_cast<size_t>(num_levels), 0.0);

    double w1 = 1;
    double w2 = 1;
    double w3 = 0;
    double w4 = 0;
    double w5 = 0.3;

    try {
        using ImageU8 = itk::Image<unsigned char, 3>;

#ifdef HW_REG
        auto future_fpga = std::async(std::launch::async, [=]() {
            FpgaContext ctx{xrt::device(device_id), xrt::uuid{}};

#ifdef TRILLI_X
            const std::string& first_bitstream = xclbin_paths[0];
#else
            const std::string& first_bitstream = xclbin_path;
#endif

            ctx.uuid = ctx.device.load_xclbin(first_bitstream);
            return ctx;
        });
#endif

        std::cout << "[PHASE] Loading volumes...\n";
        auto t_load_start = Clock::now();

        auto loadKind = type;
        if (type == "uint8" || type == "png8" || type == "png_uint8") loadKind = "png";
        if (type == "nii8") loadKind = "nii";

        ImageU8::Pointer fixed = loadImageU8(fixedPath, loadKind);
        ImageU8::Pointer moving = loadImageU8(movingPath, loadKind);
        int original_res = static_cast<int>(moving->GetLargestPossibleRegion().GetSize()[0]);

        auto t_load_end = Clock::now();
        double load_time = DurationD(t_load_end - t_load_start).count();

        if (zMode == "fixed") {
            const auto region = fixed->GetLargestPossibleRegion();
            const auto size = region.GetSize();
            const int Z = static_cast<int>(size[2]);

            if (!zStartProvided || !zEndProvided) {
                throw std::runtime_error("zmode=fixed requires --zstart and --zend.");
            }
            if (zStart < 0 || zStart >= Z || zEnd < 0 || zEnd >= Z) {
                throw std::runtime_error("zstart/zend out of bounds for volume Z size.");
            }
            if (zEnd < zStart) {
                throw std::runtime_error("zend must be >= zstart.");
            }

            Dz = zEnd - zStart + 1;
            std::cout << "[Z-RANGE] Using fixed range: [" << zStart << ", " << zEnd << "] (Dz=" << Dz << ")\n";
        }

        auto t_pre_start = Clock::now();

        std::future<std::vector<LevelRange>> future_ranges =
            std::async(std::launch::async, [=]() { return compute_ranges(num_levels, original_res); });

        const int step = 64;
        MultiResSubvolumeU8 multiRes;

        if (zMode == "fixed") {
            multiRes = extract_fixed_range(fixed, moving, zStart, zEnd, num_levels, step);
        } else {
            multiRes = selectSubvolumeUChar(
                fixed, moving, Dz, w1, w2, w3, w4, w5, num_levels, step, maskType, 0.0, gradientMode);
        }

        auto t_pre_end = Clock::now();
        double pre_processing_time = DurationD(t_pre_end - t_pre_start).count();

        auto future_convert = std::async(std::launch::async, [moving]() {
            int nz_padded = 0;
            auto vec = convert_itk_to_trilli_layout(moving, nz_padded);
            return std::make_pair(std::move(vec), nz_padded);
        });

        if (Dz % NUM_PIXELS_PER_READ != 0) {
            Dz = ((Dz + NUM_PIXELS_PER_READ - 1) / NUM_PIXELS_PER_READ) * NUM_PIXELS_PER_READ;
        }

        auto t_pyr_start = Clock::now();
        double t_setup_board = 0.0;
        double t_remaining_xclbin_load = 0.0;

#ifdef HW_REG
        auto t_get_fpga = Clock::now();
        FpgaContext fpga = future_fpga.get();
        auto t_get_fpga_end = Clock::now();
        t_remaining_xclbin_load = DurationD(t_get_fpga_end - t_get_fpga).count();
#endif

        std::vector<std::pair<const std::vector<uint8_t>*, const std::vector<uint8_t>*>> pyramid;
        for (const auto& lvl : multiRes.levels) {
            pyramid.emplace_back(&lvl.ct, &lvl.pet);
        }

        std::vector<LevelRange> ranges = future_ranges.get();
        RegistrationResult final_result;

#ifdef HW_REG
#ifdef TRILLI_X
        t_setup_board = 0.0;
        t_remaining_xclbin_load = 0.0;

        final_result = run_pyramidal_registration_trilli_reprogram_each_level(
            pyramid, ranges, Dz, fpga.device, xclbin_paths, fpga.uuid, pyr_timing, num_iter);
#else
        auto t_board_obj_create = Clock::now();

        Versal3DIR board(fpga.device, fpga.uuid, Dz, original_res, original_res);

        std::vector<uint8_t> output(board.buffer_size);
        board.output_flt = output.data();

        auto t_board_obj_create_end = Clock::now();
        t_setup_board = DurationD(t_board_obj_create_end - t_board_obj_create).count();

        final_result = run_pyramidal_registration_trilli(pyramid, ranges, Dz, board, pyr_timing, num_iter);
#endif
#else
        final_result = run_pyramidal_registration_trilli(pyramid, ranges, Dz, pyr_timing, num_iter);
#endif

        auto t_pyr_end = Clock::now();
        double pyr_total_time = DurationD(t_pyr_end - t_pyr_start).count();
        double pyr_reg_no_setup = pyr_total_time - (t_setup_board + t_remaining_xclbin_load);

        ImageU8::RegionType region = moving->GetLargestPossibleRegion();
        ImageU8::SizeType size = region.GetSize();
        const int n_row_orig = static_cast<int>(size[0]);
        const int n_col_orig = static_cast<int>(size[1]);
        const int n_slices_orig = static_cast<int>(size[2]);

        RegistrationParams final_scaled = final_result.final_params;

        cout << "[FINAL] Scaled parameters: X: " << final_scaled.tx << " Y: " << final_scaled.ty
             << " ANG: " << final_scaled.theta << endl;

        auto t_convert_time = Clock::now();
        auto convert_res = future_convert.get();
        auto& moving_original_vec = convert_res.first;
        int nz_padded = convert_res.second;
        int depth_padding = nz_padded - n_slices_orig;
        auto t_convert_time_end = Clock::now();
        double convert_time = DurationD(t_convert_time_end - t_convert_time).count();

        std::vector<uint8_t> transformed(static_cast<size_t>(n_row_orig) * n_col_orig * nz_padded, 0);

        auto t_trans_start = Clock::now();
        transform_volume(
            moving_original_vec.data(),
            transformed.data(),
            final_scaled.tx,
            final_scaled.ty,
            final_scaled.theta,
            n_row_orig,
            n_col_orig,
            nz_padded,
            true);
        auto t_trans_end = Clock::now();
        double transform_time = DurationD(t_trans_end - t_trans_start).count();

        fs::path outFinal = fs::path("output") / "final_registered";
        fs::create_directories(outFinal);

        write_volume_to_file(
            transformed.data(), n_row_orig, n_col_orig, n_slices_orig, 0, 0, depth_padding, outFinal.string());

        {
            std::string name = "accuracy_" + std::to_string(Dz);
            std::string acc_path = "output/" + name + ".csv";
            bool acc_exists = fs::exists(acc_path);

            std::ofstream acc(acc_path, std::ios::app);
            if (!acc) {
                std::cerr << "ERROR: Cannot write accuracy CSV file.\n";
            } else {
                if (!acc_exists) {
                    acc << "range,numLevels,tx,ty,ang\n";
                }

                acc << "[" << zStart << ";" << zEnd << "]"
                    << "," << num_levels << "," << final_scaled.tx << "," << final_scaled.ty << ","
                    << final_scaled.theta << "\n";
            }
        }

        double e2e_total_time = pre_processing_time + convert_time + pyr_total_time + transform_time;
        double e2e_total_time_nosetup = pre_processing_time + convert_time + pyr_reg_no_setup + transform_time;

        cout << "\n===== TIMINGS (seconds) =====\n";
        cout << "[LOAD]            " << load_time << "\n";
        cout << "[PRE-PROC]        " << pre_processing_time << "\n";
        cout << "[CONVERT]         " << convert_time << "\n";

        double reg_levels_sum = 0;
        for (int i = 0; i < num_levels; i++) {
            cout << "\n[LVL" << i + 1 << " TOTAL] " << pyr_timing.lvl_total[static_cast<size_t>(i)] << "\n";
            cout << "[LVL" << i + 1 << " LOAD]  " << pyr_timing.lvl_load_only[static_cast<size_t>(i)] << "\n";
            cout << "[LVL" << i + 1 << " EXEC]  " << pyr_timing.lvl_exec_only[static_cast<size_t>(i)] << "\n";
            reg_levels_sum += pyr_timing.lvl_total[static_cast<size_t>(i)];
        }

        cout << "\n[REG LEVELS SUM]  " << reg_levels_sum << "\n";
        cout << "[PYR TOTAL]       " << pyr_total_time << "\n";
        cout << "[PYR TOTAL NO SETUP] " << pyr_reg_no_setup << "\n";
        cout << "[FINAL TRANSFORM] " << transform_time << "\n";
        cout << "-----------------------------\n";
        cout << "[E2E TOTAL]       " << e2e_total_time << "\n";
        cout << "[E2E TOTAL - NO SETUP] " << e2e_total_time_nosetup << "\n";

        std::string csv_path =
            "output/timings_dz" + std::to_string(Dz) + "_levels" + std::to_string(num_levels) + ".csv";

        bool file_exists = fs::exists(csv_path);

        std::ofstream csv(csv_path, std::ios::app);
        if (!csv) {
            std::cerr << "ERROR: Cannot write CSV file.\n";
        } else {
            if (!file_exists) {
                csv << "PyramidalLevels,Load,Preproc,Convert,";
                for (int i = 0; i < num_levels; ++i) {
                    csv << "Lvl" << (i + 1) << "_total,"
                        << "Lvl" << (i + 1) << "_load,"
                        << "Lvl" << (i + 1) << "_exec,";
                }
                csv << "RegSum,PyrTotal,PyrTotal_nosetup,FinalTransform,E2E,e2e_total_time_nosetup\n";
            }

            csv << num_levels << "," << load_time << "," << pre_processing_time << "," << convert_time << ",";

            for (int i = 0; i < num_levels; ++i) {
                csv << pyr_timing.lvl_total[static_cast<size_t>(i)] << ","
                    << pyr_timing.lvl_load_only[static_cast<size_t>(i)] << ","
                    << pyr_timing.lvl_exec_only[static_cast<size_t>(i)] << ",";
            }

            csv << reg_levels_sum << "," << pyr_total_time << "," << pyr_reg_no_setup << "," << transform_time << ","
                << e2e_total_time << "," << e2e_total_time_nosetup << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}