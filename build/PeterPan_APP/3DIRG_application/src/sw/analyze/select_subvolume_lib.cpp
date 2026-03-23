#include "select_subvolume_lib.hpp"

#include <itkExtractImageFilter.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkMultiThreaderBase.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>

#include "utils.hpp"  // contiene alias float path + helpers

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
//                          PARAMETRI ISTOGRAMMA MI
// ============================================================================
#ifndef J_HISTO_ROWS
#define J_HISTO_ROWS 256
#endif

#ifndef J_HISTO_COLS
#define J_HISTO_COLS 256
#endif

inline double normalized_mutual_information_u8(
    const uint8_t* input_ref, const uint8_t* input_flt, int depth, int n_row, int n_col) {
    const int IMG_SIZE = n_row * n_col;
    const int N = depth;

    double j_h[J_HISTO_ROWS][J_HISTO_COLS];
    std::memset(j_h, 0, sizeof(j_h));

    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif

    std::vector<std::vector<double>> local_hist(nthreads, std::vector<double>(J_HISTO_ROWS * J_HISTO_COLS, 0.0));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_row; i++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& H = local_hist[tid];

        for (int j = 0; j < n_col; j++) {
            int base = (i * n_col + j) * depth;
            for (int k = 0; k < depth; k++) {
                unsigned int a = input_ref[base + k];
                unsigned int b = input_flt[base + k];
                H[a * J_HISTO_COLS + b] += 1.0;
            }
        }
    }

    // Reduce
    for (int t = 0; t < nthreads; t++) {
        auto& H = local_hist[t];
        for (int a = 0; a < J_HISTO_ROWS; a++)
            for (int b = 0; b < J_HISTO_COLS; b++) j_h[a][b] += H[a * J_HISTO_COLS + b];
    }

    const double norm = double(N) * double(IMG_SIZE);
    for (int a = 0; a < J_HISTO_ROWS; a++)
        for (int b = 0; b < J_HISTO_COLS; b++) j_h[a][b] /= norm;

    // Marginals
    double href[J_HISTO_ROWS] = {0.0};
    double hflt[J_HISTO_ROWS] = {0.0};

    for (int a = 0; a < J_HISTO_ROWS; a++)
        for (int b = 0; b < J_HISTO_COLS; b++) {
            href[a] += j_h[a][b];
            hflt[b] += j_h[a][b];
        }

    auto entropy = [](const double* h, int n) {
        double H = 0.0;
        for (int i = 0; i < n; i++)
            if (h[i] > 1e-15) H -= h[i] * std::log(h[i]);
        return H;
    };

    double Href = entropy(href, J_HISTO_ROWS);
    double Hflt = entropy(hflt, J_HISTO_COLS);

    // Joint entropy
    double Hjoint = 0.0;
    for (int a = 0; a < J_HISTO_ROWS; a++)
        for (int b = 0; b < J_HISTO_COLS; b++) {
            double v = j_h[a][b];
            if (v > 1e-15) Hjoint -= v * std::log(v);
        }

    // NMI classica: (H(A)+H(B)) / Hjoint
    return (Href + Hflt) / Hjoint;
}
inline double mutual_information_u8(
    const uint8_t* input_ref, const uint8_t* input_flt, int depth, int n_row, int n_col) {
    constexpr int R = 256, C = 256;
    constexpr double eps = 1e-12;

    const long Nvox = long(depth) * n_row * n_col;

    // Numero thread
    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif

    // Istogrammi locali
    std::vector<std::vector<double>> local_hist(nthreads, std::vector<double>(R * C, 0.0));

    // ==========================================================
    // PARALLEL HISTOGRAM
    // ==========================================================
#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& H = local_hist[tid];

#pragma omp for schedule(static)
        for (long idx = 0; idx < Nvox; idx++) {
            uint8_t a = input_ref[idx];
            uint8_t b = input_flt[idx];
            H[a * C + b] += 1.0;
        }
    }

    // ==========================================================
    // REDUCE INTO FINAL HISTOGRAM
    // ==========================================================
    std::vector<double> joint(R * C, 0.0);

    for (int t = 0; t < nthreads; t++) {
        const auto& H = local_hist[t];
        for (int i = 0; i < R * C; i++) joint[i] += H[i];
    }

    // Normalize
    const double invN = 1.0 / double(Nvox);
    for (double& v : joint) v *= invN;

    // ==========================================================
    // ENTROPIE E MARGINALI
    // ==========================================================
    std::vector<double> href(R, 0.0);
    std::vector<double> hflt(C, 0.0);

    double Hjoint = 0.0;

    for (int a = 0; a < R; a++) {
        for (int b = 0; b < C; b++) {
            double v = joint[a * C + b];
            href[a] += v;
            hflt[b] += v;
            if (v > eps) Hjoint -= v * std::log(v);
        }
    }

    double Href = 0.0;
    for (int a = 0; a < R; a++)
        if (href[a] > eps) Href -= href[a] * std::log(href[a]);

    double Hflt = 0.0;
    for (int b = 0; b < C; b++)
        if (hflt[b] > eps) Hflt -= hflt[b] * std::log(hflt[b]);

    return Href + Hflt - Hjoint;
}
// ============================================================================
//
//
// FUNZIONE PER ESTRARRE BLOCCO RAW U8
// ============================================================================
using ImageU8 = itk::Image<unsigned char, 3>;

inline std::vector<uint8_t> extractBlockRawU8(ImageU8::Pointer img, int z0, int Dz) {
    const auto region = img->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    int Nx = size[0];
    int Ny = size[1];

    size_t sliceSize = size_t(Nx) * Ny;
    std::vector<uint8_t> out(sliceSize * Dz);

    const uint8_t* buf = img->GetBufferPointer();
    size_t dst = 0;

    for (int z = z0; z < z0 + Dz; z++) {
        size_t src = size_t(z) * sliceSize;
        std::memcpy(out.data() + dst, buf + src, sliceSize);
        dst += sliceSize;
    }

    return out;
}

// ===================================================================
// ================ IMPLEMENTAZIONE: FLOAT (classica) =================
// ===================================================================
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> selectSubvolumeFloat(
    const std::string& ctPath,
    const std::string& petPath,
    int Dz,
    double w1,
    double w2,
    double w3,
    double w4,
    int num_vols,
    const std::string& maskMode,
    double pctValue,
    const std::string& type) {
    // Load as float (ImageType is float, defined in utils.hpp)
    ImageType::Pointer fixed = loadImage(ctPath, type);
    ImageType::Pointer moving = loadImage(petPath, type);

    const auto region = fixed->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const int Z = static_cast<int>(size[2]);

    std::cout << "Mask mode: " << maskMode;
    if (maskMode == "pct") std::cout << " (p=" << pctValue << ")";
    std::cout << std::endl;

    auto t0 = Clock::now();

    // ---- Mask ----
    auto t_mask_start = Clock::now();
    MaskType::Pointer maskF, maskM;
    if (maskMode == "zero") {
        auto f1 = std::async(std::launch::async, [&] { return makeMaskZeroBased_v2(fixed); });
        auto f2 = std::async(std::launch::async, [&] { return makeMaskZeroBased_v2(moving); });
        maskF = f1.get();
        maskM = f2.get();
    } else if (maskMode == "otsu") {
        auto f1 = std::async(std::launch::async, [&] { return makeMaskOtsu_v2(fixed); });
        auto f2 = std::async(std::launch::async, [&] { return makeMaskOtsu_v2(moving); });
        maskF = f1.get();
        maskM = f2.get();
    } else if (maskMode == "pct") {
        auto f1 = std::async(std::launch::async, [&] { return makeMaskPercentile_v2(fixed, pctValue); });
        auto f2 = std::async(std::launch::async, [&] { return makeMaskPercentile_v2(moving, pctValue); });
        maskF = f1.get();
        maskM = f2.get();
    } else {
        throw std::runtime_error("Unknown mask mode: " + maskMode);
    }
    auto t_mask_end = Clock::now();
    std::cout << "Time to compute masks: " << std::chrono::duration<double>(t_mask_end - t_mask_start).count()
              << " s\n";

    // ---- Intersection mask ----
    auto t_maskI_start = Clock::now();
    auto maskI = MaskType::New();
    maskI->SetRegions(region);
    maskI->Allocate();

#pragma omp parallel for schedule(static)
    for (int z = 0; z < Z; ++z) {
        for (int y = 0; y < (int)size[1]; ++y) {
            for (int x = 0; x < (int)size[0]; ++x) {
                MaskType::IndexType idx = {x, y, z};
                unsigned char v = maskF->GetPixel(idx) & maskM->GetPixel(idx);
                maskI->SetPixel(idx, v);
            }
        }
    }
    auto t_maskI_end = Clock::now();
    std::cout << "Time to compute intersection mask: "
              << std::chrono::duration<double>(t_maskI_end - t_maskI_start).count() << " s\n";

    // ---- Gradient (float) ----
    auto t_grad_start = Clock::now();
    using GradFilter = itk::GradientMagnitudeImageFilter<ImageType, GradType>;
    auto gfF = GradFilter::New();
    gfF->SetInput(fixed);
    auto gfM = GradFilter::New();
    gfM->SetInput(moving);
    const int wu = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    gfF->SetNumberOfWorkUnits(wu);
    gfM->SetNumberOfWorkUnits(wu);

    auto futF = std::async(std::launch::async, [&] {
        gfF->Update();
        return gfF->GetOutput();
    });
    auto futM = std::async(std::launch::async, [&] {
        gfM->Update();
        return gfM->GetOutput();
    });

    GradType::Pointer gradF = futF.get();
    GradType::Pointer gradM = futM.get();
    auto t_grad_end = Clock::now();

    std::cout << "Time to compute gradients: " << std::chrono::duration<double>(t_grad_end - t_grad_start).count()
              << " s\n";

    // ---- Prefix sums ----
    auto t_prefix_start = Clock::now();
    auto psF = prefixZSum_MT(maskF.GetPointer());
    auto psM = prefixZSum_MT(maskM.GetPointer());
    auto psI = prefixZSum_MT(maskI.GetPointer());
    auto psGF = prefixZSum_MT(gradF.GetPointer());
    auto psGM = prefixZSum_MT(gradM.GetPointer());
    auto t_prefix_end = Clock::now();
    std::cout << "Time to compute prefix sums: " << std::chrono::duration<double>(t_prefix_end - t_prefix_start).count()
              << " s\n";

    // ---- Sliding window ----
    auto t_sliding_start = Clock::now();
    double bestScore = -1e300;
    int best_z0 = 0, best_z1 = Dz;
    const double sliceVox = static_cast<long double>(size[0]) * static_cast<long double>(size[1]);
    const double min_intersection_ratio = 0.03;
    const int maxZ = static_cast<int>(size[2]) - Dz;

#pragma omp parallel
    {
        double localBestScore = -1e300;
        int localZ0 = 0, localZ1 = Dz;

#pragma omp for schedule(dynamic)
        for (int z0 = 0; z0 <= maxZ; ++z0) {
            const int z1 = z0 + Dz;
            const double nzF = psF[z1] - psF[z0];
            const double nzM = psM[z1] - psM[z0];
            const double nzI = psI[z1] - psI[z0];
            const double gF = psGF[z1] - psGF[z0];
            const double gM = psGM[z1] - psGM[z0];

            const double fgF = nzF / (Dz * sliceVox);
            const double fgM = nzM / (Dz * sliceVox);
            const double inter = nzI / (Dz * sliceVox);
            if (inter < min_intersection_ratio) continue;

            const double gfn = gF / (Dz * sliceVox);
            const double gmn = gM / (Dz * sliceVox);

            const double score = w1 * fgF + w2 * fgM + w3 * gfn + w4 * gmn;
            if (score > localBestScore) {
                localBestScore = score;
                localZ0 = z0;
                localZ1 = z1;
            }
        }

#pragma omp critical
        {
            if (localBestScore > bestScore) {
                bestScore = localBestScore;
                best_z0 = localZ0;
                best_z1 = localZ1;
            }
        }
    }
    auto t_sliding_end = Clock::now();
    std::cout << "Time to compute best Z range: "
              << std::chrono::duration<double>(t_sliding_end - t_sliding_start).count() << " s\n";

    std::cout << "Best Z range = [" << best_z0 << ", " << best_z1 - 1 << "]  (Dz=" << Dz << ")\n";

    // ---- Subvolume extraction (slices) ----
    std::cout << "Extracting slices..." << std::endl;
    auto t2 = Clock::now();
    auto slicesFixed = extractSlices(fixed, best_z0, Dz);
    auto slicesMoving = extractSlices(moving, best_z0, Dz);
    auto t3 = Clock::now();
    std::cout << "Time to extract slices: " << std::chrono::duration<double>(t3 - t2).count() << " s\n";

    // (optional) isotropic resampling + resize here, if needed. Not required for A/B/C.

    // ---- Return as flattened uint8 (per-slice normalization) ----
    auto ctVec = slicesToUint8Volume(slicesFixed);
    auto petVec = slicesToUint8Volume(slicesMoving);

    return {ctVec, petVec};
}

// ===================================================================
// =============== IMPLEMENTAZIONE: UINT8 (nativa) ====================
// ===================================================================
namespace {
// Local U8 types
using ImageU8 = itk::Image<unsigned char, 3>;
using GradU8 = ImageU8;  // use same type for gradient output
using MaskU8 = ImageU8;
using SliceU8 = itk::Image<unsigned char, 2>;

// Z-axis prefix sum on U8 buffer (fast, no ITK iterator)
inline std::vector<long double> prefixZSumU8(ImageU8* img) {
    const auto region = img->GetLargestPossibleRegion();
    const auto sz = region.GetSize();
    const size_t nx = sz[0], ny = sz[1], nz = sz[2];
    const size_t sliceSize = nx * ny;

    const unsigned char* buf = img->GetBufferPointer();
    std::vector<long double> sums(nz + 1, 0.0L);

    // Per-slice sum in parallel
    std::vector<long double> sliceSums(nz, 0.0L);
#pragma omp parallel for schedule(static)
    for (int z = 0; z < (int)nz; ++z) {
        const size_t off = static_cast<size_t>(z) * sliceSize;
        long double acc = 0.0L;
        for (size_t i = 0; i < sliceSize; ++i) acc += (long double)buf[off + i];
        sliceSums[z] = acc;
    }

    for (size_t z = 0; z < nz; ++z) sums[z + 1] = sums[z] + sliceSums[z];
    return sums;
}

// Maschera "zero" su U8: 1 se pixel!=0
inline MaskU8::Pointer makeMaskZeroU8(ImageU8::Pointer img) {
    auto mask = MaskU8::New();
    mask->SetRegions(img->GetLargestPossibleRegion());
    mask->Allocate();
    const size_t N = img->GetLargestPossibleRegion().GetNumberOfPixels();
    const unsigned char* in = img->GetBufferPointer();
    unsigned char* out = mask->GetBufferPointer();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i) out[i] = (in[i] != 0) ? 1 : 0;
    return mask;
}

// Otsu su U8: semplice histogram-based (256 bins)
inline MaskU8::Pointer makeMaskOtsuU8(ImageU8::Pointer img) {
    const auto region = img->GetLargestPossibleRegion();
    const size_t N = region.GetNumberOfPixels();
    const unsigned char* buf = img->GetBufferPointer();

    // Istogramma 256
    std::array<unsigned long long, 256> hist{};
    hist.fill(0);
#pragma omp parallel
    {
        std::array<unsigned long long, 256> hloc{};
        hloc.fill(0);
#pragma omp for nowait schedule(static)
        for (long i = 0; i < (long)N; ++i) hloc[buf[i]]++;
#pragma omp critical
        {
            for (int b = 0; b < 256; ++b) hist[b] += hloc[b];
        }
    }

    // Otsu
    unsigned long long total = 0;
    for (int b = 0; b < 256; ++b) total += hist[b];
    double sumAll = 0.0;
    for (int b = 0; b < 256; ++b) sumAll += (double)b * (double)hist[b];

    unsigned long long wB = 0;
    double sumB = 0.0;
    double maxVar = -1.0;
    int thr = 0;

    for (int b = 0; b < 256; ++b) {
        wB += hist[b];
        if (wB == 0) continue;
        unsigned long long wF = total - wB;
        if (wF == 0) break;
        sumB += (double)b * (double)hist[b];
        double mB = sumB / (double)wB;
        double mF = (sumAll - sumB) / (double)wF;
        double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) {
            maxVar = varBetween;
            thr = b;
        }
    }

    auto mask = MaskU8::New();
    mask->SetRegions(region);
    mask->Allocate();
    unsigned char* out = mask->GetBufferPointer();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i) out[i] = (buf[i] > thr) ? 1 : 0;
    return mask;
}

// Percentile su U8: soglia = quantile p (0..1) sui valori [0..255]
inline MaskU8::Pointer makeMaskPercentileU8(ImageU8::Pointer img, double p) {
    p = std::clamp(p, 0.0, 1.0);
    const auto region = img->GetLargestPossibleRegion();
    const size_t N = region.GetNumberOfPixels();
    const unsigned char* buf = img->GetBufferPointer();

    // Istogramma 256
    std::array<unsigned long long, 256> hist{};
    hist.fill(0);
#pragma omp parallel
    {
        std::array<unsigned long long, 256> hloc{};
        hloc.fill(0);
#pragma omp for nowait schedule(static)
        for (long i = 0; i < (long)N; ++i) hloc[buf[i]]++;
#pragma omp critical
        {
            for (int b = 0; b < 256; ++b) hist[b] += hloc[b];
        }
    }

    unsigned long long target = (unsigned long long)std::llround(p * (double)N);
    unsigned long long acc = 0;
    int thr = 0;
    for (int b = 0; b < 256; ++b) {
        acc += hist[b];
        if (acc >= target) {
            thr = b;
            break;
        }
    }

    auto mask = MaskU8::New();
    mask->SetRegions(region);
    mask->Allocate();
    unsigned char* out = mask->GetBufferPointer();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i) out[i] = (buf[i] > thr) ? 1 : 0;
    return mask;
}

}  // namespace

MultiResSubvolumeU8 selectSubvolumeUChar(
    itk::SmartPointer<itk::Image<unsigned char, 3>> fixed,
    itk::SmartPointer<itk::Image<unsigned char, 3>> moving,
    int Dz,
    double w1,
    double w2,
    double w3,
    double w4,
    double w5,
    int num_vols,
    int base_res,
    const std::string& maskMode,
    double pctValue,
    const std::string& gradientMode) {
    const auto region = fixed->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const int Z = static_cast<int>(size[2]);

    std::cout << "[U8] Using uint8 pipeline. Size: " << size[0] << "x" << size[1] << "x" << size[2] << "\n";
    std::cout << "Mask mode: " << maskMode;
    if (maskMode == "pct") std::cout << " (p=" << pctValue << ")";
    std::cout << "\nGradient mode: " << gradientMode << std::endl;

    // ------------------------
    // MASKS
    // ------------------------
    MaskU8::Pointer maskF, maskM;

    auto t_mask_start = Clock::now();
    if (maskMode == "zero") {
        maskF = makeMaskZeroU8(fixed);
        maskM = makeMaskZeroU8(moving);
    } else if (maskMode == "otsu") {
        maskF = makeMaskOtsuU8(fixed);
        maskM = makeMaskOtsuU8(moving);
    } else if (maskMode == "pct") {
        maskF = makeMaskPercentileU8(fixed, pctValue);
        maskM = makeMaskPercentileU8(moving, pctValue);
    } else {
        throw std::runtime_error("Unknown mask mode");
    }
    auto t_mask_end = Clock::now();
    std::cout << "[U8] Time mask = " << std::chrono::duration<double>(t_mask_end - t_mask_start).count() << " s\n";

    // ------------------------
    // INTERSECTION
    // ------------------------
    auto maskI = MaskU8::New();
    maskI->SetRegions(region);
    maskI->Allocate();

#pragma omp parallel for schedule(static)
    for (int z = 0; z < Z; z++)
        for (int y = 0; y < (int)size[1]; y++)
            for (int x = 0; x < (int)size[0]; x++) {
                ImageU8::IndexType idx = {x, y, z};
                maskI->SetPixel(idx, maskF->GetPixel(idx) & maskM->GetPixel(idx));
            }

    // ------------------------
    // GRADIENT
    // ------------------------
    GradU8::Pointer gradF, gradM;

    if (gradientMode == "itk") {
        using GradFilterU8 = itk::GradientMagnitudeImageFilter<ImageU8, GradU8>;
        auto gfF = GradFilterU8::New();
        auto gfM = GradFilterU8::New();
        gfF->SetInput(fixed);
        gfM->SetInput(moving);
        gfF->Update();
        gradF = gfF->GetOutput();
        gfM->Update();
        gradM = gfM->GetOutput();
    } else {
        if (gradientMode == "manual") {
            auto computeGradientMagnitudeFastU8 = [](ImageU8::Pointer input) -> ImageU8::Pointer {
                const auto region = input->GetLargestPossibleRegion();
                const auto size = region.GetSize();
                const size_t nx = size[0];
                const size_t ny = size[1];
                const size_t nz = size[2];
                const size_t sliceSize = nx * ny;

                auto output = ImageU8::New();
                output->SetRegions(region);
                output->CopyInformation(input);
                output->Allocate();

                const unsigned char* in = input->GetBufferPointer();
                unsigned char* out = output->GetBufferPointer();

#pragma omp parallel for schedule(static)
                for (long z = 1; z < static_cast<long>(nz) - 1; ++z) {
                    for (size_t y = 1; y < ny - 1; ++y) {
                        const size_t rowOffset = z * sliceSize + y * nx;
                        for (size_t x = 1; x < nx - 1; ++x) {
                            const size_t idx = rowOffset + x;
                            int gx = int(in[idx + 1]) - int(in[idx - 1]);
                            int gy = int(in[idx + nx]) - int(in[idx - nx]);
                            int gz = int(in[idx + sliceSize]) - int(in[idx - sliceSize]);
                            float mag = std::sqrt(float(gx * gx + gy * gy + gz * gz));
                            mag = std::min(mag, 255.0f);
                            out[idx] = static_cast<unsigned char>(mag);
                        }
                    }
                }
                return output;
            };

            auto t_grad_manual_start = Clock::now();
            auto futF = std::async(std::launch::async, [&] { return computeGradientMagnitudeFastU8(fixed); });
            auto futM = std::async(std::launch::async, [&] { return computeGradientMagnitudeFastU8(moving); });
            gradF = futF.get();
            gradM = futM.get();
            auto t_grad_manual_end = Clock::now();
            std::cout << "[U8] Time to compute gradients (manual OpenMP): "
                      << std::chrono::duration<double>(t_grad_manual_end - t_grad_manual_start).count() << " s\n";
        } else {
            throw std::runtime_error("Unknown gradient mode");
        }
    }

    // ------------------------
    // PREFIX SUMS
    // ------------------------
    auto psF = prefixZSumU8(maskF.GetPointer());
    auto psM = prefixZSumU8(maskM.GetPointer());
    auto psI = prefixZSumU8(maskI.GetPointer());
    auto psGF = prefixZSumU8(gradF.GetPointer());
    auto psGM = prefixZSumU8(gradM.GetPointer());

    // ------------------------
    // SLIDING WINDOW + CSV
    // ------------------------
    const double sliceVox = double(size[0]) * double(size[1]);
    const int maxZ = Z - Dz;
    const double min_intersection_ratio = 0.03;

    double bestScore = -1e300;
    int best_z0 = 0;

    std::ofstream csv("score_debug.csv");
    csv << "z0,fgF,fgM,gfn,gmn,MI,Score\n";

#pragma omp parallel
    {
        double localBestScore = -1e300;
        int localBestZ0 = 0;

#pragma omp for schedule(dynamic)
        for (int z0 = 0; z0 <= maxZ; z0++) {
            int z1 = z0 + Dz;

            double nzF = psF[z1] - psF[z0];
            double nzM = psM[z1] - psM[z0];
            double nzI = psI[z1] - psI[z0];

            double inter = nzI / (Dz * sliceVox);
            if (inter < min_intersection_ratio) continue;

            double fgF = nzF / (Dz * sliceVox);
            double fgM = nzM / (Dz * sliceVox);

            double gF = psGF[z1] - psGF[z0];
            double gM = psGM[z1] - psGM[z0];

            double gfn = gF / (Dz * sliceVox);
            double gmn = gM / (Dz * sliceVox);

            // ----- MI -----
            double MI = 0.0;
            if (w5 != 0.0) {
                auto blkF = extractBlockRawU8(fixed, z0, Dz);
                auto blkM = extractBlockRawU8(moving, z0, Dz);
                MI = mutual_information_u8(blkF.data(), blkM.data(), Dz, size[1], size[0]);
            }

            double score = w1 * fgF + w2 * fgM + w3 * gfn + w4 * gmn + w5 * MI;

#pragma omp critical(write_csv)
            csv << z0 << "," << fgF << "," << fgM << "," << gfn << "," << gmn << "," << MI << "," << score << "\n";

            if (score > localBestScore) {
                localBestScore = score;
                localBestZ0 = z0;
            }
        }

#pragma omp critical(best_reduce)
        {
            if (localBestScore > bestScore) {
                bestScore = localBestScore;
                best_z0 = localBestZ0;
            }
        }
    }

    csv.close();
    int best_z1 = best_z0 + Dz;

    std::cout << "[U8] Best Z range = [" << best_z0 << ", " << (best_z1 - 1) << "]  Dz=" << Dz << "\n";
    // ========================================================================
    // SAVE SUBVOLUME TO CSV: [start;end]
    // ========================================================================
    //{
    //    // Build filename based on weights
    //    std::ostringstream fname;
    //    fname << "w1w2w3w4w5_" << w1 << "_" << w2 << "_" << w3 << "_" << w4 << "_" << w5 << ".csv";
    //
    //    std::ofstream out(fname.str(), std::ios::app);
    //    if (!out) {
    //        std::cerr << "Error: unable to write file " << fname.str() << "\n";
    //    } else {
    //        out << "[" << best_z0 << ";" << (best_z1 - 1) << "]\n";
    //        std::cout << "📁 Saved selected range in: " << fname.str() << "\n";
    //    }
    //}
    // ------------------------
    // EXTRACT SUBVOLUME
    // ------------------------
    auto slicesCT = extractSlicesU8(fixed, best_z0, Dz);
    auto slicesPET = extractSlicesU8(moving, best_z0, Dz);

    // ------------------------
    // PYRAMID BUILD
    // ------------------------
    MultiResSubvolumeU8 pyr;

    for (int level = 1; level <= num_vols; level++) {
        int resX = (level < num_vols ? base_res * level : size[0]);

        MultiResSubvolumeU8::Level L;

        if (level < num_vols) {
            auto rCT = resizeSlicesU8(slicesCT, resX);
            auto rPET = resizeSlicesU8(slicesPET, resX);
            L.ct = flattenSlicesU8_DL(rCT);
            L.pet = flattenSlicesU8_DL(rPET);
        } else {
            L.ct = flattenVolumeU83D_DL(slicesCT);
            L.pet = flattenVolumeU83D_DL(slicesPET);
        }

        pyr.levels.push_back(std::move(L));
    }

    return pyr;
}

MultiResSubvolumeU8 extract_fixed_range(
    itk::SmartPointer<itk::Image<unsigned char, 3>> fixed,
    itk::SmartPointer<itk::Image<unsigned char, 3>> moving,
    int zStart,  // inclusivo
    int zEnd,    // inclusivo
    int num_vols,
    int base_res) {
    using ImageU8 = itk::Image<unsigned char, 3>;

    const auto region = fixed->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const int Z = static_cast<int>(size[2]);

    std::cout << "[U8] Using uint8 fixed-range pipeline. Size: " << size[0] << "x" << size[1] << "x" << size[2] << "\n";

    // --- Controlli range ---
    if (zStart < 0 || zStart >= Z) {
        throw std::runtime_error("extract_fixed_range: zStart out of bounds.");
    }
    if (zEnd < 0 || zEnd >= Z) {
        throw std::runtime_error("extract_fixed_range: zEnd out of bounds.");
    }
    if (zEnd < zStart) {
        throw std::runtime_error("extract_fixed_range: zEnd < zStart.");
    }

    const int Dz = zEnd - zStart + 1;
    std::cout << "[U8] Fixed Z range = [" << zStart << ", " << zEnd << "]  (Dz=" << Dz << ")\n";

    // === Subvolume extraction ===
    auto t_extract_start = Clock::now();
    auto futCT = std::async(std::launch::async, [&] { return extractSlicesU8(fixed, zStart, Dz); });
    auto futPET = std::async(std::launch::async, [&] { return extractSlicesU8(moving, zStart, Dz); });
    auto slicesCT = futCT.get();
    auto slicesPET = futPET.get();
    auto t_extract_end_init = Clock::now();
    std::cout << "[U8] Time to extract slices (fixed range): "
              << std::chrono::duration<double>(t_extract_end_init - t_extract_start).count() << " s\n";

    const int origX = size[0];
    const int origY = size[1];

    if (origX != origY) {
        throw std::runtime_error("extract_fixed_range: Only square images are supported.");
    }

    // maximum number of levels possible with base_res step:
    int maxLevels = (origX / base_res) + 1;

    if (num_vols > maxLevels) {
        throw std::runtime_error(
            "extract_fixed_range: Too many levels requested for the given image size. "
            "Maximum allowed: " +
            std::to_string(maxLevels));
    }

    MultiResSubvolumeU8 pyr;
    auto tStart = Clock::now();

    for (int level = 1; level <= num_vols; ++level) {
        int resX;

        if (level < num_vols) {
            resX = base_res * level;
        } else {
            // last level = original image
            resX = origX;
        }

        MultiResSubvolumeU8::Level lvl;

        if (level < num_vols) {
            // ============================
            // INTERMEDIATE LEVELS: RESIZE
            // ============================
            auto ctFuture = std::async(std::launch::async, [&]() {
                auto resized = resizeSlicesU8(slicesCT, resX);
                return flattenSlicesU8_DL(resized);
            });

            auto petFuture = std::async(std::launch::async, [&]() {
                auto resized = resizeSlicesU8(slicesPET, resX);
                return flattenSlicesU8_DL(resized);
            });

            lvl.ct = ctFuture.get();
            lvl.pet = petFuture.get();
        } else {
            // ============================
            // LAST LEVEL: NO RESIZE!
            // ============================
            lvl.ct = flattenVolumeU83D_DL(slicesCT);
            lvl.pet = flattenVolumeU83D_DL(slicesPET);
        }

        pyr.levels.push_back(std::move(lvl));

        std::cout << "[Pyramid] Level " << level << " → " << resX << " x " << resX
                  << (level == num_vols ? " (original)" : "") << "\n";
    }

    auto tEnd = Clock::now();
    std::cout << "[U8] Time to resize+flatten slices (fixed range): "
              << std::chrono::duration<double>(tEnd - tStart).count() << " s\n";

    return pyr;
}