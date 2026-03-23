
#include "utils.hpp"

#include <itkCastImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageSeriesReader.h>
#include <itkJoinSeriesImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMultiThreaderBase.h>
#include <itkNumericSeriesFileNames.h>
#include <itkOtsuThresholdImageFilter.h>
#include <itkPNGImageIO.h>
#include <itkResampleImageFilter.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using Clock = std::chrono::high_resolution_clock;

using SliceType = itk::Image<PixelType, 2>;
using SliceUChar = itk::Image<unsigned char, 2>;

// -------------------- Resize slice to NxN pixels (preserving FOV) --------------------
SliceType::Pointer resizeSliceToPixels(SliceType::Pointer in, int outSizePx) {
    using ResampleFilter = itk::ResampleImageFilter<SliceType, SliceType>;
    using TransformType = itk::IdentityTransform<double, 2>;
    using InterpType = itk::LinearInterpolateImageFunction<SliceType, double>;

    auto inRegion = in->GetLargestPossibleRegion();
    auto inSize = inRegion.GetSize();
    auto inOrigin = in->GetOrigin();
    auto inDir = in->GetDirection();
    auto inSpacing = in->GetSpacing();

    if (outSizePx <= 0) {
        throw std::runtime_error("resizeSliceToPixels: outSizePx must be > 0");
    }

    SliceType::SizeType outSize;
    outSize[0] = outSizePx;
    outSize[1] = outSizePx;

    // Adjust spacing so physical FOV stays the same
    SliceType::SpacingType outSpacing;
    outSpacing[0] = inSpacing[0] * static_cast<double>(inSize[0]) / static_cast<double>(outSizePx);
    outSpacing[1] = inSpacing[1] * static_cast<double>(inSize[1]) / static_cast<double>(outSizePx);

    auto transform = TransformType::New();
    transform->SetIdentity();

    auto interp = InterpType::New();
    interp->SetInputImage(in);

    auto resample = ResampleFilter::New();
    resample->SetInput(in);
    resample->SetTransform(transform);
    resample->SetInterpolator(interp);
    resample->SetOutputSpacing(outSpacing);
    resample->SetSize(outSize);
    resample->SetOutputOrigin(inOrigin);
    resample->SetOutputDirection(inDir);
    resample->SetDefaultPixelValue(0);
    resample->UpdateLargestPossibleRegion();

    // Deep copy (buffer only) to fully detach from pipeline
    SliceType::Pointer out = SliceType::New();
    out->SetRegions(resample->GetOutput()->GetLargestPossibleRegion());
    out->CopyInformation(resample->GetOutput());
    out->Allocate();
    std::memcpy(
        out->GetBufferPointer(),
        resample->GetOutput()->GetBufferPointer(),
        resample->GetOutput()->GetPixelContainer()->Size() * sizeof(PixelType));

    return out;
}

// -------------------- Batch resize to NxN (parallel) --------------------
std::vector<SliceType::Pointer> resizeSlicesToNxN(const std::vector<SliceType::Pointer>& slices, int outSizePx) {
    const size_t N = slices.size();
    std::vector<SliceType::Pointer> out(N);

#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(N); ++i) {
        out[i] = resizeSliceToPixels(slices[i], outSizePx);
    }
    return out;
}

// -------------------- Resample slice to isotropic 1.0mm spacing --------------------
SliceType::Pointer resampleSliceIsotropic1mm(SliceType::Pointer in) {
    using ResampleFilter = itk::ResampleImageFilter<SliceType, SliceType>;
    using TransformType = itk::IdentityTransform<double, 2>;
    using InterpType = itk::LinearInterpolateImageFunction<SliceType, double>;

    auto inRegion = in->GetLargestPossibleRegion();
    auto inSize = inRegion.GetSize();
    auto inSpacing = in->GetSpacing();
    auto inOrigin = in->GetOrigin();
    auto inDir = in->GetDirection();

    const double outSpacingVal = 1.0;
    SliceType::SpacingType outSpacing;
    outSpacing[0] = outSpacingVal;
    outSpacing[1] = outSpacingVal;

    SliceType::SizeType outSize;
    outSize[0] = static_cast<size_t>(std::round(inSize[0] * (inSpacing[0] / outSpacing[0])));
    outSize[1] = static_cast<size_t>(std::round(inSize[1] * (inSpacing[1] / outSpacing[1])));
    if (outSize[0] < 1) outSize[0] = 1;
    if (outSize[1] < 1) outSize[1] = 1;

    auto transform = TransformType::New();
    transform->SetIdentity();

    auto interp = InterpType::New();
    interp->SetInputImage(in);

    auto resample = ResampleFilter::New();
    resample->SetInput(in);
    resample->SetTransform(transform);
    resample->SetInterpolator(interp);
    resample->SetOutputSpacing(outSpacing);
    resample->SetSize(outSize);
    resample->SetOutputOrigin(inOrigin);
    resample->SetOutputDirection(inDir);
    resample->SetDefaultPixelValue(0);
    resample->UpdateLargestPossibleRegion();

    // Deep copy (buffer only) to fully detach from pipeline
    SliceType::Pointer out = SliceType::New();
    out->SetRegions(resample->GetOutput()->GetLargestPossibleRegion());
    out->CopyInformation(resample->GetOutput());
    out->Allocate();
    std::memcpy(
        out->GetBufferPointer(),
        resample->GetOutput()->GetBufferPointer(),
        resample->GetOutput()->GetPixelContainer()->Size() * sizeof(PixelType));

    return out;
}

// -------------------- Batch isotropic resample (parallel) --------------------
std::vector<SliceType::Pointer> resampleSlicesToIso1mm(const std::vector<SliceType::Pointer>& slices) {
    const size_t N = slices.size();
    std::vector<SliceType::Pointer> out(N);

#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(N); ++i) {
        out[i] = resampleSliceIsotropic1mm(slices[i]);
    }
    return out;
}

// -------------------- Extract a list of 2D slices from a 3D image --------------------
std::vector<SliceType::Pointer> extractSlices(ImageType::Pointer img, int z0, int Dz) {
    std::vector<SliceType::Pointer> slices(Dz);  // pre-size

    const auto region = img->GetLargestPossibleRegion();
    const auto size = region.GetSize();

    if (z0 < 0 || z0 + Dz > static_cast<int>(size[2])) {
        throw std::runtime_error("extractSlices: requested range out of bounds");
    }

    for (int i = 0; i < Dz; ++i) {
        const int z = z0 + i;

        ImageType::IndexType start;
        start[0] = 0;
        start[1] = 0;
        start[2] = z;

        ImageType::SizeType sliceSize;
        sliceSize[0] = size[0];
        sliceSize[1] = size[1];
        sliceSize[2] = 0;

        ImageType::RegionType extractRegion;
        extractRegion.SetIndex(start);
        extractRegion.SetSize(sliceSize);

        using Extractor = itk::ExtractImageFilter<ImageType, SliceType>;
        auto extractor = Extractor::New();
        extractor->SetInput(img);
        extractor->SetExtractionRegion(extractRegion);
        extractor->SetDirectionCollapseToSubmatrix();
        extractor->UpdateLargestPossibleRegion();

        SliceType::Pointer slice = extractor->GetOutput();

        // Fast deep copy of the pixel buffer to detach from pipeline
        SliceType::Pointer sliceCopy = SliceType::New();
        sliceCopy->SetRegions(slice->GetLargestPossibleRegion());
        sliceCopy->CopyInformation(slice);
        sliceCopy->Allocate();
        std::memcpy(
            sliceCopy->GetBufferPointer(),
            slice->GetBufferPointer(),
            slice->GetPixelContainer()->Size() * sizeof(PixelType));

        slices[i] = sliceCopy;
    }

    return slices;
}

void saveSlicesAsPNG(const std::vector<SliceType::Pointer>& slices, const std::string& outFolder) {
    std::error_code ec;
    std::filesystem::create_directories(outFolder, ec);
    if (ec) throw std::runtime_error("Cannot create folder: " + outFolder);

#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)slices.size(); ++i) {
        auto img = slices[i];

        // compute min/max
        itk::ImageRegionConstIterator<SliceType> it(img, img->GetLargestPossibleRegion());
        float vmin = std::numeric_limits<float>::max();
        float vmax = -std::numeric_limits<float>::max();
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            float v = it.Get();
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        if (vmax == vmin) vmax = vmin + 1.0f;

        // scale to [0..255]
        using OutType = SliceUChar;
        OutType::Pointer out = OutType::New();
        out->SetRegions(img->GetLargestPossibleRegion());
        out->Allocate();

        itk::ImageRegionIterator<OutType> ot(out, out->GetLargestPossibleRegion());
        itk::ImageRegionConstIterator<SliceType> it2(img, img->GetLargestPossibleRegion());
        for (it2.GoToBegin(), ot.GoToBegin(); !it2.IsAtEnd(); ++it2, ++ot) {
            float v = (it2.Get() - vmin) / (vmax - vmin);
            v = std::clamp(v, 0.0f, 1.0f);
            ot.Set(static_cast<unsigned char>(v * 255.0f));
        }

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/slice_%03ld.png", outFolder.c_str(), i);

        using Writer = itk::ImageFileWriter<OutType>;
        auto writer = Writer::New();
        writer->SetFileName(filename);
        writer->SetInput(out);
        writer->Update();
    }
}
// -------------------- PNG stack loader --------------------
ImageType::Pointer loadPNGStack(const std::string& folderPath) {
    using ReaderType = itk::ImageSeriesReader<ImageType>;
    auto reader = ReaderType::New();

    std::vector<std::string> files;
    for (auto& f : std::filesystem::directory_iterator(folderPath)) {
        if (f.path().extension() == ".png") files.push_back(f.path().string());
    }

    if (files.empty()) {
        throw std::runtime_error("No PNG files found in " + folderPath);
    }

    std::sort(files.begin(), files.end());
    reader->SetFileNames(files);
    reader->Update();
    return reader->GetOutput();
}

// -------------------- NIfTI loader --------------------
ImageType::Pointer loadNifti(const std::string& filePath) {
    using Reader = itk::ImageFileReader<ImageType>;
    auto reader = Reader::New();
    reader->SetFileName(filePath);
    reader->Update();
    return reader->GetOutput();
}

// -------------------- Auto loader --------------------
ImageType::Pointer loadImage(const std::string& inputPath, const std::string& type) {
    if (type == "nii") {
        return loadNifti(inputPath);
    } else if (type == "png") {
        return loadPNGStack(inputPath);
    } else {
        throw std::runtime_error("Unknown type: " + type);
    }
}

// -------------------- Mask builders (unchanged) --------------------
MaskType::Pointer makeMaskZeroBased_v2(ImageType::Pointer img) {
    auto mask = MaskType::New();
    mask->SetRegions(img->GetLargestPossibleRegion());
    mask->Allocate();

    const size_t N = img->GetLargestPossibleRegion().GetNumberOfPixels();
    const float* in = img->GetBufferPointer();
    unsigned char* out = mask->GetBufferPointer();

#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i) out[i] = (in[i] != 0.0f) ? 1 : 0;

    return mask;
}
MaskType::Pointer makeMaskPercentile_v2(ImageType::Pointer img, double p = 0.02) {
    p = std::clamp(p, 0.0, 1.0);
    const auto region = img->GetLargestPossibleRegion();
    const size_t N = region.GetNumberOfPixels();
    const float* buf = img->GetBufferPointer();

    // 1) parallel min/max
    float vmin = std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();

#pragma omp parallel
    {
        float lmin = std::numeric_limits<float>::infinity();
        float lmax = -std::numeric_limits<float>::infinity();

#pragma omp for nowait schedule(static)
        for (long i = 0; i < (long)N; ++i) {
            float v = buf[i];
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        }
#pragma omp critical
        {
            if (lmin < vmin) vmin = lmin;
            if (lmax > vmax) vmax = lmax;
        }
    }

    if (!std::isfinite(vmin) || !std::isfinite(vmax) || vmax <= vmin) {
        auto mask = MaskType::New();
        mask->SetRegions(region);
        mask->Allocate();
        mask->FillBuffer(0);
        return mask;
    }

    // 2) parallel histogram
    constexpr int BINS = 4096;
    std::vector<unsigned long long> hist(BINS, 0);
    const double scale = (double)(BINS - 1) / (double)(vmax - vmin);

#pragma omp parallel
    {
        std::vector<unsigned long long> hloc(BINS, 0);
#pragma omp for nowait schedule(static)
        for (long i = 0; i < (long)N; ++i) {
            int b = (int)((buf[i] - vmin) * scale);
            if (b < 0)
                b = 0;
            else if (b >= BINS)
                b = BINS - 1;
            hloc[b]++;
        }
#pragma omp critical
        {
            for (int b = 0; b < BINS; ++b) hist[b] += hloc[b];
        }
    }

    // 3) find percentile bin
    unsigned long long target = (unsigned long long)std::llround(p * (double)N);
    unsigned long long acc = 0;
    int thrBin = 0;
    for (int b = 0; b < BINS; ++b) {
        acc += hist[b];
        if (acc >= target) {
            thrBin = b;
            break;
        }
    }

    const float thr = vmin + (float)thrBin / (float)(BINS - 1) * (vmax - vmin);

    // 4) thresholding
    auto mask = MaskType::New();
    mask->SetRegions(region);
    mask->Allocate();
    unsigned char* out = mask->GetBufferPointer();

#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i) out[i] = (buf[i] > thr) ? 1 : 0;

    return mask;
}
MaskType::Pointer makeMaskOtsu_v2(ImageType::Pointer img) {
    const auto region = img->GetLargestPossibleRegion();
    const size_t N = region.GetNumberOfPixels();
    const float* buf = img->GetBufferPointer();

    // 1) parallel min/max
    float vmin = std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();

#pragma omp parallel
    {
        float lmin = std::numeric_limits<float>::infinity();
        float lmax = -std::numeric_limits<float>::infinity();

#pragma omp for nowait schedule(static)
        for (long i = 0; i < (long)N; ++i) {
            float v = buf[i];
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        }
#pragma omp critical
        {
            if (lmin < vmin) vmin = lmin;
            if (lmax > vmax) vmax = lmax;
        }
    }

    if (!std::isfinite(vmin) || !std::isfinite(vmax) || vmax <= vmin) {
        auto mask = MaskType::New();
        mask->SetRegions(region);
        mask->Allocate();
        mask->FillBuffer(0);
        return mask;
    }

    // 2) histogram
    constexpr int BINS = 4096;
    std::vector<unsigned long long> hist(BINS, 0);
    const double scale = (double)(BINS - 1) / (double)(vmax - vmin);

#pragma omp parallel
    {
        std::vector<unsigned long long> hloc(BINS, 0);
#pragma omp for nowait schedule(static)
        for (long i = 0; i < (long)N; ++i) {
            int b = (int)((buf[i] - vmin) * scale);
            if (b < 0)
                b = 0;
            else if (b >= BINS)
                b = BINS - 1;
            hloc[b]++;
        }
#pragma omp critical
        {
            for (int b = 0; b < BINS; ++b) hist[b] += hloc[b];
        }
    }

    // 3) Otsu on histogram
    unsigned long long total = 0;
    for (auto v : hist) total += v;
    double sumAll = 0.0;
    for (int b = 0; b < BINS; ++b) sumAll += (double)b * (double)hist[b];

    unsigned long long wB = 0;
    double sumB = 0.0;
    double maxVar = -1.0;
    int thrBin = 0;

    for (int b = 0; b < BINS; ++b) {
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
            thrBin = b;
        }
    }

    const float thr = vmin + (float)thrBin / (float)(BINS - 1) * (vmax - vmin);

    // 4) thresholding
    auto mask = MaskType::New();
    mask->SetRegions(region);
    mask->Allocate();
    unsigned char* out = mask->GetBufferPointer();

#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i) out[i] = (buf[i] > thr) ? 1 : 0;

    return mask;
}

static MaskType::Pointer makeMaskZeroBased(ImageType::Pointer img) {
    auto mask = MaskType::New();
    mask->SetRegions(img->GetLargestPossibleRegion());
    mask->Allocate();

    itk::ImageRegionIterator<ImageType> it(img, img->GetLargestPossibleRegion());
    itk::ImageRegionIterator<MaskType> mt(mask, mask->GetLargestPossibleRegion());
    for (it.GoToBegin(), mt.GoToBegin(); !it.IsAtEnd(); ++it, ++mt) mt.Set(it.Get() != 0 ? 1 : 0);

    return mask;
}

static MaskType::Pointer makeMaskOtsu(ImageType::Pointer img) {
    using OtsuFilter = itk::OtsuThresholdImageFilter<ImageType, MaskType>;
    auto otsu = OtsuFilter::New();
    otsu->SetInput(img);
    otsu->SetInsideValue(0);
    otsu->SetOutsideValue(1);
    otsu->Update();
    return otsu->GetOutput();
}

static MaskType::Pointer makeMaskPercentile(ImageType::Pointer img, double p = 0.02) {
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;

    std::vector<float> values;
    values.reserve(img->GetLargestPossibleRegion().GetNumberOfPixels());

    itk::ImageRegionConstIterator<ImageType> it(img, img->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) values.push_back(it.Get());

    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>(p * values.size());
    if (idx >= values.size()) idx = values.size() - 1;
    float thr = values[idx];

    auto mask = MaskType::New();
    mask->SetRegions(img->GetLargestPossibleRegion());
    mask->Allocate();

    itk::ImageRegionIterator<MaskType> mt(mask, mask->GetLargestPossibleRegion());
    for (it.GoToBegin(), mt.GoToBegin(); !it.IsAtEnd(); ++it, ++mt) mt.Set(it.Get() > thr ? 1 : 0);

    return mask;
}

// -------------------- Z prefix sums (unchanged) --------------------
template <typename TImage>
std::vector<long double> prefixZSum(TImage* img) {
    const auto region = img->GetLargestPossibleRegion();
    const auto size = region.GetSize();

    using IndexType = typename TImage::IndexType;
    using IV = typename IndexType::IndexValueType;

    std::vector<long double> sums(size[2] + 1, 0.0L);

    for (IV z = 0; z < static_cast<IV>(size[2]); ++z) {
        long double s = 0.0L;
        for (IV y = 0; y < static_cast<IV>(size[1]); ++y)
            for (IV x = 0; x < static_cast<IV>(size[0]); ++x) {
                IndexType idx = {x, y, z};
                s += static_cast<long double>(img->GetPixel(idx));
            }
        sums[z + 1] = sums[z] + s;
    }
    return sums;
}

template <typename TImage>
std::vector<long double> prefixZSum_MT(TImage* img) {
    const auto region = img->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const size_t nx = size[0];
    const size_t ny = size[1];
    const size_t nz = size[2];

    const size_t sliceSize = nx * ny;

    // output prefix sums (nz+1)
    std::vector<long double> sums(nz + 1, 0.0L);

    // raw pointer (contiguous in memory)
    const auto* buf = img->GetBufferPointer();

    // temporary per-thread slice sums (no race)
    std::vector<long double> sliceSums(nz, 0.0L);

#pragma omp parallel for schedule(static)
    for (int z = 0; z < (int)nz; ++z) {
        const size_t offset = static_cast<size_t>(z) * sliceSize;
        long double acc = 0.0L;
        for (size_t i = 0; i < sliceSize; ++i) acc += static_cast<long double>(buf[offset + i]);
        sliceSums[z] = acc;
    }

    // serial prefix (very fast)
    for (size_t z = 0; z < nz; ++z) sums[z + 1] = sums[z] + sliceSums[z];

    return sums;
}

// -----------------------------------------------------------------------------
// Converts a vector of ITK slices (float) into a flattened uint8_t 3D buffer
// Each slice is normalized locally in [0..255]
// -----------------------------------------------------------------------------
std::vector<uint8_t> slicesToUint8Volume(const std::vector<SliceType::Pointer>& slices) {
    if (slices.empty()) return {};

    const int Dz = static_cast<int>(slices.size());
    const auto region = slices[0]->GetLargestPossibleRegion();
    const size_t Nslice = region.GetNumberOfPixels();
    const size_t Ntotal = static_cast<size_t>(Dz) * Nslice;

    std::vector<uint8_t> out(Ntotal);

    for (int z = 0; z < Dz; ++z) {
        const float* buf = slices[z]->GetBufferPointer();
        size_t offset = static_cast<size_t>(z) * Nslice;

        float vmin = std::numeric_limits<float>::max();
        float vmax = -std::numeric_limits<float>::max();

#pragma omp parallel for reduction(min : vmin) reduction(max : vmax) schedule(static)
        for (long i = 0; i < static_cast<long>(Nslice); ++i) {
            float v = buf[i];
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }

        if (vmax == vmin) vmax = vmin + 1.0f;

#pragma omp parallel for schedule(static)
        for (long i = 0; i < static_cast<long>(Nslice); ++i) {
            float norm = (buf[i] - vmin) / (vmax - vmin);
            if (norm < 0.f) norm = 0.f;
            if (norm > 1.f) norm = 1.f;
            out[offset + static_cast<size_t>(i)] = static_cast<uint8_t>(norm * 255.0f);
        }
    }

    return out;
}
void saveSlicesAsPNG(const std::vector<SliceUChar::Pointer>& slices, const std::string& outFolder) {
    std::error_code ec;
    std::filesystem::create_directories(outFolder, ec);
    if (ec) throw std::runtime_error("Cannot create folder: " + outFolder);

#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)slices.size(); ++i) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/slice_%03ld.png", outFolder.c_str(), i);

        using Writer = itk::ImageFileWriter<SliceUChar>;
        auto writer = Writer::New();
        writer->SetFileName(filename);
        writer->SetInput(slices[i]);
        writer->Update();
    }
}

template std::vector<long double> prefixZSum_MT<ImageType>(ImageType*);
template std::vector<long double> prefixZSum_MT<MaskType>(MaskType*);

// ================================================================
// ========== Direct uint8 loader (no conversion) ========
// ================================================================

using ImageTypeU8 = itk::Image<unsigned char, 3>;
using SliceTypeU8 = itk::Image<unsigned char, 2>;

// --- PNG stack loader (uint8) ---
static ImageTypeU8::Pointer loadPNGStackU8(const std::string& folderPath) {
    using ReaderType = itk::ImageSeriesReader<ImageTypeU8>;
    auto reader = ReaderType::New();

    std::vector<std::string> files;
    for (auto& f : std::filesystem::directory_iterator(folderPath)) {
        if (f.path().extension() == ".png") {
            files.push_back(f.path().string());
        }
    }

    if (files.empty()) {
        throw std::runtime_error("No PNG files found in " + folderPath);
    }

    // --- Numeric sort based on the number in the filename ---
    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
        auto extract_num = [](const std::string& s) -> int {
            std::string name = std::filesystem::path(s).stem().string();  // e.g.: "slice_12"
            // Find where trailing digits start
            size_t pos = name.find_last_not_of("0123456789");
            if (pos == std::string::npos || pos + 1 >= name.size()) return 0;
            try {
                return std::stoi(name.substr(pos + 1));
            } catch (...) {
                return 0;
            }
        };
        return extract_num(a) < extract_num(b);
    });

    reader->SetFileNames(files);
    reader->Update();
    return reader->GetOutput();
}

// --- NIfTI loader (uint8) ---
static ImageTypeU8::Pointer loadNiftiU8(const std::string& filePath) {
    using Reader = itk::ImageFileReader<ImageTypeU8>;
    auto reader = Reader::New();
    reader->SetFileName(filePath);
    reader->Update();
    return reader->GetOutput();
}

// --- Auto loader: decides based on "type" ---
ImageTypeU8::Pointer loadImageU8(const std::string& inputPath, const std::string& type) {
    if (type == "nii")
        return loadNiftiU8(inputPath);
    else if (type == "png")
        return loadPNGStackU8(inputPath);
    else
        throw std::runtime_error("Unknown type for uint8 load: " + type);
}

// ================================================================
// ========== Slice operations (uint8) =============================
// ================================================================
itk::Image<unsigned char, 3>::Pointer extractSlicesU8(itk::Image<unsigned char, 3>::Pointer img, int z0, int Dz) {
    const auto region = img->GetLargestPossibleRegion();
    const auto size = region.GetSize();

    if (z0 < 0 || z0 + Dz > static_cast<int>(size[2])) {
        throw std::runtime_error("extractSubvolumeU8: out of bounds");
    }

    // Define extraction region
    itk::Image<unsigned char, 3>::IndexType start;
    start.Fill(0);
    start[2] = z0;

    itk::Image<unsigned char, 3>::SizeType subSize;
    subSize[0] = size[0];
    subSize[1] = size[1];
    subSize[2] = Dz;

    itk::Image<unsigned char, 3>::RegionType extractRegion;
    extractRegion.SetIndex(start);
    extractRegion.SetSize(subSize);

    using ExtractFilter = itk::ExtractImageFilter<itk::Image<unsigned char, 3>, itk::Image<unsigned char, 3>>;

    auto extractor = ExtractFilter::New();
    extractor->SetInput(img);
    extractor->SetExtractionRegion(extractRegion);
    extractor->SetDirectionCollapseToSubmatrix();
    extractor->UpdateLargestPossibleRegion();

    // No memcpy: ITK returns only a view (shallow copy of the buffer)
    return extractor->GetOutput();
}

// --- Save uint8 slices without normalization ---
void saveSlicesAsPNG_U8(const std::vector<SliceTypeU8::Pointer>& slices, const std::string& outFolder) {
    std::error_code ec;
    std::filesystem::create_directories(outFolder, ec);
    if (ec) throw std::runtime_error("Cannot create folder: " + outFolder);

#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)slices.size(); ++i) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/slice_%03ld.png", outFolder.c_str(), i);

        using Writer = itk::ImageFileWriter<SliceTypeU8>;
        auto writer = Writer::New();
        writer->SetFileName(filename);
        writer->SetInput(slices[i]);
        writer->Update();
    }
}

std::vector<uint8_t> flattenSlicesU8(itk::Image<unsigned char, 3>::Pointer subvol) {
    const auto region = subvol->GetLargestPossibleRegion();
    const size_t N = region.GetNumberOfPixels();
    const unsigned char* src = subvol->GetBufferPointer();

    // Single contiguous copy
    std::vector<uint8_t> out(N);
    std::memcpy(out.data(), src, N * sizeof(uint8_t));
    return out;
}

std::vector<uint8_t> flattenSlicesU8(const std::vector<SliceUChar::Pointer>& slices) {
    if (slices.empty()) {
        return {};
    }

    // All slices must have the same dimensions
    const auto region = slices[0]->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const size_t slicePixels = static_cast<size_t>(size[0]) * size[1];
    const size_t numSlices = slices.size();

    std::vector<uint8_t> out(slicePixels * numSlices);

    // Contiguous copy of slices
    for (size_t z = 0; z < numSlices; ++z) {
        const unsigned char* src = slices[z]->GetBufferPointer();
        std::memcpy(out.data() + z * slicePixels, src, slicePixels * sizeof(uint8_t));
    }

    return out;
}

// -------------------- Resize slice to NxN pixels (preserving FOV) --------------------
SliceTypeU8::Pointer resizeSliceToPixels_U8(SliceTypeU8::Pointer in, int outSizePx) {
    using ResampleFilter = itk::ResampleImageFilter<SliceTypeU8, SliceTypeU8>;
    using TransformType = itk::IdentityTransform<double, 2>;
    using InterpType = itk::LinearInterpolateImageFunction<SliceTypeU8, double>;

    auto inRegion = in->GetLargestPossibleRegion();
    auto inSize = inRegion.GetSize();
    auto inOrigin = in->GetOrigin();
    auto inDir = in->GetDirection();
    auto inSpacing = in->GetSpacing();

    if (outSizePx <= 0) {
        throw std::runtime_error("resizeSliceToPixels_U8: outSizePx must be > 0");
    }

    SliceTypeU8::SizeType outSize;
    outSize[0] = outSizePx;
    outSize[1] = outSizePx;

    // Adjust spacing so the physical FOV stays constant
    SliceTypeU8::SpacingType outSpacing;
    outSpacing[0] = inSpacing[0] * static_cast<double>(inSize[0]) / static_cast<double>(outSizePx);
    outSpacing[1] = inSpacing[1] * static_cast<double>(inSize[1]) / static_cast<double>(outSizePx);

    auto transform = TransformType::New();
    transform->SetIdentity();

    auto interp = InterpType::New();
    interp->SetInputImage(in);

    auto resample = ResampleFilter::New();
    resample->SetInput(in);
    resample->SetTransform(transform);
    resample->SetInterpolator(interp);
    resample->SetOutputSpacing(outSpacing);
    resample->SetSize(outSize);
    resample->SetOutputOrigin(inOrigin);
    resample->SetOutputDirection(inDir);
    resample->SetDefaultPixelValue(0);
    resample->UpdateLargestPossibleRegion();

    // Deep copy (detached from ITK pipeline)
    SliceTypeU8::Pointer out = SliceTypeU8::New();
    out->SetRegions(resample->GetOutput()->GetLargestPossibleRegion());
    out->CopyInformation(resample->GetOutput());
    out->Allocate();
    std::memcpy(
        out->GetBufferPointer(),
        resample->GetOutput()->GetBufferPointer(),
        resample->GetOutput()->GetPixelContainer()->Size() * sizeof(unsigned char));

    return out;
}

// -------------------- Batch resize to NxN (parallel) --------------------
std::vector<SliceTypeU8::Pointer> resizeSlicesToNxN_U8(const std::vector<SliceTypeU8::Pointer>& slices, int outSizePx) {
    const size_t N = slices.size();
    std::vector<SliceTypeU8::Pointer> out(N);

#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(N); ++i) {
        out[i] = resizeSliceToPixels_U8(slices[i], outSizePx);
    }

    return out;
}
std::vector<SliceTypeU8::Pointer> resizeSlicesU8(ImageTypeU8::Pointer input, int outSizePx) {
    using ImageType3D = ImageTypeU8;
    using SliceType2D = itk::Image<unsigned char, 2>;
    using ExtractFilterType = itk::ExtractImageFilter<ImageType3D, SliceType2D>;
    using ResampleFilter = itk::ResampleImageFilter<SliceType2D, SliceType2D>;
    using TransformType = itk::IdentityTransform<double, 2>;
    using InterpType = itk::LinearInterpolateImageFunction<SliceType2D, double>;

    const auto region = input->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const auto index = region.GetIndex();
    const unsigned int numSlices = size[2];
    const long baseZ = static_cast<long>(index[2]);

    std::vector<SliceType2D::Pointer> resized(numSlices);
    resized.reserve(numSlices);

    // Allocate reusable resources (transform is identity, so it can be shared)
    auto transform = TransformType::New();
    transform->SetIdentity();

    for (long z = 0; z < static_cast<long>(numSlices); ++z) {
        // ---- Extract slice ----
        ImageType3D::IndexType start = {index[0], index[1], baseZ + z};
        ImageType3D::SizeType sliceSize = {size[0], size[1], 0};
        ImageType3D::RegionType extractRegion;
        extractRegion.SetIndex(start);
        extractRegion.SetSize(sliceSize);
        extractRegion.Crop(region);

        auto extractor = ExtractFilterType::New();
        extractor->SetInput(input);
        extractor->SetExtractionRegion(extractRegion);
        extractor->SetDirectionCollapseToSubmatrix();
        extractor->Update();

        auto inSlice = extractor->GetOutput();
        const auto inSize = inSlice->GetLargestPossibleRegion().GetSize();
        const auto inSpacing = inSlice->GetSpacing();

        // ---- Compute new spacing to preserve FOV ----
        SliceType2D::SizeType outSize;
        outSize[0] = outSizePx;
        outSize[1] = outSizePx;
        SliceType2D::SpacingType outSpacing;
        outSpacing[0] = inSpacing[0] * static_cast<double>(inSize[0]) / static_cast<double>(outSizePx);
        outSpacing[1] = inSpacing[1] * static_cast<double>(inSize[1]) / static_cast<double>(outSizePx);

        // ---- Resample (local) ----
        auto resampler = ResampleFilter::New();
        resampler->SetInput(inSlice);
        resampler->SetTransform(transform);
        resampler->SetInterpolator(InterpType::New());
        resampler->SetOutputSpacing(outSpacing);
        resampler->SetSize(outSize);
        resampler->SetOutputOrigin(inSlice->GetOrigin());
        resampler->SetOutputDirection(inSlice->GetDirection());
        resampler->SetDefaultPixelValue(0);
        resampler->Update();

        // ---- Deep copy detach (necessary for thread safety) ----
        auto out = SliceType2D::New();
        out->SetRegions(resampler->GetOutput()->GetLargestPossibleRegion());
        out->CopyInformation(resampler->GetOutput());
        out->Allocate();

        std::memcpy(
            out->GetBufferPointer(),
            resampler->GetOutput()->GetBufferPointer(),
            resampler->GetOutput()->GetPixelContainer()->Size() * sizeof(unsigned char));

        resized[z] = out;
    }

    return resized;
}

std::vector<uint8_t> flattenSlicesU8_DL(const std::vector<SliceUChar::Pointer>& slices) {
    if (slices.empty()) return {};

    const auto region = slices[0]->GetLargestPossibleRegion();
    const auto size = region.GetSize();

    const int n_row = static_cast<int>(size[1]);  // Y
    const int n_col = static_cast<int>(size[0]);  // X
    const int n_slices = static_cast<int>(slices.size());

    // ---- Padding ----
    const int n_slices_padded = ((n_slices + 31) / 32) * 32;

    const bool needs_padding = (n_slices != n_slices_padded);

    std::vector<uint8_t> out(static_cast<size_t>(n_row) * n_col * n_slices_padded, 0);

    for (int k = 0; k < n_slices; ++k) {
        const unsigned char* src = slices[k]->GetBufferPointer();

        for (int j = 0; j < n_col; ++j) {
            for (int i = 0; i < n_row; ++i) {
                const size_t itk_idx = static_cast<size_t>(j) + static_cast<size_t>(i) * n_col;

                const size_t out_idx = (static_cast<size_t>(i) * n_col + j) * n_slices_padded + static_cast<size_t>(k);

                out[out_idx] = src[itk_idx];
            }
        }
    }

    return out;
}

std::vector<uint8_t> flattenVolumeU83D_DL(itk::Image<unsigned char, 3>::Pointer vol) {
    using ImageU8 = itk::Image<unsigned char, 3>;

    const auto region = vol->GetLargestPossibleRegion();
    const auto size = region.GetSize();

    const int nx = static_cast<int>(size[0]);  // X
    const int ny = static_cast<int>(size[1]);  // Y
    const int nz = static_cast<int>(size[2]);  // Z (number of slices)

    const size_t sliceSize = static_cast<size_t>(nx) * ny;

    const unsigned char* src = vol->GetBufferPointer();

    // ---- Padding ----
    const int nz_padded = ((nz + 31) / 32) * 32;

    std::vector<uint8_t> out(static_cast<size_t>(nx) * ny * nz_padded, 0);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < nx; ++j) {
            for (int i = 0; i < ny; ++i) {
                const size_t itk_idx =
                    static_cast<size_t>(j) + static_cast<size_t>(i) * nx + static_cast<size_t>(k) * sliceSize;

                const size_t out_idx = (static_cast<size_t>(i) * nx + j) * nz_padded + static_cast<size_t>(k);

                out[out_idx] = src[itk_idx];
            }
        }
    }

    return out;
}