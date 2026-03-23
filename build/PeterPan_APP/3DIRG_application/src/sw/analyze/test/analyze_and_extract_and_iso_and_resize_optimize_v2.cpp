// Usage:
//   ./select_subvolume_itk --type=nii <fixed.nii.gz> <moving.nii.gz> <Dz> [--mask=zero|otsu|pct:0.02]
//   ./select_subvolume_itk --type=png <fixed_folder/> <moving_folder/> <Dz> [--mask=zero|otsu|pct:0.02]
//
// Example:
//   ./select_subvolume_itk --type=png ct_slices/ mr_slices/ 40 --mask=pct:0.02
//   ./select_subvolume_itk --type=nii ct.nii.gz mr.nii.gz 32 --mask=otsu

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

constexpr unsigned int Dim = 3;
using PixelType = float;

using ImageType = itk::Image<PixelType, Dim>;
using MaskType = itk::Image<unsigned char, Dim>;
using GradType = ImageType;
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
static MaskType::Pointer makeMaskZeroBased_v2(ImageType::Pointer img) {
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
static MaskType::Pointer makeMaskPercentile_v2(ImageType::Pointer img, double p = 0.02) {
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
static MaskType::Pointer makeMaskOtsu_v2(ImageType::Pointer img) {
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

// -------------------- Main --------------------
int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage:\n"
             << "  ./select_subvolume_itk --type=nii <fixed.nii.gz> <moving.nii.gz> <Dz> [--mask=...]\n"
             << "  ./select_subvolume_itk --type=png <fixed_folder/> <moving_folder/> <Dz> [--mask=...]\n";
        return 1;
    }

    // Use all cores for ITK internal threading (does not affect output)

    string typeArg = argv[1];
    if (typeArg.rfind("--type=", 0) != 0) {
        cerr << "First argument must be --type=nii or --type=png\n";
        return 1;
    }
    string imgType = typeArg.substr(7);

    string fixedPath = argv[2];
    string movingPath = argv[3];
    int Dz = stoi(argv[4]);

    ImageType::Pointer fixed = loadImage(fixedPath, imgType);
    ImageType::Pointer moving = loadImage(movingPath, imgType);

    // ---- Debug: save raw input volume as PNG slices (no processing applied) ----
    {
        const auto region = fixed->GetLargestPossibleRegion();
        const int totalZ = region.GetSize()[2];

        auto rawFixed = extractSlices(fixed, 0, totalZ);
        auto rawMoving = extractSlices(moving, 0, totalZ);

        saveSlicesAsPNG(rawFixed, "fixed_raw");
        saveSlicesAsPNG(rawMoving, "moving_raw");

        std::cout << "[DEBUG] Saved raw input slices into ./fixed_raw/ and ./moving_raw/" << std::endl;
    }

    const auto region = fixed->GetLargestPossibleRegion();
    const auto size = region.GetSize();

    if (size[2] < static_cast<unsigned>(Dz)) {
        cerr << "Dz is larger than number of slices.\n";
        return 1;
    }

    // ---- Parse mask mode ----
    string maskMode = "otsu";
    double pctValue = 0.02;

    if (argc >= 6) {
        string arg = argv[5];
        if (arg.rfind("--mask=", 0) == 0) {
            maskMode = arg.substr(7);
            if (maskMode.rfind("pct:", 0) == 0) {
                pctValue = stod(maskMode.substr(4));
                maskMode = "pct";
            }
        }
    }

    cout << "Mask mode: " << maskMode;
    if (maskMode == "pct") cout << " (p=" << pctValue << ")";
    cout << endl;

    auto t0 = Clock::now();

    auto t_mask_start = Clock::now();
    // ---- Build masks ----
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
        std::cerr << "Unknown mask mode: " << maskMode << "\n";
        return 1;
    }

    auto t_mask_end = Clock::now();
    double maskMs = std::chrono::duration<double>(t_mask_end - t_mask_start).count();
    cout << "Time to compute masks: " << maskMs << " s" << endl;

    auto t_maskI_start = Clock::now();

    // ---- Intersection mask (parallel) ----
    auto maskI = MaskType::New();
    maskI->SetRegions(region);
    maskI->Allocate();

    const int Z = size[2];

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
    double maskIMs = std::chrono::duration<double>(t_maskI_end - t_maskI_start).count();
    cout << "Time to compute intersection mask: " << maskIMs << " s" << endl;

    auto t_grad_start = Clock::now();
    // ---- Gradient magnitude (ITK, multithread + parallel F/M execution) ----
    using GradFilter = itk::GradientMagnitudeImageFilter<ImageType, GradType>;

    auto gfF = GradFilter::New();
    gfF->SetInput(fixed);
    // ITK 5.x: uses WorkUnits, not Threads
    gfF->SetNumberOfWorkUnits(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));

    auto gfM = GradFilter::New();
    gfM->SetInput(moving);
    gfM->SetNumberOfWorkUnits(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));

    // esegui i due Update() in parallelo
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
    double gradMs = std::chrono::duration<double>(t_grad_end - t_grad_start).count();
    std::cout << "Time to compute gradients: " << gradMs << " s" << std::endl;

    auto t_prefix_start = Clock::now();

    // ---- Prefix sums ----
    auto psF = prefixZSum_MT(maskF.GetPointer());
    auto psM = prefixZSum_MT(maskM.GetPointer());
    auto psI = prefixZSum_MT(maskI.GetPointer());
    auto psGF = prefixZSum_MT(gradF.GetPointer());
    auto psGM = prefixZSum_MT(gradM.GetPointer());
    auto t_prefix_end = Clock::now();
    double prefixMs = std::chrono::duration<double>(t_prefix_end - t_prefix_start).count();
    cout << "Time to compute prefix sums: " << prefixMs << " s" << endl;

    auto t_sliding_start = Clock::now();
    // ---- Sliding window search ----
    double bestScore = -1e300;
    int best_z0 = 0, best_z1 = Dz;

    const double sliceVox = static_cast<long double>(size[0]) * static_cast<long double>(size[1]);
    const double w1 = 1.0, w2 = 1.0, w3 = 0.5, w4 = 0.5;
    const double min_intersection_ratio = 0.03;

    int maxZ = static_cast<int>(size[2]) - Dz;

#pragma omp parallel
    {
        double localBestScore = -1e300;
        int localZ0 = 0, localZ1 = Dz;

#pragma omp for schedule(dynamic)
        for (int z0 = 0; z0 <= maxZ; ++z0) {
            int z1 = z0 + Dz;
            double nzF = psF[z1] - psF[z0];
            double nzM = psM[z1] - psM[z0];
            double nzI = psI[z1] - psI[z0];
            double gF = psGF[z1] - psGF[z0];
            double gM = psGM[z1] - psGM[z0];

            double fgF = nzF / (Dz * sliceVox);
            double fgM = nzM / (Dz * sliceVox);
            double inter = nzI / (Dz * sliceVox);
            if (inter < min_intersection_ratio) continue;

            double gfn = gF / (Dz * sliceVox);
            double gmn = gM / (Dz * sliceVox);

            double score = w1 * fgF + w2 * fgM + w3 * gfn + w4 * gmn;

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
    double slidingMs = std::chrono::duration<double>(t_sliding_end - t_sliding_start).count();
    cout << "Time to compute best Z range: " << slidingMs << " s" << endl;
    auto t1 = Clock::now();
    double elapsedMs = std::chrono::duration<double>(t1 - t0).count();

    cout << "Best Z range = [" << best_z0 << ", " << best_z1 - 1 << "]  (Dz=" << Dz << ")\n";
    cout << "Time to compute best range: " << elapsedMs << " s" << endl;

    ofstream out("selected_range.txt");
    out << best_z0 << " " << best_z1 << endl;

    cout << "Extracting slices..." << endl;
    auto t2 = Clock::now();
    auto slicesFixed = extractSlices(fixed, best_z0, Dz);
    auto slicesMoving = extractSlices(moving, best_z0, Dz);
    auto t3 = Clock::now();
    double extractMs = std::chrono::duration<double>(t3 - t2).count();
    cout << "Time to extract slices: " << extractMs << " s" << endl;

    cout << "Resampling slices to isotropic 1.0 mm..." << endl;
    auto t4 = Clock::now();
    auto slicesFixedIso = resampleSlicesToIso1mm(slicesFixed);
    auto slicesMovingIso = resampleSlicesToIso1mm(slicesMoving);
    auto t5 = Clock::now();
    double isoMs = std::chrono::duration<double>(t5 - t4).count();
    cout << "Time to resample isotropic: " << isoMs << " s" << endl;

    int DzPx = Dz;
    int Dz2Px = Dz * 2;

    auto t6 = Clock::now();

    // Limita i thread OpenMP per evitare oversubscription
    int hw = std::thread::hardware_concurrency();
    int ompPerTask = std::max(1, hw / 4);
    omp_set_num_threads(ompPerTask);

    auto f1 = std::async(std::launch::async, [&] { return resizeSlicesToNxN(slicesFixedIso, DzPx); });
    auto f2 = std::async(std::launch::async, [&] { return resizeSlicesToNxN(slicesMovingIso, DzPx); });
    auto f3 = std::async(std::launch::async, [&] { return resizeSlicesToNxN(slicesFixedIso, Dz2Px); });
    auto f4 = std::async(std::launch::async, [&] { return resizeSlicesToNxN(slicesMovingIso, Dz2Px); });

    // <-- qui avviene l'attesa implicita (join)
    auto slicesFixedDz = f1.get();
    auto slicesMovingDz = f2.get();
    auto slicesFixed2Dz = f3.get();
    auto slicesMoving2Dz = f4.get();

    auto t7 = Clock::now();
    double resizeMs = std::chrono::duration<double>(t7 - t6).count();
    cout << "Time to parallel-resize: " << resizeMs << " s" << endl;

    cout << "Saving slices..." << endl;
    saveSlicesAsPNG(slicesFixed, "fixed_sub");
    saveSlicesAsPNG(slicesMoving, "moving_sub");
    saveSlicesAsPNG(slicesFixedIso, "fixed_iso");
    saveSlicesAsPNG(slicesMovingIso, "moving_iso");
    saveSlicesAsPNG(slicesFixedDz, "fixed_iso_Dz");
    saveSlicesAsPNG(slicesMovingDz, "moving_iso_Dz");
    saveSlicesAsPNG(slicesFixed2Dz, "fixed_iso_2Dz");
    saveSlicesAsPNG(slicesMoving2Dz, "moving_iso_2Dz");

    cout << "Saved " << Dz << " slices to ./fixed_sub/, ./moving_sub/, ./fixed_iso/, ./moving_iso/" << endl;

    auto total_dimMs = extractMs + isoMs + resizeMs + elapsedMs;
    cout << "Total processing time: " << total_dimMs << " s" << endl;
    return 0;
}
