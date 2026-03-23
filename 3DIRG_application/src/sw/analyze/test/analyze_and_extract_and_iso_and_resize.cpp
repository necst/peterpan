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
#include <itkImageDuplicator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageSeriesReader.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNumericSeriesFileNames.h>
#include <itkOtsuThresholdImageFilter.h>
#include <itkPNGImageIO.h>
#include <itkResampleImageFilter.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
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

    // new size: NxN
    SliceType::SizeType outSize;
    outSize[0] = outSizePx;
    outSize[1] = outSizePx;

    // new physical spacing computed to preserve the same field of view
    SliceType::SpacingType outSpacing;
    outSpacing[0] = inSpacing[0] * (double)inSize[0] / (double)outSizePx;
    outSpacing[1] = inSpacing[1] * (double)inSize[1] / (double)outSizePx;

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

    // deep copy
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
std::vector<SliceType::Pointer> resizeSlicesToNxN(const std::vector<SliceType::Pointer>& slices, int outSizePx) {
    std::vector<SliceType::Pointer> out;
    out.reserve(slices.size());
    for (auto& s : slices) {
        out.push_back(resizeSliceToPixels(s, outSizePx));
    }
    return out;
}

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

    // deep copy del buffer per evitare dipendenze di pipeline
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

std::vector<SliceType::Pointer> resampleSlicesToIso1mm(const std::vector<SliceType::Pointer>& slices) {
    std::vector<SliceType::Pointer> out;
    out.reserve(slices.size());
    for (auto& s : slices) {
        out.push_back(resampleSliceIsotropic1mm(s));
    }
    return out;
}

// -------------------- Extract a list of 2D slices from a 3D image --------------------
std::vector<SliceType::Pointer> extractSlices(ImageType::Pointer img, int z0, int Dz) {
    std::vector<SliceType::Pointer> slices;
    slices.reserve(Dz);

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

        // ==== ✅ FAST DEEP COPY OF PIXEL BUFFER ONLY ====
        SliceType::Pointer sliceCopy = SliceType::New();
        sliceCopy->SetRegions(slice->GetLargestPossibleRegion());
        sliceCopy->CopyInformation(slice);
        sliceCopy->Allocate();

        std::memcpy(
            sliceCopy->GetBufferPointer(),
            slice->GetBufferPointer(),
            slice->GetPixelContainer()->Size() * sizeof(PixelType));
        // ===============================================

        slices.push_back(sliceCopy);
    }

    return slices;
}
// -------------------- Save a vector of 2D slices as PNG --------------------
void saveSlicesAsPNG(const std::vector<SliceType::Pointer>& slices, const std::string& outFolder) {
    std::error_code ec;
    std::filesystem::create_directories(outFolder, ec);

    if (ec) {
        throw std::runtime_error("Cannot create folder: " + outFolder + " (" + ec.message() + ")");
    }

    for (size_t i = 0; i < slices.size(); ++i) {
        using Caster = itk::CastImageFilter<SliceType, SliceUChar>;
        auto caster = Caster::New();
        caster->SetInput(slices[i]);
        caster->Update();

        char filename[256];
        std::snprintf(filename, sizeof(filename), "%s/slice_%03zu.png", outFolder.c_str(), i);

        using Writer = itk::ImageFileWriter<SliceUChar>;
        auto writer = Writer::New();
        writer->SetFileName(filename);
        writer->SetInput(caster->GetOutput());
        writer->Update();
    }
}

// -------------------- PNG stack loader --------------------
ImageType::Pointer loadPNGStack(const std::string& folderPath) {
    using ReaderType = itk::ImageSeriesReader<ImageType>;
    auto reader = ReaderType::New();

    vector<string> files;
    for (auto& f : std::filesystem::directory_iterator(folderPath)) {
        if (f.path().extension() == ".png") files.push_back(f.path().string());
    }

    if (files.empty()) {
        throw std::runtime_error("No PNG files found in " + folderPath);
    }

    sort(files.begin(), files.end());
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

    vector<float> values;
    values.reserve(img->GetLargestPossibleRegion().GetNumberOfPixels());

    itk::ImageRegionConstIterator<ImageType> it(img, img->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) values.push_back(it.Get());

    sort(values.begin(), values.end());
    size_t idx = (size_t)(p * values.size());
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
                s += (long double)img->GetPixel(idx);
            }
        sums[z + 1] = sums[z] + s;
    }
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

    const auto region = fixed->GetLargestPossibleRegion();
    const auto size = region.GetSize();

    if (size[2] < (unsigned)Dz) {
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

    // ---- Build masks ----
    MaskType::Pointer maskF, maskM;
    if (maskMode == "zero")
        maskF = makeMaskZeroBased(fixed), maskM = makeMaskZeroBased(moving);
    else if (maskMode == "otsu")
        maskF = makeMaskOtsu(fixed), maskM = makeMaskOtsu(moving);
    else if (maskMode == "pct")
        maskF = makeMaskPercentile(fixed, pctValue), maskM = makeMaskPercentile(moving, pctValue);
    else {
        cerr << "Unknown mask mode: " << maskMode << "\n";
        return 1;
    }

    // ---- Intersection mask ----
    auto maskI = MaskType::New();
    maskI->SetRegions(region);
    maskI->Allocate();
    {
        itk::ImageRegionIterator<MaskType> itF(maskF, region), itM(maskM, region), itI(maskI, region);
        for (itF.GoToBegin(), itM.GoToBegin(), itI.GoToBegin(); !itF.IsAtEnd(); ++itF, ++itM, ++itI)
            itI.Set(itF.Get() & itM.Get());
    }

    // ---- Gradient magnitude ----
    using GradFilter = itk::GradientMagnitudeImageFilter<ImageType, GradType>;
    auto gfF = GradFilter::New();
    gfF->SetInput(fixed);
    gfF->Update();
    auto gfM = GradFilter::New();
    gfM->SetInput(moving);
    gfM->Update();
    GradType::Pointer gradF = gfF->GetOutput();
    GradType::Pointer gradM = gfM->GetOutput();

    // ---- Prefix sums ----
    auto psF = prefixZSum(maskF.GetPointer());
    auto psM = prefixZSum(maskM.GetPointer());
    auto psI = prefixZSum(maskI.GetPointer());
    auto psGF = prefixZSum(gradF.GetPointer());
    auto psGM = prefixZSum(gradM.GetPointer());

    // ---- Sliding window search ----
    long double bestScore = -1e300;
    int best_z0 = 0, best_z1 = Dz;

    const long double sliceVox = (long double)size[0] * size[1];
    const double w1 = 1.0, w2 = 1.0, w3 = 0.5, w4 = 0.5;
    const double min_intersection_ratio = 0.03;

    int maxZ = (int)size[2] - Dz;

#pragma omp parallel
    {
        long double localBestScore = -1e300;
        int localZ0 = 0, localZ1 = Dz;

#pragma omp for schedule(dynamic)
        for (int z0 = 0; z0 <= maxZ; ++z0) {
            int z1 = z0 + Dz;
            long double nzF = psF[z1] - psF[z0];
            long double nzM = psM[z1] - psM[z0];
            long double nzI = psI[z1] - psI[z0];
            long double gF = psGF[z1] - psGF[z0];
            long double gM = psGM[z1] - psGM[z0];

            long double fgF = nzF / (Dz * sliceVox);
            long double fgM = nzM / (Dz * sliceVox);
            long double inter = nzI / (Dz * sliceVox);
            if (inter < min_intersection_ratio) continue;

            long double gfn = gF / (Dz * sliceVox);
            long double gmn = gM / (Dz * sliceVox);

            long double score = w1 * fgF + w2 * fgM + w3 * gfn + w4 * gmn;

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
    auto slicesFixedDz = resizeSlicesToNxN(slicesFixedIso, DzPx);
    auto slicesMovingDz = resizeSlicesToNxN(slicesMovingIso, DzPx);
    auto slicesFixed2Dz = resizeSlicesToNxN(slicesFixedIso, Dz2Px);
    auto slicesMoving2Dz = resizeSlicesToNxN(slicesMovingIso, Dz2Px);
    auto t7 = Clock::now();
    double resizeMs = std::chrono::duration<double>(t7 - t6).count();
    cout << "Time to resize to Dz x Dz: " << resizeMs << " s" << endl;

    saveSlicesAsPNG(slicesFixed, "fixed_sub");
    saveSlicesAsPNG(slicesMoving, "moving_sub");
    saveSlicesAsPNG(slicesFixedIso, "fixed_iso");
    saveSlicesAsPNG(slicesMovingIso, "moving_iso");
    saveSlicesAsPNG(slicesFixedDz, "fixed_iso_Dz");
    saveSlicesAsPNG(slicesMovingDz, "moving_iso_Dz");
    saveSlicesAsPNG(slicesFixed2Dz, "fixed_iso_2Dz");
    saveSlicesAsPNG(slicesMoving2Dz, "moving_iso_2Dz");

    cout << "Saved " << Dz << " slices to ./fixed_sub/, ./moving_sub/, ./fixed_iso/, ./moving_iso/" << endl;

    // somma i vari resize/preprocess time
    auto total_dimMs = extractMs + isoMs + resizeMs + elapsedMs;
    cout << "Total processing time: " << total_dimMs << " s" << endl;
    return 0;
}
