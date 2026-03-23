#pragma once
#include <itkImage.h>

#include <cstdint>
#include <string>
#include <vector>

constexpr unsigned int Dim = 3;
using PixelType = float;

using ImageType = itk::Image<PixelType, Dim>;
using MaskType = itk::Image<unsigned char, Dim>;
using GradType = ImageType;
using SliceType = itk::Image<PixelType, 2>;
using SliceUChar = itk::Image<unsigned char, 2>;

// --- Loading and mask functions ---
ImageType::Pointer loadImage(const std::string& path, const std::string& type);
MaskType::Pointer makeMaskOtsu_v2(ImageType::Pointer img);
MaskType::Pointer makeMaskZeroBased_v2(ImageType::Pointer img);
MaskType::Pointer makeMaskPercentile_v2(ImageType::Pointer img, double pctValue);
// --- Template function declared here ---
template <typename TImage>
std::vector<long double> prefixZSum_MT(TImage* img);

// --- Explicit instantiations ---
extern template std::vector<long double> prefixZSum_MT<ImageType>(ImageType*);
extern template std::vector<long double> prefixZSum_MT<MaskType>(MaskType*);
extern template std::vector<long double> prefixZSum_MT<GradType>(GradType*);
// --- Processing and utility functions ---

std::vector<SliceType::Pointer> extractSlices(ImageType::Pointer img, int z0, int Dz);

void saveSlicesAsPNG(const std::vector<SliceType::Pointer>& slices, const std::string& folder);
void saveSlicesAsPNG(const std::vector<SliceUChar::Pointer>& slices, const std::string& folder);

std::vector<SliceType::Pointer> resampleSlicesToIso1mm(const std::vector<SliceType::Pointer>& slices);
std::vector<SliceType::Pointer> resizeSlicesToNxN(const std::vector<SliceType::Pointer>& slices, int N);
std::vector<uint8_t> slicesToUint8Volume(const std::vector<SliceType::Pointer>& slices);

// ------------------------------------------------------------- UINT 8 VERSION
// --- Additional types and functions for uint8 pipeline ---
using ImageTypeU8 = itk::Image<unsigned char, 3>;
using SliceTypeU8 = itk::Image<unsigned char, 2>;
using MaskTypeU8 = itk::Image<unsigned char, 3>;

// Dedicated loader for 8-bit images
ImageTypeU8::Pointer loadImageU8(const std::string& inputPath, const std::string& type);

// Core utilities (uint8 versions)
itk::Image<unsigned char, 3>::Pointer extractSlicesU8(itk::Image<unsigned char, 3>::Pointer img, int z0, int Dz);
void saveSlicesAsPNG_U8(const std::vector<SliceTypeU8::Pointer>& slices, const std::string& folder);
std::vector<uint8_t> flattenSlicesU8(itk::Image<unsigned char, 3>::Pointer subvol);
std::vector<uint8_t> flattenSlicesU8(const std::vector<SliceUChar::Pointer>& slices);

std::vector<SliceUChar::Pointer> resizeSlicesU8(ImageTypeU8::Pointer input, int outSizePx);
std::vector<uint8_t> flattenSlicesU8_DL(const std::vector<SliceUChar::Pointer>& slices);
std::vector<uint8_t> flattenVolumeU83D_DL(itk::Image<unsigned char, 3>::Pointer vol);
