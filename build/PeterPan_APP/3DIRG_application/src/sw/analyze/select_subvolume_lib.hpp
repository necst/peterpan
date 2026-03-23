#pragma once
#include <itkImage.h>
#include <itkSmartPointer.h>

#include <string>
#include <utility>
#include <vector>

struct MultiResSubvolumeU8 {
    struct Level {
        std::vector<uint8_t> ct;
        std::vector<uint8_t> pet;
    };

    std::vector<Level> levels;  // levels[0] = finest, levels[N-1] = coarsest
};
// Note: these aliases (ImageType, MaskType, GradType, SliceType, etc.)
// are already defined in utils.hpp (float path). Here we include only
// forward declarations of our public APIs.

// ----------------------------------------------------------
// API: "classic" float pipeline
//   - finds best range
//   - extracts subvolume
//   - returns flattened uint8 buffers (for CT and PET)
// ----------------------------------------------------------
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> selectSubvolumeFloat(
    const std::string& ctPath,
    const std::string& petPath,
    int Dz,
    double w1,                    // weight fgF (fixed foreground)
    double w2,                    // weight fgM (moving foreground)
    double w3,                    // weight gfn (fixed gradient)
    double w4,                    // weight gmn (moving gradient)
    const std::string& maskMode,  // "otsu" | "zero" | "pct"
    double pctValue,              // used if maskMode == "pct"
    const std::string& type       // "png" | "nii"
);

// ----------------------------------------------------------
// API: "native" uint8 pipeline
//   - same logic (best range, extraction)
//   - no float conversion; everything is unsigned char
//   - type: "uint8"|"png8"|"nii8" (o "png"|"nii" se sai che sono 8-bit)
MultiResSubvolumeU8 selectSubvolumeUChar(
    itk::SmartPointer<itk::Image<unsigned char, 3>> fixed,
    itk::SmartPointer<itk::Image<unsigned char, 3>> moving,
    int Dz,
    double w1,
    double w2,
    double w3,
    double w4,
    double w5,
    const int num_vols,
    const int base_res,
    const std::string& maskMode,
    double pctValue,
    const std::string& gradientMode = "itk");

// ----------------------------------------------------------
// API: uint8 pipeline with "fixed range"
//   - does NOT search for best range
//   - directly uses the [zStart, zEnd] range (slice indices)
//   - same extraction/multires logic as selectSubvolumeUChar
// ----------------------------------------------------------
MultiResSubvolumeU8 extract_fixed_range(
    itk::SmartPointer<itk::Image<unsigned char, 3>> fixed,
    itk::SmartPointer<itk::Image<unsigned char, 3>> moving,
    int zStart,  // initial slice index (inclusive)
    int zEnd,    // final slice index (inclusive)
    const int num_vols,
    const int base_res);