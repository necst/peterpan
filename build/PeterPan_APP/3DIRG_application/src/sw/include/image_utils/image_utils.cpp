/*
MIT License

Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide
Conficconi, Eleonora D'Arnese

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "image_utils.hpp"

int cache_hits = 0;
int cache_misses = 0;
int cache_size = 0;

void get_cache_stats(int& hits, int& misses, int& size) {
    hits = cache_hits;
    misses = cache_misses;
    size = cache_size;
}

void reset_cache_stats() {
    cache_hits = 0;
    cache_misses = 0;
    cache_size = 0;
}

#define CACHE_STRATEGY_SIGNATURE2                                                                      \
    uint8_t *source, int Pi, int Pj, int i, int j, int k, int index, int n_row, int n_col, int LAYERS, \
        bool can_overwrite
typedef uint8_t (*CacheStrategy2)(CACHE_STRATEGY_SIGNATURE2);
#define NO_CACHING2 NULL

uint8_t read_from_cache(CACHE_STRATEGY_SIGNATURE2, CacheStrategy2 cache_strategy = NO_CACHING2) {
    if (index == -1) return 0;
    if (k > 0 || cache_strategy == NO_CACHING2) return source[index];

    return cache_strategy(source, Pi, Pj, i, j, k, index, n_row, n_col, LAYERS, can_overwrite);
}

// macro for the function signature of a caching strategy with name X
#define CACHE_STRATEGY_DEF2(X) uint8_t cache_strategy2_##X(CACHE_STRATEGY_SIGNATURE2)
#define CACHE_STRATEGY2(X) &cache_strategy2_##X

// ---------- caching strategies (n_row, n_col) ----------

// just for testing
CACHE_STRATEGY_DEF2(dummy) { return source[index]; }

// like that paper
// (https://sfat.massey.ac.nz/research/centres/crisp/pdfs/2004_DELTA_126.pdf)
CACHE_STRATEGY_DEF2(GribbonBailey) {
    struct CacheElem {
        int y;
        uint8_t value;
    };
    thread_local static std::unordered_map<int, CacheElem> cache;

    bool hit = (cache.find(Pi) != cache.end() && cache[Pi].y == Pj);

    if (hit) {
        cache_hits++;
        return cache[Pi].value;
    }

    cache_misses++;
    uint8_t new_value = source[index];
    if (can_overwrite) cache[Pi] = {Pj, new_value};
    return new_value;
}

template <class T>
int compute_buffer_offset(const int n_row, const int n_col, const int LAYERS, const T i, const T j, const T k) {
#ifndef USE_OLD_FORMAT
    // NEW FORMAT: row-major sui pixel
    // linear = (i * n_col + j) * LAYERS + k
    return std::round(i * (T)(n_col * LAYERS) + j * (T)LAYERS + k);
#else
    // OLD FORMAT invariato
    return std::round(k * (T)(n_row * n_col) + j * (T)n_row + i);
#endif
}

int transform_coords(
    const int n_row,
    const int n_col,
    const int LAYERS,
    const int TX,
    const int TY,
    const float ANG,
    const int i,
    const int j,
    const int k);

inline uint8_t transform_nearest_neighbour(
    uint8_t* volume_src,
    const int TX,
    const int TY,
    const float ANG,
    const int n_row,
    const int n_col,
    const int LAYERS,
    const int i,
    const int j,
    const int k) {
    // compute source index (transform [i,j] coordinates)
    int old_index = transform_coords(n_row, n_col, LAYERS, TX, TY, ANG, i, j, k);

    // read pixel from input volume
#ifndef TRACK_READS
    uint8_t pixel = (old_index != -1 ? volume_src[old_index] : 0);
#else
    uint8_t pixel = (old_index != -1 ? track_reads(volume_src, old_index) : 0);  // count sequential reads for stats
#endif

    return pixel;
}

// NOTE: assuming that k is always in bounds
inline bool is_out_of_bounds(const int n_row, const int n_col, const int LAYERS, const float i, const float j) {
    return (i < 0 || i >= n_row || j < 0 || j >= n_col);
}

inline uint8_t transform_bilinear(
    uint8_t* volume_src,
    const float TX,
    const float TY,
    const float ANG,
    const int n_row,
    const int n_col,
    const int LAYERS,
    const int i,
    const int j,
    const int k) {
    // centered coordinates
    const float ci = i - n_row / 2.f;
    const float cj = j - n_col / 2.f;

    // counterclockwise rotation
    const float P_i = (ci - TY) * std::cos(ANG) - (cj - TX) * std::sin(ANG) + n_row / 2.f;
    const float P_j = (ci - TY) * std::sin(ANG) + (cj - TX) * std::cos(ANG) + n_col / 2.f;
    // positions of the 4 surrounding pixels
    const float P_left = std::floor(P_i);
    const float P_right = std::ceil(P_i);
    const float P_top = std::floor(P_j);
    const float P_bottom = std::ceil(P_j);

    const int Q11_index =
        (!is_out_of_bounds(n_row, n_col, LAYERS, P_left, P_top)
             ? compute_buffer_offset<int>(n_row, n_col, LAYERS, P_left, P_top, k)
             : -1);
    const int Q12_index =
        (!is_out_of_bounds(n_row, n_col, LAYERS, P_right, P_top)
             ? compute_buffer_offset<int>(n_row, n_col, LAYERS, P_right, P_top, k)
             : -1);
    const int Q21_index =
        (!is_out_of_bounds(n_row, n_col, LAYERS, P_left, P_bottom)
             ? compute_buffer_offset<int>(n_row, n_col, LAYERS, P_left, P_bottom, k)
             : -1);
    const int Q22_index =
        (!is_out_of_bounds(n_row, n_col, LAYERS, P_right, P_bottom)
             ? compute_buffer_offset<int>(n_row, n_col, LAYERS, P_right, P_bottom, k)
             : -1);

    const float Q11_val = (float)read_from_cache(
        volume_src, P_left, P_top, i, j, k, Q11_index, n_row, n_col, LAYERS, true, CACHE_STRATEGY2(GribbonBailey));
    const float Q12_val = (float)read_from_cache(
        volume_src, P_right, P_top, i, j, k, Q12_index, n_row, n_col, LAYERS, true, CACHE_STRATEGY2(GribbonBailey));
    const float Q21_val = (float)read_from_cache(
        volume_src, P_left, P_bottom, i, j, k, Q21_index, n_row, n_col, LAYERS, true, CACHE_STRATEGY2(GribbonBailey));
    const float Q22_val = (float)read_from_cache(
        volume_src, P_right, P_bottom, i, j, k, Q22_index, n_row, n_col, LAYERS, true, CACHE_STRATEGY2(GribbonBailey));

    const float R_i = P_i - P_left;
    const float R_j = P_j - P_top;
    const float R_i_inv = 1.f - R_i;
    const float R_j_inv = 1.f - R_j;

    const float val_left = Q11_val * R_i_inv + Q12_val * R_i;
    const float val_right = Q21_val * R_i_inv + Q22_val * R_i;
    const float pixel = std::round(val_left * R_j_inv + val_right * R_j);

    return pixel;
}

void transform_volume(
    uint8_t* volume_src,
    uint8_t* volume_dest,
    const float TX,
    const float TY,
    float ANG,
    const int n_row,
    const int n_col,
    const int LAYERS,
    const bool bilinear_interpolation) {
    ANG = -ANG;  // per convenzione, angolo positivo = rotazione
                 // antioraria
    float n_sin = -std::sin(ANG);
    float p_cos = std::cos(ANG);
    float p_sin = std::sin(ANG);
    float n_cos = -std::cos(ANG);

    float delta_row_fixed_row = p_sin;
    float delta_col_fixed_row = p_cos;

    float delta_row_fixed_col = p_cos;
    float delta_col_fixed_col = n_sin;

    float cc_first_tra =
        (0 - n_col / 2.f) * p_cos + (0 - n_row / 2.f) * n_sin + (n_col / 2.f) - TX - delta_col_fixed_row;
    float rr_first_tra =
        (0 - n_col / 2.f) * p_sin + (0 - n_row / 2.f) * p_cos + (n_row / 2.f) - TY - delta_row_fixed_row;

    float cc_tra = 0;
    float rr_tra = 0;

// Iterazione righe (i), colonne (j), layers (k)
#pragma omp parallel for collapse(3)
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            for (int k = 0; k < LAYERS; k++) {
                uint16_t pixel;
                if (bilinear_interpolation)
                    pixel = transform_bilinear(volume_src, TX, TY, ANG, n_row, n_col, LAYERS, i, j, k);
                else
                    pixel = transform_nearest_neighbour(volume_src, TX, TY, ANG, n_row, n_col, LAYERS, i, j, k);

#ifndef USE_OLD_FORMAT
                const int dest_index = i * n_col * LAYERS + j * LAYERS + k;
#else
                const int dest_index = k * n_row * n_col + j * n_row + i;
#endif

                volume_dest[dest_index] = pixel;
            }
        }
    }
}

// allows translation along x-y plane, and rotation around z-axis
int transform_coords(
    const int n_row,
    const int n_col,
    const int LAYERS,
    const int TX,
    const int TY,
    const float ANG,
    const int i,
    const int j,
    const int k) {
    // Coordinates centered on the image center
    float ci = i - n_row / 2.f;
    float cj = j - n_col / 2.f;

    float new_i_f = (ci - TY) * std::cos(ANG) - (cj - TX) * std::sin(ANG) + n_row / 2.f;
    float new_j_f = (ci - TY) * std::sin(ANG) + (cj - TX) * std::cos(ANG) + n_col / 2.f;

    float new_k_f = static_cast<float>(k);

    // Out-of-bounds check
    if (new_i_f < 0.f || new_i_f >= n_row || new_j_f < 0.f || new_j_f >= n_col || new_k_f < 0.f || new_k_f >= LAYERS) {
        return -1;  // invalid index
    }

#ifndef USE_FLOAT_INDEX
    // Use round to get the integer index
    int new_i = std::round(new_i_f);
    int new_j = std::round(new_j_f);
#else
    int new_i = static_cast<int>(new_i_f);
    int new_j = static_cast<int>(new_j_f);
#endif

    // Compute linear index in the buffer
    int out_index = compute_buffer_offset<int>(n_row, n_col, LAYERS, new_i, new_j, k);
    return out_index;
}

void write_slice_in_buffer(
    uint8_t* src, uint8_t* dest, const int slice_index, const int n_row, const int n_col, const int LAYERS) {
    for (int i = 0; i < n_row * n_col; i++) {
#ifndef USE_OLD_FORMAT
        const int dest_index = i * LAYERS + slice_index;  // new formula
#else
        const int dest_index = slice_index * n_row * n_col + i;  // old formula
#endif
        dest[dest_index] = src[i];
    }
}

void read_slice_from_buffer(
    uint8_t* src, uint8_t* dest, const int slice_index, const int n_row, const int n_col, const int LAYERS) {
    for (int i = 0; i < n_row * n_col; i++) {
#ifndef USE_OLD_FORMAT
        const int src_index = i * LAYERS + slice_index;  // new formula
#else
        const int src_index = slice_index * n_row * n_col + i;  // old formula
#endif
        dest[i] = src[src_index];
    }
}

/// Round up to next higher power of 2 (return x if it's already a power of 2).
inline unsigned int pow2roundup(unsigned int x) {
    if (x < 0) return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

inline uint8_t convertDepth8(uint32_t pixel, const int originalDepth) {
    const uint32_t maxPixelValue = ((int32_t)1 << originalDepth) - 1;
    const float normalized = pixel / (float)maxPixelValue;
    return std::round(255.0f * normalized);
}

int read_volume_from_file_DICOM(
    uint8_t* volume, const int N_ROW, const int N_COL, const int N_COUPLES, const std::string& path) {
#ifndef COMPILE_WITHOUT_DCMTK
    uint8_t* imgData = new uint8_t[N_ROW * N_COL];

    for (int i = 0; i < N_COUPLES; i++) {
        std::string s = path + "IM" + std::to_string(i) + ".dcm";
        DcmFileFormat fileformat;
        OFCondition status = fileformat.loadFile(s.c_str());

        if (!status.good()) {
            std::cerr << "Error: cannot load DICOM file (" << status.text() << ")" << std::endl;
            return -1;
        }

        DicomImage* dcmImage = new DicomImage(&fileformat, EXS_Unknown);
        if (dcmImage == nullptr || !dcmImage->isMonochrome()) {
            std::cerr << "Error: cannot read DICOM image (" << status.text() << ")" << std::endl;
            return -1;
        }

        const int bitsPerPixel = dcmImage->getDepth();
        const unsigned long numPixels = dcmImage->getWidth() * dcmImage->getHeight();
        if (numPixels != N_ROW * N_COL) {
            std::cerr << "Error: size of image (" << dcmImage->getWidth() << "*" << dcmImage->getHeight()
                      << ") different from required size (" << N_COL << "*" << N_ROW << ")" << std::endl;
            return -1;
        }

        const size_t bufferSize = numPixels * (pow2roundup(bitsPerPixel) / 8);

        if (imgData == nullptr) {
            std::cerr << "Error: cannot allocate " << bufferSize << " bytes" << std::endl;
            perror("Error");
            return -1;
        }

        if (dcmImage->getOutputData(imgData, bufferSize, 8, 0) == false) {
            std::cerr << "Error: cannot read pixels (" << status.text() << ")" << std::endl;
            return -1;
        }

        write_slice_in_buffer((uint8_t*)imgData, volume, i, N_ROW, N_COL, N_COUPLES);
    }
    delete[] imgData;
    return 0;

#else
    std::cerr << "Error: DICOM support is disabled" << std::endl;
    return -1;
#endif
}

int load_volumes_from_data(
    uint8_t* input_ref,
    uint8_t* input_flt,
    uint8_t* output_flt,
    const std::vector<uint8_t>& ref_volume,
    const std::vector<uint8_t>& float_volume,
    int n_row,
    int n_col,
    int n_couples,
    int row_padding,
    int col_padding,
    int depth_padding) {
    // Compute dimensions with padding
    const int padded_rows = n_row + row_padding;
    const int padded_cols = n_col + col_padding;
    const int padded_depth = n_couples + depth_padding;

    // Total buffer for safety (useful for initial memset)
    const size_t buffer_size = static_cast<size_t>(padded_rows) * padded_cols * padded_depth;

    // Check input volume dimensions
    if (ref_volume.size() != static_cast<size_t>(n_row * n_col * n_couples) ||
        float_volume.size() != static_cast<size_t>(n_row * n_col * n_couples)) {
        std::cerr << "Error: wrong input volume size\n";
        return -1;
    }

    // Symmetric padding
    const int top = row_padding / 2;
    const int left = col_padding / 2;

    for (int k = 0; k < n_couples; ++k) {
        // Temporary slice with padding
        std::vector<uint8_t> padded_slice(static_cast<size_t>(padded_rows) * padded_cols, 0);

        // Copy the original slice into the correct position
        for (int r = 0; r < n_row; ++r) {
            for (int c = 0; c < n_col; ++c) {
                padded_slice[(r + top) * padded_cols + (c + left)] =
                    ref_volume[static_cast<size_t>(k) * n_row * n_col + static_cast<size_t>(r) * n_col + c];
            }
        }

        // Write the slice into the 3D volume
        write_slice_in_buffer(padded_slice.data(), input_ref, k, padded_rows, padded_cols, padded_depth);
    }

    for (int k = 0; k < depth_padding; ++k) {
        std::vector<uint8_t> zero_slice(static_cast<size_t>(padded_rows) * padded_cols, 0);

        write_slice_in_buffer(zero_slice.data(), input_ref, n_couples + k, padded_rows, padded_cols, padded_depth);
    }

    for (int k = 0; k < n_couples; ++k) {
        std::vector<uint8_t> padded_slice(static_cast<size_t>(padded_rows) * padded_cols, 0);

        for (int r = 0; r < n_row; ++r) {
            for (int c = 0; c < n_col; ++c) {
                padded_slice[(r + top) * padded_cols + (c + left)] =
                    float_volume[static_cast<size_t>(k) * n_row * n_col + static_cast<size_t>(r) * n_col + c];
            }
        }

        write_slice_in_buffer(padded_slice.data(), input_flt, k, padded_rows, padded_cols, padded_depth);
    }

    for (int k = 0; k < depth_padding; ++k) {
        std::vector<uint8_t> zero_slice(static_cast<size_t>(padded_rows) * padded_cols, 0);

        write_slice_in_buffer(zero_slice.data(), input_flt, n_couples + k, padded_rows, padded_cols, padded_depth);
    }

    return 0;
}

int read_volume_from_file_PNG(
    uint8_t* volume,
    const int N_ROW,
    const int N_COL,
    const int N_COUPLES,
    const int row_padding,
    const int col_padding,
    const int depth_padding,
    const std::string& path) {
    std::printf("Reading volume from file\n");

    // compute symmetric padding
    int top = row_padding / 2;
    int bottom = row_padding - top;
    int left = col_padding / 2;
    int right = col_padding - left;

    int padded_rows = N_ROW + row_padding;
    int padded_cols = N_COL + col_padding;
    int padded_depth = N_COUPLES + depth_padding;

    for (int i = 0; i < N_COUPLES; i++) {
        std::string s = path + "IM" + std::to_string(i) + ".png";
        cv::Mat image = cv::imread(s, cv::IMREAD_GRAYSCALE);
        if (!image.data) {
            std::cout << "Not Found " << s << std::endl;
            return -1;
        }

        // add symmetric padding
        cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, 0);

        // copy the slice into the buffer
        std::vector<uint8_t> tmp(padded_rows * padded_cols);
        tmp.assign(image.begin<uint8_t>(), image.end<uint8_t>());
        write_slice_in_buffer(tmp.data(), volume, i, padded_rows, padded_cols, padded_depth);
    }

    // depth padding (slice vuote)
    for (int i = 0; i < depth_padding; i++) {
        std::vector<uint8_t> tmp(padded_rows * padded_cols, 0);
        write_slice_in_buffer(tmp.data(), volume, N_COUPLES + i, padded_rows, padded_cols, padded_depth);
    }

    return 0;
}
// NOTE: N_ROW, N_COL and N_COUPLES should reflect the shape of the dataset, and
// thus not include the padding.
int read_volume_from_file_PNG(
    uint8_t* volume,
    const int N_ROW,
    const int N_COL,
    const int N_COUPLES,
    const int BORDER_PADDING,
    const int DEPTH_PADDING,
    const std::string& path) {
    std::printf("Reading volume from file\n");
    for (int i = 0; i < N_COUPLES; i++) {
        std::string s = path + "IM" + std::to_string(i) + ".png";
        cv::Mat image = cv::imread(s, cv::IMREAD_GRAYSCALE);
        if (!image.data) {
            std::cout << "Not Found " << s << std::endl;
            return -1;
        }

        // add border-padding around the image
        cv::copyMakeBorder(
            image, image, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, cv::BORDER_CONSTANT, 0);

        // copy the slice into the buffer
        std::vector<uint8_t> tmp((N_ROW + 2 * BORDER_PADDING) * (N_COL + 2 * BORDER_PADDING));
        tmp.assign(image.begin<uint8_t>(), image.end<uint8_t>());
        write_slice_in_buffer(
            tmp.data(), volume, i, N_ROW + 2 * BORDER_PADDING, N_COL + 2 * BORDER_PADDING, N_COUPLES + DEPTH_PADDING);
    }

    for (int i = 0; i < DEPTH_PADDING; i++) {
        // copy the slice into the buffer
        std::vector<uint8_t> tmp((N_ROW + 2 * BORDER_PADDING) * (N_COL + 2 * BORDER_PADDING));
        tmp.assign(tmp.size(), 0);
        write_slice_in_buffer(
            tmp.data(),
            volume,
            N_COUPLES + i,
            N_ROW + 2 * BORDER_PADDING,
            N_COL + 2 * BORDER_PADDING,
            N_COUPLES + DEPTH_PADDING);
    }

    return 0;
}

// NOTE: SIZE and N_COUPLES should reflect the shape of the dataset, and thus
// not include the padding.

// NOTE: N_ROW, N_COL and N_COUPLES should reflect the shape of the dataset, and
// thus not include the padding.
void write_volume_to_file(
    uint8_t* volume,
    const int N_ROW,
    const int N_COL,
    const int N_COUPLES,
    const int BORDER_PADDING,
    const int DEPTH_PADDING,
    const std::string& path) {
    std::cout << "Writing volume to file" << std::endl;
    for (int i = 0; i < N_COUPLES; i++) {
        std::vector<uint8_t> tmp((N_ROW + 2 * BORDER_PADDING) * (N_COL + 2 * BORDER_PADDING));
        read_slice_from_buffer(
            volume, tmp.data(), i, N_ROW + 2 * BORDER_PADDING, N_COL + 2 * BORDER_PADDING, N_COUPLES + DEPTH_PADDING);
        cv::Mat slice = (cv::Mat(N_ROW + 2 * BORDER_PADDING, N_COL + 2 * BORDER_PADDING, CV_8U, tmp.data())).clone();
        slice = slice(cv::Rect(BORDER_PADDING, BORDER_PADDING, N_COL,
                               N_ROW));  // remove border-padding
        std::string s = path + "/IM" + std::to_string(i) + ".png";
        cv::imwrite(s, slice);
    }
}

void write_volume_to_file(
    uint8_t* volume,
    const int N_ROW,
    const int N_COL,
    const int N_COUPLES,
    const int row_padding,
    const int col_padding,
    const int DEPTH_PADDING,
    const std::string& path) {
    std::cout << "Writing volume to file" << std::endl;

    // compute total dimension with padding
    int padded_rows = N_ROW + row_padding;
    int padded_cols = N_COL + col_padding;

    // compute symmetric margins
    int top = row_padding / 2;
    int left = col_padding / 2;

    for (int i = 0; i < N_COUPLES; i++) {
        std::vector<uint8_t> tmp(padded_rows * padded_cols);
        read_slice_from_buffer(volume, tmp.data(), i, padded_rows, padded_cols, N_COUPLES + DEPTH_PADDING);

        cv::Mat slice_full = cv::Mat(padded_rows, padded_cols, CV_8U, tmp.data()).clone();

        // remove padding
        cv::Mat slice_cropped = slice_full(cv::Rect(left, top, N_COL, N_ROW));

        std::string s = path + "/IM" + std::to_string(i) + ".png";
        cv::imwrite(s, slice_cropped);
    }
}

// NOTE: SIZE and N_COUPLES should reflect the shape of the dataset, and thus
// not include the padding.
int read_volume_from_file(
    uint8_t* volume,
    const int n_row,
    const int n_col,
    const int N_COUPLES,
    const int row_padding,
    const int col_padding,
    const int DEPTH_PADDING,
    const std::string& path,
    const ImageFormat imageFormat) {
    switch (imageFormat) {
        case ImageFormat::PNG:
            return read_volume_from_file_PNG(
                volume, n_row, n_col, N_COUPLES, row_padding, col_padding, DEPTH_PADDING, path);
        case ImageFormat::DICOM:
            return read_volume_from_file_DICOM(volume, n_row, n_col, N_COUPLES, path);
    }
    return -1;
}

uint8_t track_reads(uint8_t* mem, const int index, float* ratio) {
    static unsigned int sequential_count = 0;
    static unsigned int total_count = 0;
    static unsigned int last_index = -2;

    if (ratio == NULL) {
        ++total_count;
        if (index == last_index + 1) {
            ++sequential_count;
        }
        last_index = index;

#ifdef DEBUG_ACCESSED_INDEXES
        std::cout << index << " ";
#endif

        return mem[index];
    }

    if (total_count != 0)
        *ratio = 100.0f * sequential_count / (float)total_count;
    else
        *ratio = 0.0f;

    return -1;
}

int cast_mats_to_vector(
    uint8_t* volume,
    std::vector<cv::Mat> images,
    const int n_row,
    const int n_col,
    const int N_COUPLES,
    const int BORDER_PADDING,
    const int DEPTH_PADDING) {
    const int padded_rows = n_row + 2 * BORDER_PADDING;
    const int padded_cols = n_col + 2 * BORDER_PADDING;
    const int LAYERS = N_COUPLES + DEPTH_PADDING;

    for (int i = 0; i < N_COUPLES; i++) {
        // Convert cv::Mat -> buffer
        std::vector<uint8_t> tmp(padded_rows * padded_cols);
        tmp.assign(images[i].begin<uint8_t>(), images[i].end<uint8_t>());

        write_slice_in_buffer(tmp.data(), volume, i, padded_rows, padded_cols, LAYERS);
    }

    // Depth padding slices
    for (int i = 0; i < DEPTH_PADDING; i++) {
        std::vector<uint8_t> tmp(padded_rows * padded_cols, 0);
        write_slice_in_buffer(tmp.data(), volume, N_COUPLES + i, padded_rows, padded_cols, LAYERS);
    }

    return 0;
}