#include "Versal3DIR.hpp"

// ------------------------------------------------------
// Constructor
// ------------------------------------------------------
Versal3DIR::Versal3DIR(xrt::device& device, xrt::uuid& xclbin_uuid, int n_couples, int n_row, int n_col)
    : device(device),
      xclbin_uuid(xclbin_uuid),
      n_couples(n_couples),
      n_row(n_row),
      n_col(n_col),
      row_padding((NUM_PIXELS_PER_READ - (n_row % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ),
      col_padding((NUM_PIXELS_PER_READ - (n_col % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ),
      depth_padding((NUM_PIXELS_PER_READ - (n_couples % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ),
      buffer_size((n_row + row_padding) * (n_col + col_padding) * (n_couples + depth_padding)),
      curr_size((n_row + row_padding) * (n_col + col_padding) * (n_couples + depth_padding)) {
    std::cout << "Going to allocate kernels of buffer size: " << buffer_size << std::endl;

    krnl_setup_aie = xrt::kernel(device, xclbin_uuid, "setup_aie");
    krnl_fetcher_A = xrt::kernel(device, xclbin_uuid, "fetcher_A");
    krnl_fetcher_B = xrt::kernel(device, xclbin_uuid, "fetcher_B");
    krnl_fetcher_C = xrt::kernel(device, xclbin_uuid, "fetcher_C");
    krnl_fetcher_D = xrt::kernel(device, xclbin_uuid, "fetcher_D");
    krnl_scheduler_IPE = xrt::kernel(device, xclbin_uuid, "scheduler_IPE");
    krnl_setup_mi = xrt::kernel(device, xclbin_uuid, "setup_mi");
    krnl_mutual_info = xrt::kernel(device, xclbin_uuid, "mutual_information_master");

    // Bank setup
    bank_fetcher_A_flt_in = krnl_fetcher_A.group_id(arg_fetcher_in_flt_original_ptr);
    bank_fetcher_B_flt_in = krnl_fetcher_B.group_id(arg_fetcher_in_flt_original_ptr);
    bank_fetcher_C_flt_in = krnl_fetcher_C.group_id(arg_fetcher_in_flt_original_ptr);
    bank_fetcher_D_flt_in = krnl_fetcher_D.group_id(arg_fetcher_in_flt_original_ptr);
    bank_setup_mi = krnl_setup_mi.group_id(arg_setup_mi_pixel_out);
    bank_mutual_info = krnl_mutual_info.group_id(arg_mutual_info_reference);
    bank_mutual_info_output = krnl_mutual_info.group_id(arg_mutual_info_mi);

    // Buffers
    buffer_fetcher_A_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_A_flt_in);
    buffer_fetcher_B_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_B_flt_in);
    buffer_fetcher_C_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_C_flt_in);
    buffer_fetcher_D_flt_in = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_fetcher_D_flt_in);

    buffer_setup_mi_flt_transformed = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_setup_mi);

    buffer_mutual_info_reference = xrt::bo(device, buffer_size, xrt::bo::flags::normal, bank_mutual_info);

    buffer_mutual_info_output = xrt::bo(device, sizeof(float), xrt::bo::flags::normal, bank_mutual_info_output);

    // Runners
    run_setup_aie = xrt::run(krnl_setup_aie);
    run_fetcher_A = xrt::run(krnl_fetcher_A);
    run_fetcher_B = xrt::run(krnl_fetcher_B);
    run_fetcher_C = xrt::run(krnl_fetcher_C);
    run_fetcher_D = xrt::run(krnl_fetcher_D);
    run_scheduler_IPE = xrt::run(krnl_scheduler_IPE);
    run_setup_mi = xrt::run(krnl_setup_mi);
    run_mutual_info = xrt::run(krnl_mutual_info);

    // Set static kernel args
    run_fetcher_A.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_A_flt_in);
    run_fetcher_A.set_arg(arg_fetcher_in_n_couples, n_couples + depth_padding);
    run_fetcher_A.set_arg(arg_fetcher_in_n_row, n_row + row_padding);
    run_fetcher_A.set_arg(arg_fetcher_in_n_col, n_col + col_padding);

    run_fetcher_B.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_B_flt_in);
    run_fetcher_B.set_arg(arg_fetcher_in_n_couples, n_couples + depth_padding);
    run_fetcher_B.set_arg(arg_fetcher_in_n_row, n_row + row_padding);
    run_fetcher_B.set_arg(arg_fetcher_in_n_col, n_col + col_padding);

    run_fetcher_C.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_C_flt_in);
    run_fetcher_C.set_arg(arg_fetcher_in_n_couples, n_couples + depth_padding);
    run_fetcher_C.set_arg(arg_fetcher_in_n_row, n_row + row_padding);
    run_fetcher_C.set_arg(arg_fetcher_in_n_col, n_col + col_padding);

    run_fetcher_D.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_D_flt_in);
    run_fetcher_D.set_arg(arg_fetcher_in_n_couples, n_couples + depth_padding);
    run_fetcher_D.set_arg(arg_fetcher_in_n_row, n_row + row_padding);
    run_fetcher_D.set_arg(arg_fetcher_in_n_col, n_col + col_padding);

    run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_couples, n_couples + depth_padding);
    run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_row, n_row + row_padding);
    run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_col, n_col + col_padding);

    run_setup_mi.set_arg(arg_setup_mi_pixel_out, buffer_setup_mi_flt_transformed);
    run_setup_mi.set_arg(arg_setup_mi_n_couples, n_couples + depth_padding);
    run_setup_mi.set_arg(arg_setup_mi_n_row, n_row + row_padding);
    run_setup_mi.set_arg(arg_setup_mi_n_col, n_col + col_padding);

    run_mutual_info.set_arg(arg_mutual_info_reference, buffer_mutual_info_reference);
    run_mutual_info.set_arg(arg_mutual_info_mi, buffer_mutual_info_output);
    run_mutual_info.set_arg(arg_mutual_info_input_size, (unsigned int)buffer_size);
}

void Versal3DIR::rebuild_runners() {
    run_setup_aie = xrt::run(krnl_setup_aie);
    run_fetcher_A = xrt::run(krnl_fetcher_A);
    run_fetcher_B = xrt::run(krnl_fetcher_B);
    run_fetcher_C = xrt::run(krnl_fetcher_C);
    run_fetcher_D = xrt::run(krnl_fetcher_D);
    run_scheduler_IPE = xrt::run(krnl_scheduler_IPE);
    run_setup_mi = xrt::run(krnl_setup_mi);
    run_mutual_info = xrt::run(krnl_mutual_info);
}
// ------------------------------------------------------
// Load from FILES (class allocates => owns the memory)
// ------------------------------------------------------
int Versal3DIR::read_volumes_from_file(
    const std::string& path_ref, const std::string& path_flt, const ImageFormat imageFormat) {
    input_flt = new uint8_t[buffer_size];
    input_ref = new uint8_t[buffer_size];
    output_flt = new uint8_t[buffer_size];
    owns_flt = owns_ref = owns_out = true;

    if (read_volume_from_file(
            input_flt, n_row, n_col, n_couples, row_padding, col_padding, depth_padding, path_flt, imageFormat) == -1)
        return -1;

    if (read_volume_from_file(
            input_ref, n_row, n_col, n_couples, row_padding, col_padding, depth_padding, path_ref, imageFormat) == -1)
        return -1;

    return 0;
}
//
//// XRT + DMA require page-aligned (4096B) host memory. std::vector.data()
//// is not guaranteed to be page-aligned, so using it directly can cause
//// "failed to allocate userptr bo". We allocate 4096-aligned buffers via
//// posix_memalign() and copy the data. CPU code does not require this.
//
//
// int Versal3DIR::load_volumes_from_data(const std::vector<uint8_t>& ref_volume,
//                                       const std::vector<uint8_t>& float_volume)
//{
//
//    if (ref_volume.size() > buffer_size || float_volume.size() > buffer_size) return -1;
//
//    void* aligned_flt = nullptr;
//    void* aligned_ref = nullptr;
//
//    posix_memalign(&aligned_flt, 4096, buffer_size);
//    posix_memalign(&aligned_ref, 4096, buffer_size);
//
//    std::memcpy(aligned_flt, float_volume.data(), float_volume.size());
//    std::memcpy(aligned_ref, ref_volume.data(), ref_volume.size());
//
//    input_flt = reinterpret_cast<uint8_t*>(aligned_flt);
//    input_ref = reinterpret_cast<uint8_t*>(aligned_ref);
//
//    owns_ref = owns_flt = false;
//
//    return 0;
//}
//

int Versal3DIR::load_volumes_from_data(
    const std::vector<uint8_t>& ref_volume, const std::vector<uint8_t>& float_volume) {
    if (ref_volume.size() > buffer_size || float_volume.size() > buffer_size) return -1;

    // Use std::vector buffers directly (not aligned)
    std::cout << "size of input_flt " << float_volume.size() << std::endl;
    std::cout << "size of input_ref " << ref_volume.size() << std::endl;
    std::cout << "value of buffer size " << buffer_size << std::endl;

    input_flt = const_cast<uint8_t*>(float_volume.data());
    input_ref = const_cast<uint8_t*>(ref_volume.data());

    owns_ref = owns_flt = false;

    return 0;
}

void Versal3DIR::zero_buffer() {
    std::vector<uint8_t> zeros(buffer_size, 0);
    buffer_fetcher_A_flt_in.write(zeros.data());
    buffer_fetcher_A_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    buffer_fetcher_B_flt_in.write(zeros.data());
    buffer_fetcher_B_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    buffer_fetcher_C_flt_in.write(zeros.data());
    buffer_fetcher_C_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    buffer_fetcher_D_flt_in.write(zeros.data());
    buffer_fetcher_D_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // setup_mi transformed floating buffer
    buffer_setup_mi_flt_transformed.write(zeros.data());
    buffer_setup_mi_flt_transformed.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // reference buffer
    buffer_mutual_info_reference.write(zeros.data());
    buffer_mutual_info_reference.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // mutual info output buffer (only 4 bytes -> use memset)
    float zero_float = 0.0f;
    buffer_mutual_info_output.write(&zero_float);
    buffer_mutual_info_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

// ------------------------------------------------------
void Versal3DIR::set_transform_params(float TX, float TY, float ANG) {
    run_setup_aie.set_arg(arg_setup_aie_in_tx, TX);
    run_setup_aie.set_arg(arg_setup_aie_in_ty, TY);
    run_setup_aie.set_arg(arg_setup_aie_in_ang, ANG);
    run_setup_aie.set_arg(arg_setup_aie_in_n_couples, n_couples + depth_padding);
    run_setup_aie.set_arg(arg_setup_aie_in_n_row, n_row + row_padding);
    run_setup_aie.set_arg(arg_setup_aie_in_n_col, n_col + col_padding);
}

// ------------------------------------------------------
void Versal3DIR::write_floating_volume(double* duration) {
    Timer t;
    if (duration) t.start();
    buffer_fetcher_A_flt_in.write(input_flt, curr_size, 0);
    buffer_fetcher_A_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fetcher_B_flt_in.write(input_flt, curr_size, 0);
    buffer_fetcher_B_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fetcher_C_flt_in.write(input_flt, curr_size, 0);
    buffer_fetcher_C_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fetcher_D_flt_in.write(input_flt, curr_size, 0);
    buffer_fetcher_D_flt_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    if (duration) *duration += t.getElapsedSeconds();
}

// ------------------------------------------------------
void Versal3DIR::write_reference_volume(double* duration) {
    Timer t;
    if (duration) t.start();
    buffer_mutual_info_reference.write(input_ref, curr_size, 0);
    buffer_mutual_info_reference.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (duration) *duration += 0;
}

// ------------------------------------------------------
void Versal3DIR::run(double* duration) {
    Timer t;
    if (duration) t.start();

    run_setup_aie.start();
    run_fetcher_A.start();
    run_fetcher_B.start();
    run_fetcher_C.start();
    run_fetcher_D.start();
    run_scheduler_IPE.start();
    run_setup_mi.start();
    run_mutual_info.start();

    run_setup_aie.wait();
    run_fetcher_A.wait();
    run_fetcher_B.wait();
    run_fetcher_C.wait();
    run_fetcher_D.wait();
    run_scheduler_IPE.wait();
    run_setup_mi.wait();
    run_mutual_info.wait();

    if (duration) *duration += t.getElapsedSeconds();
}

// ------------------------------------------------------
void Versal3DIR::read_flt_transformed(double* duration) {
    Timer t;
    if (duration) t.start();

    buffer_setup_mi_flt_transformed.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (output_flt && owns_out) buffer_setup_mi_flt_transformed.read(output_flt);

    if (duration) *duration += t.getElapsedSeconds();
}

// ------------------------------------------------------
float Versal3DIR::read_mutual_information() {
    float out = -1.0f;
    buffer_mutual_info_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffer_mutual_info_output.read(&out);
    return out;
}

float Versal3DIR::hw_exec(float TX, float TY, float ANG, double* duration_exec) {
    set_transform_params(TX, TY, ANG);

    double dummy = 0.0;
    double* d = duration_exec ? duration_exec : &dummy;
    run(d);

    return read_mutual_information();
}

float Versal3DIR::hw_exec_tx(float TX, float TY, float ANG, double* duration_exec, bool save) {
    set_transform_params(TX, TY, ANG);

    double dummy = 0.0;
    double* d = duration_exec ? duration_exec : &dummy;

    write_floating_volume(d);
    write_reference_volume(d);
    run(d);

    if (save) read_flt_transformed(d);

    return read_mutual_information();
}
void Versal3DIR::update_dimensions(int new_n_couples, int new_n_row, int new_n_col) {
    // --- 1) Update values ---
    n_couples = new_n_couples;
    n_row = new_n_row;
    n_col = new_n_col;

    row_padding = (NUM_PIXELS_PER_READ - (n_row % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ;
    col_padding = (NUM_PIXELS_PER_READ - (n_col % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ;
    depth_padding = (NUM_PIXELS_PER_READ - (n_couples % NUM_PIXELS_PER_READ)) % NUM_PIXELS_PER_READ;
    int eff_row = n_row + row_padding;
    int eff_col = n_col + col_padding;
    int eff_couples = n_couples + depth_padding;

    curr_size = eff_row * eff_col * eff_couples;
    std::cout << "into update parameters - curr size: " << curr_size << std::endl;
    // --- 2) Rebuild runners: required ---
    rebuild_runners();

    // --- 3) Re-set arguments STATICI (che puntano ai BO) ---
    run_fetcher_A.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_A_flt_in);
    run_fetcher_B.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_B_flt_in);
    run_fetcher_C.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_C_flt_in);
    run_fetcher_D.set_arg(arg_fetcher_in_flt_original_ptr, buffer_fetcher_D_flt_in);

    run_setup_mi.set_arg(arg_setup_mi_pixel_out, buffer_setup_mi_flt_transformed);

    run_mutual_info.set_arg(arg_mutual_info_reference, buffer_mutual_info_reference);
    run_mutual_info.set_arg(arg_mutual_info_mi, buffer_mutual_info_output);

    // setup_aie
    run_setup_aie.set_arg(arg_setup_aie_in_n_couples, eff_couples);
    run_setup_aie.set_arg(arg_setup_aie_in_n_row, eff_row);
    run_setup_aie.set_arg(arg_setup_aie_in_n_col, eff_col);

    // fetchers A/B/C/D
    run_fetcher_A.set_arg(arg_fetcher_in_n_couples, eff_couples);
    run_fetcher_A.set_arg(arg_fetcher_in_n_row, eff_row);
    run_fetcher_A.set_arg(arg_fetcher_in_n_col, eff_col);

    run_fetcher_B.set_arg(arg_fetcher_in_n_couples, eff_couples);
    run_fetcher_B.set_arg(arg_fetcher_in_n_row, eff_row);
    run_fetcher_B.set_arg(arg_fetcher_in_n_col, eff_col);

    run_fetcher_C.set_arg(arg_fetcher_in_n_couples, eff_couples);
    run_fetcher_C.set_arg(arg_fetcher_in_n_row, eff_row);
    run_fetcher_C.set_arg(arg_fetcher_in_n_col, eff_col);

    run_fetcher_D.set_arg(arg_fetcher_in_n_couples, eff_couples);
    run_fetcher_D.set_arg(arg_fetcher_in_n_row, eff_row);
    run_fetcher_D.set_arg(arg_fetcher_in_n_col, eff_col);

    // scheduler IPE
    run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_couples, eff_couples);
    run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_row, eff_row);
    run_scheduler_IPE.set_arg(arg_scheduler_IPE_in_n_col, eff_col);

    // setup_mi
    run_setup_mi.set_arg(arg_setup_mi_n_couples, eff_couples);
    run_setup_mi.set_arg(arg_setup_mi_n_row, eff_row);
    run_setup_mi.set_arg(arg_setup_mi_n_col, eff_col);

    run_mutual_info.set_arg(arg_mutual_info_input_size, (unsigned int)curr_size);
}

// ------------------------------------------------------
Versal3DIR::~Versal3DIR() {
    if (owns_ref) delete[] input_ref;
    if (owns_flt) delete[] input_flt;
    if (owns_out) delete[] output_flt;
}
