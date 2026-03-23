#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from math import ceil
sys.path.append("./build")
import trilli_wrapper
import dataloader
from dataloader import load_nii_gz, read_volume_with_datalayout, write_volume_with_datalayout, save_volume_as_nifti
from utils import estimate_background_mode, replace_background,rebuild_volume_from_flat, create_flat_buffers, crop_volume, pad_to_resolution

BORDER_PADDING = 0
RANGE_TX = 80
RANGE_TY = 80
RANGE_ANG = 1

def run_registration(ref_flat, flt_flat, output_folder, depth, n_row, n_col, xclbin):
    """Run rigid registration using trilli_wrapper."""
    return trilli_wrapper.run_rigid_registration_trilli_from_data(
        ref_flat,
        flt_flat,
        output_folder,
        depth,
        n_row,
        n_col,
        RANGE_TX,
        RANGE_TY,
        RANGE_ANG,
        xclbin
    )

# -----------------------------
# Main workflow
# -----------------------------
def main(args):
    # Load volumes
    ref_volume = load_nii_gz(args.ref)
    flt_volume = load_nii_gz(args.flt)

    # Normalize background
    bg_ref = estimate_background_mode(ref_volume)
    bg_flt = estimate_background_mode(flt_volume)
    print(f"Estimated background - ref: {bg_ref}, flt: {bg_flt}")
    if bg_ref != bg_flt:
        print("ALERT - different backgrounds - making them homogeneous by replacing REF values.")
        ref_volume = replace_background(ref_volume, bg_ref, bg_flt)

    if ref_volume.shape != flt_volume.shape:
        raise ValueError(f"Input volumes must have same shape, got {ref_volume.shape} vs {flt_volume.shape}")

    n_row, n_col, depth = ref_volume.shape

    # Compute padded dimensions
    row_padded = ceil(n_row / 32) * 32
    col_padded = ceil(n_col / 32) * 32
    depth_padded = ceil(depth / 32) * 32

    # Pad volumes
    ref_volume = pad_to_resolution(ref_volume, (col_padded, row_padded, depth_padded))
    flt_volume = pad_to_resolution(flt_volume, (col_padded, row_padded, depth_padded))
    print(f"Volume shape after resolution adjustment: {ref_volume.shape}, dtype: {ref_volume.dtype}")

    # Convert to uint8
    ref_volume = ref_volume.astype(np.uint8)
    flt_volume = flt_volume.astype(np.uint8)

    # Create interlaced flat buffers
    ref_flat, _, _ = create_flat_buffers(ref_volume, BORDER_PADDING)
    flt_flat, _, _ = create_flat_buffers(flt_volume, BORDER_PADDING)

    # Save input volumes
    write_volume_with_datalayout(ref_flat, col_padded, row_padded, depth_padded, os.path.join(args.output_folder, "input_ref"))
    write_volume_with_datalayout(flt_flat, col_padded, row_padded, depth_padded, os.path.join(args.output_folder, "input_flt"))

    # Run registration
    print("Running rigid registration...")
    registered_flat = run_registration(ref_flat, flt_flat, args.output_folder, depth, n_row, n_col, args.xclbin_file)

    # Save registered volume as PNG slices
    write_volume_with_datalayout(registered_flat, col_padded, row_padded, depth_padded, os.path.join(args.output_folder, "output", "registered"))

    # Save NIfTI if requested
    if args.save_nifti:
        registered_array = rebuild_volume_from_flat(registered_flat, (row_padded, col_padded, depth_padded))
        registered_array_cropped = crop_volume(registered_array, (n_row, n_col, depth))
        code = os.path.basename(args.flt).split('_')[0]
        filename = f"registered_{code}.nii.gz"
        out_path = os.path.join(args.output_folder_nifti, filename)
        save_volume_as_nifti(registered_array_cropped, out_path, affine=np.eye(4))
    else:
        print("Registration completed (no NIfTI file saved).")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run rigid registration using trilli_wrapper.")
    parser.add_argument("--ref", required=True, help="Path to the reference volume (.nii.gz)")
    parser.add_argument("--flt", required=True, help="Path to the floating volume (.nii.gz)")
    parser.add_argument("--output_folder", default="./output_folder/", help="Output folder path")
    parser.add_argument("--save_nifti", action="store_true", help="Save registered volume as NIfTI")
    parser.add_argument("--output_folder_nifti", default="./output_folder/", help="Output folder path for NIfTI files")
    parser.add_argument("--xclbin_file", default="./overlay_hw.xclbin", help="path to the xclbin file for the hw execution")

    args = parser.parse_args()

    main(args)
