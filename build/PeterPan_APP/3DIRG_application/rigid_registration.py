#!/usr/bin/env python3

import os
import sys
import argparse
from PIL import Image  # per leggere dimensioni immagini

# Add build directory to sys.path to import the pybind11 wrapper
sys.path.append("./build")
import trilli_wrapper

# Fixed parameters
RANGE_TX = 80
RANGE_TY = 80
RANGE_ANG = 1.0

# Argument parser
parser = argparse.ArgumentParser(description="Run rigid registration using trilli_wrapper.")
parser.add_argument("--ref_folder", required=True, help="Path to the reference volume folder (already prepared)")
parser.add_argument("--flt_folder", required=True, help="Path to the floating volume folder (already prepared)")
parser.add_argument("--output_folder", required=True, help="Path to the output folder")
parser.add_argument("--xclbin_file", default="./overlay_hw.xclbin", help="Path to the xclbin file for HW execution")

args = parser.parse_args()

# Ensure output folder exists
os.makedirs(args.output_folder, exist_ok=True)

# --------------------------------------------------------------------------------
# Determine volume dimensions by inspecting reference folder
# --------------------------------------------------------------------------------
print(f"Inspecting reference volume in: {args.ref_folder}")

# List all files in the folder (assuming PNG slices)
valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".raw")
slice_files = sorted(
    [f for f in os.listdir(args.ref_folder) if f.lower().endswith(valid_ext)]
)

if len(slice_files) == 0:
    raise RuntimeError(f"No slice images found in {args.ref_folder}")

# Use first slice to get width and height
first_slice_path = os.path.join(args.ref_folder, slice_files[0])
with Image.open(first_slice_path) as img:
    n_col, n_row = img.size  # PIL returns (width, height)

n_couples = len(slice_files)

print(f"Volume dimensions detected -> n_row: {n_row}, n_col: {n_col}, n_couples (depth): {n_couples}")

# --------------------------------------------------------------------------------
# Run registration
# --------------------------------------------------------------------------------
print("Running rigid registration with file-path interface...")

trilli_wrapper.run_rigid_registration_trilli(
    args.ref_folder,
    args.flt_folder,
    args.output_folder,
    n_couples,
    n_row,
    n_col,
    RANGE_TX,
    RANGE_TY,
    RANGE_ANG,
    args.xclbin_file
)

print("Rigid registration completed successfully.")
