#!/bin/bash

# Set the paths as variables for easier maintenance and reuse
CURRENT_DIR=$(pwd)
SRC_PATH="$CURRENT_DIR"
DATASET_PATH="$CURRENT_DIR/dataset"
OUTPUT_PATH="$CURRENT_DIR/outputpow_sitk"

mkdir -p $DATASET_PATH
mkdir -p $DATASET_PATH/CT
mkdir -p $DATASET_PATH/PET

# Define the input and output parameters
CP_PATH="$DATASET_PATH/CT"
PP_PATH="$DATASET_PATH/PET/"
RP_PATH="$OUTPUT_PATH/output"

ITERS=200
OPT="pow"
OUTPUT_FILE="sitkpow.csv"

# Execute the Python script with the specified arguments
python3 "$SRC_PATH/itk3D.py" -cp "$CP_PATH" -pp "$PP_PATH" -rp "$RP_PATH" -it "$ITERS" -opt "$OPT" -f "$RP_PATH/$OUTPUT_FILE"
