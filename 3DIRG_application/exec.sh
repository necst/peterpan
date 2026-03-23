#!/bin/bash

# Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide Conficconi, Eleonora D'Arnese

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#Get Current Directory
APP_NAME=trilli_app_sw
VOLUME_DEPTH=246
OUTPUT_FOLDER=./output_folder/
REF_FOLDER=/home/gsorrentino/CT/
FLT_FOLDER=/home/gsorrentino/PET/
RANGE_TX=50
RANGE_TY=50
RANGE_ANG=1

if [ -n "$1" ]; then
    FLT_FOLDER=$1
fi

echo "Reference folder: $REF_FOLDER"
echo "Floating folder: $FLT_FOLDER"
echo "Registered output folder: $OUTPUT_FOLDER"

mkdir -p $OUTPUT_FOLDER

./$APP_NAME $REF_FOLDER $FLT_FOLDER $OUTPUT_FOLDER $VOLUME_DEPTH $RANGE_TX $RANGE_TY $RANGE_ANG
