#!/bin/bash

# MIT License

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

IMG_DIM=512
PYCODE_onestep=single_step.py
PYCODE_Powell=powell_torch.py

DATASET_FLDR=./
BASE_DATASET=Dataset/dataset
RES_PATH=Dataset/OutputDate_prova
metric=(MI)

NUM_REPEAT=50

dev=(cuda:0)
#dev=(cpu)

CSV_PREFIX=athena_times

PYENV_DIR=pyenv_athena
if [ ! -d "$PYENV_DIR" ]; then
  echo "Python virtual env '$PYENV_DIR' does not exist: launch 'install_env.sh' first"
  exit 1
fi

source $PYENV_DIR/bin/activate

python --version

for k in {1..1}
do
    FILE_NAME_BILINEAR="$CSV_PREFIX"_bilinear_"$IMG_DIM".csv

    echo tot,tx,mi > $FILE_NAME_BILINEAR

    for j in "${dev[@]}"
    do
        for i in $(seq 1 $NUM_REPEAT)
        do
          echo --------------------------
          echo DIMENSION: $IMG_DIM
          echo --------------------------

          cd $BASE_DATASET > /dev/null 2>&1
          rm -f IM*.png
          cp size_$IMG_DIM/IM1.png .
          ./duplicate_slices.sh
          cd - > /dev/null 2>&1
          
          command_bilinear="CUDA_VISIBLE_DEVICES=0 python3 $PYCODE_onestep -cp $BASE_DATASET -pp $BASE_DATASET -rp . -t 1 -px $DATASET_FLDR -im $IMG_DIM -dvc $j -vol 246 -f $FILE_NAME_BILINEAR -m bilinear"

          echo $command_bilinear
          eval $command_bilinear

          echo 
        done
    done

    IMG_DIM=$(($IMG_DIM*2))
done
