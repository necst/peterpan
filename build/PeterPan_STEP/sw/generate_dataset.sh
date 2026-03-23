#!/bin/bash

n_row=$1
n_col=$2
slices=$3

echo "Generating ${n_row}x${n_col}x${slices} volume"

cd dataset || exit

cp size_"${n_row}_${n_col}"/IM1.png .

./duplicate_slices.sh 2 "${slices}"

echo "Done"
