#!/bin/bash

dim=$1
slices=$2

echo "Generating ${dim}x${dim}x${slices} volume"

cd dataset || exit

cp size_"${dim}"/IM1.png .

./duplicate_slices.sh 2 "${slices}"

echo "Done"
