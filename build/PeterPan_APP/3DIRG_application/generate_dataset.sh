#!/bin/bash

dim=512
slices=245

echo "Generating ${dim}x${dim}x${slices} volume"

cd CT || exit

cp size_"${dim}"/IM1.png IM0.png
cp size_"${dim}"/IM1.png .

./duplicate_slices.sh 2 "${slices}"

cd ../PET || exit

cp size_"${dim}"/IM1.png IM0.png
cp size_"${dim}"/IM1.png .

./duplicate_slices.sh 2 "${slices}"

echo "Done"
