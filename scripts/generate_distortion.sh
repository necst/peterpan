#!/bin/bash

n_couples=$1
tx=$2
ty=$3
ang=$4

cd STEP_32IPE || exit 1

echo "> Running bitstream STEP_32IPE to generate distortion"

# copying dataset into transformation folder
cp ../3DIRG_Application/PET/*.png dataset/

for i in $(seq 0 $((n_couples-1)) | tac); do
    mv dataset/IM$i.png dataset/IM$((i+1)).png
done

# applying distortion
./host_overlay.exe $n_couples $tx $ty $ang 1

echo "Done"

# copying images into plotting folder
# cp dataset/IM1.png ../paper_fig/figure8/data/gold_IM0.png
# cp dataset_output/IM1.png ../paper_fig/figure8/data/float_IM0.png

for i in $(seq 1 $((n_couples))); do
    mv dataset_output/IM$i.png dataset_output/IM$((i-1)).png
done

cd -
