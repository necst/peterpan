#!/bin/bash

n_couples_list=(32 64 128 256 512)

tx=30
ty=30
ang=25
reps=50

./generate_dataset.sh 512 512

for n_couples in "${n_couples_list[@]}"
do
    echo "> Running for depth = $n_couples"
    ./host_overlay.exe $n_couples $tx $ty $ang $reps 
    echo "--------------------"
done
