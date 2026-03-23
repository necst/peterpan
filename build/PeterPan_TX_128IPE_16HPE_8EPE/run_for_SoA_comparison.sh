#!/bin/bash

n_couples=246
tx=30
ty=30
ang=25
reps=50

./generate_dataset.sh 512 246

echo "> Running"
./host_overlay.exe $n_couples $tx $ty $ang $reps 
echo "--------------------"
