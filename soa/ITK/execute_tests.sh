#!/bin/bash

mkdir ./output

N=1
shift

CT_PATH="./PNG_CT" 
PET_PATH="./PNG_PET" 
OUTPUT_PATH="./output" 

COMMAND="./build_pow/TestRegistration $CT_PATH $PET_PATH $OUTPUT_PATH"

for ((i = 1; i <= N; i++)); do
  echo "Executing $i / $N..."
  eval $COMMAND
done

echo "Finished."
