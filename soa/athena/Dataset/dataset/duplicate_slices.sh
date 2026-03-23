#!/bin/bash

# Set the base filename
base="IM"

# Set the starting number (should be $1)
start=2

# Set the number of copies (should be $2)
copies=256
echo "generating volume of depth ${copies} by copying ${base}1.png"

# Loop through the copies and copy the file
for ((i=$start; i<$start+$copies-1; i++)); do
    cp "${base}1.png" "${base}${i}.png"
done
