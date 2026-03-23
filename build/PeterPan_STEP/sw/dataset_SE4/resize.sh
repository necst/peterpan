#!/bin/bash

# for each IMx.png with x from 1 to N
# resize IMx.png to Dx2D and save it as resized_D/IMx.png
# where D is the dimension of the resized image
# and 2D is the number of dimensions of the resized image

# if arguments are not 2, error
if [ $# -ne 2 ]
then
    echo "Usage: $0 <new dimension> <number of slices>"
    exit 1
fi

D=$1
N=$2

OUTPUT_FOLDER="resized_$D"

echo "Resizing images to $D x $D pixels in folder $OUTPUT_FOLDER"

# shell code:
mkdir -p $OUTPUT_FOLDER
for ((i=1; i<=$N; i++))
do
    convert IM$i.png -resize $Dx$D $OUTPUT_FOLDER/IM$i.png
done
