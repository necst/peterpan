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

import numpy as np
import pathlib
import argparse
import sys
import cv2
import os
import math

# python3 image_generator.py 64 64 sw/dataset_BW


# generate 16x16 grayscale grid, of size "size"
def generate_grid(size, color_scale_factor=1):
    image = np.zeros((size, size), dtype=np.uint8)
    grid_size = size // 16
    for i in range(16):
        for j in range(16):
            if (i + j) % 2 == 0:
                image[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = (255 // ((j + i) / 4 + 1)) * color_scale_factor
    return image



if __name__ == '__main__':
    # use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a volume of images')
    parser.add_argument('size', type=int, help='size of the images', default=512)
    parser.add_argument('n_couples', type=int, help='number of images', default=256)
    parser.add_argument('path', type=str, help='path to save the images', default='generated_images')

    args = parser.parse_args()
    
    # if path does not exist, create it
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    else:
        print("Directory already exists, overwriting files")

    # save the volume as a .png file with opencv
    for i in range(args.n_couples):
        cv2.imwrite(args.path + "/IM" + str(i+1) + ".png", generate_grid(args.size, 0.25*(3 - i % 4) + 0.25))



