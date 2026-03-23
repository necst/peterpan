#!/usr/bin/env python3
"""
Script to resize an image to the specified dimensions.
Usage: python resize_image.py <nrow> <ncol>
"""

import sys
from PIL import Image
import os

def resize_image(input_path, output_path, nrow, ncol):
    """
    Resize an image to the specified dimensions.

    Args:
        input_path: input image path
        output_path: path where the resized image is saved
        nrow: desired number of rows (height)
        ncol: desired number of columns (width)
    """
    try:
        # Open image
        img = Image.open(input_path)
        print(f"original image: {img.size[0]}x{img.size[1]} (width x height)")
        
        # Resize image
        # Note: PIL uses (width, height) while here we use (rows, cols)
        # so `ncol` is width and `nrow` is height
        img_resized = img.resize((ncol, nrow), Image.Resampling.LANCZOS)
        
        # Save image in current folder
        img_resized.save(output_path)
        print(f"resized image saved: {output_path}")
        print(f"new dimensions: {ncol}x{nrow} (width x height)")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python resize_image.py <nrow> <ncol>")
        print("Example: python resize_image.py 64 128")
        sys.exit(1)
    
    try:
        nrow = int(sys.argv[1])
        ncol = int(sys.argv[2])
    except ValueError:
        print("Error: nrow and ncol must be integers")
        sys.exit(1)
    
    # Paths
    input_path = "./IM1.png"
    output_path = "IM1.png"  # Save in current folder
    
    # Verify input file exists
    if not os.path.exists(input_path):
        print(f"Error: the file {input_path} does not exist")
        sys.exit(1)
    
    # Run resize
    success = resize_image(input_path, output_path, nrow, ncol)
    
    if success:
        print(f"\nResize completed successfully!")
        print(f"The image IM1.png has been saved in the current folder with dimensions {nrow}x{ncol}")
    else:
        print("\nResize failed")
        sys.exit(1)

if __name__ == "__main__":
    main()