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
        print(f"Immagine originale: {img.size[0]}x{img.size[1]} (larghezza x altezza)")
        
        # Resize image
        # Note: PIL uses (width, height) while here we use (rows, cols)
        # so `ncol` is width and `nrow` is height
        img_resized = img.resize((ncol, nrow), Image.Resampling.LANCZOS)
        
        # Save image in current folder
        img_resized.save(output_path)
        print(f"Immagine ridimensionata salvata: {output_path}")
        print(f"Nuove dimensioni: {ncol}x{nrow} (larghezza x altezza)")
        
        return True
        
    except Exception as e:
        print(f"Errore: {e}")
        return False

def main():
    # Check arguments
    if len(sys.argv) != 3:
        print("Uso: python resize_image.py <nrow> <ncol>")
        print("Esempio: python resize_image.py 64 128")
        sys.exit(1)
    
    try:
        nrow = int(sys.argv[1])
        ncol = int(sys.argv[2])
    except ValueError:
        print("Errore: nrow e ncol devono essere numeri interi")
        sys.exit(1)
    
    # Paths
    input_path = "/home/gsorrentino/projects/trilli_sw_prog/trilli-private/sw/dataset/size_32_32/IM1.png"
    output_path = "IM1.png"  # Save in current folder
    
    # Verify input file exists
    if not os.path.exists(input_path):
        print(f"Errore: il file {input_path} non esiste")
        sys.exit(1)
    
    # Run resize
    success = resize_image(input_path, output_path, nrow, ncol)
    
    if success:
        print(f"\nRidimensionamento completato con successo!")
        print(f"L'immagine IM1.png è stata salvata nella cartella corrente con dimensioni {nrow}x{ncol}")
    else:
        print("\nRidimensionamento fallito")
        sys.exit(1)

if __name__ == "__main__":
    main()