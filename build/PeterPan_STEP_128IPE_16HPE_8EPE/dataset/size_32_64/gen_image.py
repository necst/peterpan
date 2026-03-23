#!/usr/bin/env python3
"""
Script per ridimensionare un'immagine alle dimensioni specificate.
Uso: python resize_image.py <nrow> <ncol>
"""

import sys
from PIL import Image
import os

def resize_image(input_path, output_path, nrow, ncol):
    """
    Ridimensiona un'immagine alle dimensioni specificate.
    
    Args:
        input_path: percorso dell'immagine di input
        output_path: percorso dove salvare l'immagine ridimensionata
        nrow: numero di righe (altezza) desiderate
        ncol: numero di colonne (larghezza) desiderate
    """
    try:
        # Apri l'immagine
        img = Image.open(input_path)
        print(f"Immagine originale: {img.size[0]}x{img.size[1]} (larghezza x altezza)")
        
        # Ridimensiona l'immagine
        # Nota: PIL usa (width, height) mentre tu usi (rows, cols)
        # quindi ncol è la larghezza e nrow è l'altezza
        img_resized = img.resize((ncol, nrow), Image.Resampling.LANCZOS)
        
        # Salva l'immagine nella cartella corrente
        img_resized.save(output_path)
        print(f"Immagine ridimensionata salvata: {output_path}")
        print(f"Nuove dimensioni: {ncol}x{nrow} (larghezza x altezza)")
        
        return True
        
    except Exception as e:
        print(f"Errore: {e}")
        return False

def main():
    # Controlla gli argomenti
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
    
    # Percorsi
    input_path = "/home/gsorrentino/projects/trilli_sw_prog/trilli-private/sw/dataset/size_32_32/IM1.png"
    output_path = "IM1.png"  # Salva nella cartella corrente
    
    # Verifica che il file di input esista
    if not os.path.exists(input_path):
        print(f"Errore: il file {input_path} non esiste")
        sys.exit(1)
    
    # Esegui il resize
    success = resize_image(input_path, output_path, nrow, ncol)
    
    if success:
        print(f"\nRidimensionamento completato con successo!")
        print(f"L'immagine IM1.png è stata salvata nella cartella corrente con dimensioni {nrow}x{ncol}")
    else:
        print("\nRidimensionamento fallito")
        sys.exit(1)

if __name__ == "__main__":
    main()