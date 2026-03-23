import os
import glob
import numpy as np
from PIL import Image

folder1 = "/home/gsorrentino/projects/trilli_sw_prog/trilli-private/data_movers/testbench/dataset_output_new"
folder2 = "/home/gsorrentino/projects/trilli/data_movers/testbench/dataset_output_new"

output_diff_folder = "stat_diff_slices"
os.makedirs(output_diff_folder, exist_ok=True)

files1 = sorted(glob.glob(os.path.join(folder1, "*.png")))
files2 = sorted(glob.glob(os.path.join(folder2, "*.png")))

total_mse = 0.0
total_abs_diff = 0.0
global_max_diff = -np.inf
global_min_diff = np.inf

for idx, (f1, f2) in enumerate(zip(files1, files2)):
    img1 = np.array(Image.open(f1).convert("L"), dtype=np.float32)
    img2 = np.array(Image.open(f2).convert("L"), dtype=np.float32)

    if img1.shape != img2.shape:
        print(f"[WARNING] Immagini {idx} hanno dimensioni diverse: {img1.shape} vs {img2.shape}")
        continue

    diff = np.abs(img1 - img2)
    mse = np.mean(diff**2)
    abs_diff = np.mean(diff)
    max_pixel_diff = diff.max()
    min_pixel_diff = diff.min()

    total_mse += mse
    total_abs_diff += abs_diff
    global_max_diff = max(global_max_diff, max_pixel_diff)
    global_min_diff = min(global_min_diff, min_pixel_diff)

    # --- save the difference as a scaled PNG ---
    if max_pixel_diff > 0:
        diff_scaled = (diff / max_pixel_diff * 255).astype(np.uint8)
    else:
        diff_scaled = np.zeros_like(diff, dtype=np.uint8)

    diff_path = os.path.join(output_diff_folder, f"diff_{idx:02d}.png")
    Image.fromarray(diff_scaled).save(diff_path)

num_images = len(files1)
print("\n=== Statistiche globali ===")
print(f"Numero immagini confrontate: {num_images}")
print(f"MSE media: {total_mse/num_images:.2f}")
print(f"MAE media: {total_abs_diff/num_images:.2f}")
print(f"Diff minima assoluta su tutte le immagini: {global_min_diff}")
print(f"Diff massima assoluta su tutte le immagini: {global_max_diff}")
print(f"Differenze slice salvate in: {output_diff_folder}")
