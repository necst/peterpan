import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import kornia

def main(tx: float, ty: float, ang: float):
    input_dir = "/home/gsorrentino/projects/trilli_sw_prog/trilli-private/sw/dataset"
    output_dir = "/home/gsorrentino/projects/trilli_sw_prog/trilli-private/data_movers/testbench/ultimate/output_kornia"
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    for idx, fpath in enumerate(files):
        img = np.array(Image.open(fpath).convert("L"))
        img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1,1,H,W)

        B, C, H, W = img_t.shape

        center = torch.tensor([[W/2.0, H/2.0]])
        scale = torch.tensor([[1.0, 1.0]])  # <-- forma corretta (B,2)
        theta = kornia.geometry.transform.get_rotation_matrix2d(center, torch.tensor([ang]), scale=scale)
        theta[:, :, 2] += torch.tensor([tx, ty])

        out = kornia.geometry.transform.warp_affine(img_t, theta, dsize=(H, W))
        out_np = (out.squeeze().clamp(0,1).numpy() * 255).astype(np.uint8)

        out_path = os.path.join(output_dir, f"img_{idx:02d}.png")
        Image.fromarray(out_np).save(out_path)
        print(f"[OK] salvata {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trasforma immagini con Kornia (CPU-only)")
    parser.add_argument("--tx", type=float, required=True, help="Traslazione X (pixel)")
    parser.add_argument("--ty", type=float, required=True, help="Traslazione Y (pixel)")
    parser.add_argument("--ang", type=float, required=True, help="Rotazione in gradi")

    args = parser.parse_args()
    main(args.tx, args.ty, args.ang)
