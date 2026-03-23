# -----------------------------
# Helper functions
# -----------------------------
import numpy as np
import dataloader

def estimate_background_mode(data: np.ndarray) -> int:
    """Estimate the most frequent value (mode) in a volume, used as background."""
    data = data.astype(np.uint8)
    values, counts = np.unique(data, return_counts=True)
    return values[np.argmax(counts)]

def replace_background(data: np.ndarray, old_bg_value: int, new_bg_value: int) -> np.ndarray:
    """Replace old background values with a new value to make volumes consistent."""
    data = data.astype(np.uint8)
    data[data == old_bg_value] = new_bg_value
    return data

def pad_to_resolution(volume: np.ndarray, target_dims: tuple) -> np.ndarray:
    """Pad a volume to the target dimensions (cols, rows, depth)."""
    col_padded, row_padded, depth_padded = target_dims
    return dataloader.to_resolution(volume, col_padded, row_padded, depth_padded)

def create_flat_buffers(vol: np.ndarray, border_padding: int = 0) -> tuple:
    """Convert volume to interlaced flat buffers."""
    return dataloader.read_volume_with_datalayout_from_array(vol, border_padding=border_padding, depth_padding=0)

def crop_volume(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop a padded volume back to the target shape (rows, cols, depth)."""
    target_rows, target_cols, target_depth = target_shape
    rows, cols, depth = volume.shape
    crop_row_start = (rows - target_rows) // 2
    crop_col_start = (cols - target_cols) // 2
    crop_row_end = crop_row_start + target_rows
    crop_col_end = crop_col_start + target_cols
    return volume[crop_row_start:crop_row_end, crop_col_start:crop_col_end, 0:target_depth]

def rebuild_volume_from_flat(flat_buffer: np.ndarray, shape: tuple) -> np.ndarray:
    """Rebuild 3D volume from interlaced flat buffer."""
    n_row, n_col, depth = shape
    volume = np.zeros((n_row, n_col, depth), dtype=np.uint8)
    slice_area = n_row * n_col
    for z in range(depth):
        slice_flat = [flat_buffer[i * depth + z] for i in range(slice_area)]
        volume[:, :, z] = np.array(slice_flat, dtype=np.uint8).reshape((n_row, n_col), order='C')
    return volume
