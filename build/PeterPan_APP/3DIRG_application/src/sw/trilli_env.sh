# ITK libraries
export LD_LIBRARY_PATH=/scratch/gsorrentino/ITK-install/lib:$LD_LIBRARY_PATH

# OpenCV & local libs
export LD_LIBRARY_PATH=/scratch/gsorrentino/local/lib:$LD_LIBRARY_PATH

# pkg-config for OpenCV (and others)
export PKG_CONFIG_PATH=/scratch/gsorrentino/local/lib/pkgconfig:$PKG_CONFIG_PATH

echo "[env_trilli] Environment correctly loaded."
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
