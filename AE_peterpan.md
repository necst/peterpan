# Artifact Evaluation

This document describes how to reproduce the results presented in the paper "Adaptive AIE–PL Systems for Efficient End-to-End Pyramidal 3D Image Registration", submitted to the 34th IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM 2026).

Specifically, we provide instructions on how to compile the host applications and run the experiments using the prepared bitstreams to reproduce the results presented in the paper.

The following figures can be reproduced:

- [Figure 6](#figure-6-registration-step) — Registration Step
- [Figure 7](#figure-7-pyramidal-level-scaling) — Pyramidal Level Scaling
- [Figure 8](#figure-8-end-to-end-pyramidal-3d-image-registration) — End-to-End Pyramidal 3D Image Registration

---

# Abstract

3D rigid image registration is a pivotal procedure in computer vision that aligns a floating volume with a reference one to correct positional and rotational distortions. It serves either as a stand-alone process or as a pre-processing step for non-rigid registration, where the rigid part dominates the computational cost.

Various hardware accelerators have been proposed to optimize its compute-intensive components: geometric transformation with interpolation and similarity metric computation. However, existing solutions fail to address both components effectively, as GPUs excel at image transformation, while FPGAs excel in similarity metric computation.

To close this gap, we propose **PeterPan**, a Versal-based accelerator for 3D rigid image registration. PeterPan optimally maps each computational step on the appropriate heterogeneous hardware component and enables efficient execution of the complete registration workflow on the target heterogeneous platform.

---

# Requirements

## Software

- **Vitis 2023.1**
- **XRT 2024.2**
- **OpenCV ≥ 3.0.0**
- **Python 3.10**
- **PyENV**
- **GCC 11**
- **ITK == 5.3.0**

*Note: Different Linux-based operating systems may work as well.*

## Hardware

- **Device:** Versal platform with **QDMA 2022.2 (PCIe Gen4 x8)**
- **CPU:** AMD EPYC 7V13 64-Core Processor

---

# Experiments Workflow

The folder `bitstreams/` already contains the pre-built bitstream `PeterPan.xclbin` used for the evaluation.

You can directly use this bitstream to reproduce the results.  
Alternatively, you may rebuild the bitstream by following the instructions below. In that case, replace the existing `PeterPan.xclbin` in the `bitstreams/` folder with the newly generated one.

The hardware configuration used in the experiments is defined in `default.cfg` and does not require modifications.

```
# ARCHITECTURE PARAMETERS (any change requires hardware build)

DIMENSION := 512
N_COUPLES_MAX := 512
HIST_PE := 16
ENTROPY_PE := 4
INT_PE := 128
PIXELS_PER_READ := 32
DS_PE := 8
```

---

# 4. Preliminary Steps

## 4.a Clone the repository

```
git clone https://github.com/necst/peterpan.git
```

## 4.b Move into the repository

```
cd peterpan
```

## 4.c Source Vitis and XRT

```
source <PATH_TO_XRT>/setup.sh
source <PATH_TO_VITIS>/settings64.sh # Only required if building a new bitstream
```

## 4.d (Optional) Build PeterPan Design

Skip this step if you want to use the provided bitstream.

```
make config
make build_hw TARGET=hw TASK=STEP
```

If you build the bitstream, replace the existing `PeterPan.xclbin` in the `bitstreams/` folder with the generated one.

---

## 4.e Dependencies Installation

### 4.e.1 OpenCV Installation

```
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
mkdir -p ~/opencv_build/opencv/build && cd ~/opencv_build/opencv/build

cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D CMAKE_INSTALL_PREFIX=$HOME/local -D BUILD_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules ..

make -j
make install
```

To add OpenCV to the PATH through the `.bashrc` file:

```
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
```

Optional dependencies: to be installed in case of failures during opencv install process:

```
sudo apt install build-essential cmake git libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev openexr libatlas-base-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev gfortran -y
```

### 4.e.2 ITK Installation

```
wget https://codeload.github.com/InsightSoftwareConsortium/ITK/zip/refs/tags/v5.3.0
unzip v5.3.0
cd ITK-5.3.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Alternatively:

```
sudo apt install libinsighttoolkit5-dev
```

### 4.e.3 PyENV Installation

```
sudo apt update; sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev \
xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

---

# 5. Prepare the build

```
export PATH=/usr/bin:$PATH
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
export OPENCV_DIR=$HOME/opencv_build/opencv/build
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
./prepare.sh
```

This will create the `build/` directory containing the `.zip` files for the experiments.

---

# 6. Deployment

Move the `build/` directory to the deployment machine.

---

# Figure 6: Image Registration Step Comparison

```
unzip PeterPan_STEP
cd build/PeterPan_STEP/sw
make build_sw
./run_registration_step.sh <device_id>
```

- `<device_id>`: Device ID for the target platform

The output image will be in the running folder, named `figure6.pdf`.

---

# Figure 7 & 8: Preliminary

```
unzip PeterPan_APP
cd build/PeterPan_APP
chmod u+x ./dataset_downloader.sh
./dataset_downloader.sh
```

---

# Figure 7: Pyramidal Levels Exploration

```
cd PeterPan_APP/3DIRG_Application/
./ae.sh HW <OpenCV_DIR> <ITK_DIR> <FIXED_images_dir> <MOVING_images_dir> <xclbin_path> <device_id> 7

// Example
./ae.sh HW /home/USER/path/to/opencv/build/ /home/USER/path/to/ITK-5.3.0/build/ /home/USER/path/to/FIXED/ /home/USER/path/to/MOVING/ /home/USER/path/to/PeterPan.xclbin 0 7
```

- `<OpenCV_DIR>`: Path to OpenCV CMake config folder  
- `<ITK_DIR>`: Path to ITK CMake config folder  
- `<FIXED_images_dir>`: Path to Fixed Images Folder  
- `<MOVING_images_dir>`: Path to Moving Images Folder  
- `<xclbin_path>`: Path to XCLBIN file  
- `<device_id>`: Device ID for the target platform  
- Experiment number corresponding to the paper figure  

The output image will be in the running folder, named `figure7.pdf`.

---

# Figure 8: End-To-End Image Registration

```
cd PeterPan_APP/3DIRG_application/
```

**PeterPan Rigid Registration**
```
./ae.sh HW <opencv_dir> <itk_dir> <FIXED_folder> <MOVING_folder> <xclbin_path> <device_id> 8 auto
```

**PeterPan Only Pyramidal (No Subvolume extraction)**
```
./ae.sh HW <opencv_dir> <itk_dir> <FIXED_folder> <MOVING_folder> <xclbin_path> <device_id> 8 fixed
```

NOTE:

Only the `auto` mode generates Figure 8.

- `<OpenCV_DIR>`: Path to OpenCV CMake config folder  
- `<ITK_DIR>`: Path to ITK CMake config folder  
- `<fixed_images_dir>`: Path to Fixed Images Folder  
- `<moving_images_dir>`: Path to Moving Images Folder  
- `<xclbin_path>`: Path to XCLBIN file  
- `<device_id>`: Device ID for the target platform  
- Experiment number corresponding to the paper figure  
- Subvolume selection mode: `fixed` disables it  

The output image will be in the running folder, named `figure8.pdf`.
