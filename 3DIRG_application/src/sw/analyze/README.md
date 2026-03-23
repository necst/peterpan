# Analize Library

The present project contains the library to analyze the volume

## Library Description

[To Be Filled With Component Specific to Analyze the Volume]

## Build

'''
mkdir build
cd build
cmake ..     -DITK_DIR=/scratch/gsorrentino/ITK-install/lib/cmake/ITK-5.3
make -j
make install
'''

## Prerequisites

The following libraries must be installed

### ITK

```
wget https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v5.3.0.tar.gz
tar -xvf v5.3.0.tar.gz
mv ITK-5.3.0 ITK-src
mkdir ITK-build
cd ITK-build
cmake ../ITK-src     -DCMAKE_INSTALL_PREFIX=/scratch/gsorrentino/ITK-install     -DBUILD_SHARED_LIBS=ON     -DCMAKE_BUILD_TYPE=Release
make -j
make install
```

## OpenCV
To install OpenCV on your machine, use the following commands:
```
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
mkdir -p ~/opencv_build/opencv/build && cd ~/opencv_build/opencv/build

cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D CMAKE_INSTALL_PREFIX=$HOME/local -D BUILD_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules ..

make -j
make install
```

To add your OpenCV to the PATH through the .bashrc file, modify the .bashrc file as follows:
```
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
```

Depending on the system, some sub-dependencies may be needed: 
```
sudo apt install build-essential cmake git libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev openexr libatlas-base-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev gfortran -y
```