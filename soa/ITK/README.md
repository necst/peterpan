## Image ToolKit

The following readme supports the evaluation of ITK for 3D Image Registration. Each code has been built starting from the ITK guides. 

To reproduce the complete 3D image registration, please contact the authors, we will privately send you the datasets. 
We use Powell as optimizer to be fair with the proposed powell-based 3D image registration. Then, we rely on perf to evaluate both transformation and registration step (TX+Interpolato+MI) latencies. 

*Note: * This test takes around 45 minutes to be executed


### Requirements: 
- *Cmake 3.10*
- *C++ 17*
- *ITK library - Package: libinsighttoolkit5-dev - Version: 5.2.1-3*
- *perf*

### Step 1: create repositories
```bash
mkdir build_pow
cd build_pow
```

### Step 2: prepare and build cmake
```bash
cmake .. -DSRCS=itk_pow.cpp
make -j
```

### Step 3: Execute the test
```bash
./execute_tests.sh
```

### How to obtain latencies for transformation and registration step: 
We run the ITK test using perf, analyzing the obtained report to assess the latenies of both transformation and rigid registration step (transformation + interpolation + mutual information)