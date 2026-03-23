## AE: Matlab Image Registration Toolkit

We provide a code-base to evaluate matlab image registration toolkit. 
As matlab itself offer 1+1 optimizer only, we use such code-base and matlab profiler to evaluate the transformation and rigid step time. 

*To reproduce the complete 3D image registration, please contact the authors or refer to [ATHENA](https://github.com/necst/athena). We will privately send you the dataset, that are required to ensure a correct image registration step.*

Please notice that an exemplifying result is already provided in each paper_fig subfolder and can be used to reproduce paper figures.
 
### How To evaluate
All these steps are meant to be executed into MATLAB environment, through command line. 

# Step 1: Run Profiler
```MATLAB
profile on
```
# Step 2: Run image registration
```MATLAB
imageregistration
```

Run the full image registration

# Step 3: Stop profiler
```MATLAB
profile off
```

# Step 4: View the Report
Check function `imwarp` for geometric transformation, and function `imregister` for the complete registration step.
```MATLAB 
profile viewer
```


