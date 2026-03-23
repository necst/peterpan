## AE: Athena - 3D Rigid Image Registration on GPUs

This readme supports the Artifact Evaluation process in reproducing Athena results collected for our work. We remark that a copy of these results is already provided in each paper_plot subfolder, to reproduce each of our results. 


### HW Requirements

*Evaluated GPUs:* A5000, A100, RTX4050 - results for V100 are provided in [ATHENA](https://github.com/necst/athena) paper. 

### SW Requirements

*Python 3.8: neeeded to guarantee compatibility with the original Athena repository. Newer versions may work as well, with different times*

All the other requirements can be solved by using:

    ```bash
    ./install_env.sh
    ```

## Reproducing Results: 

Step1: Reproduce single step measurements: 

    ```bash
    ./run_script_single_step.sh
    ```
Note: By default the script targets the GPU0


Step2: Reproduce Complete 3D Image Registration: 

To reproduce the complete 3D image registration, please contact the authors or refer to [ATHENA](https://github.com/necst/athena). We will privately send you the dataset

    ```bash
    ./prepare_data.sh
    ./run_script.sh
    ```
Note: By default the script targets the GPU0


