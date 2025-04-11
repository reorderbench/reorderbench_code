# ReorderBench: A Benchmark for Matrix Reordering

This repository contains the code for generating benchmark data, unified scoring models, and reordering models described in our paper "ReorderBench: A Benchmark for Matrix Reordering". For more information, please visit https://reorderbench.github.io/.

![alt text](system_pipeline.png)

## Steps to Reproduce the Results of the Unified Scoring Model **(for TVCG Replicability Stamp)**

The steps below will reproduce the scoring result for example matrices. This corresponds to **Figure 15** (`unified_scoring_model/results.png`) in our paper.

### 1. Prepare the Environment

Ensure you have a Linux system with Python installed. If Python is not installed, navigate to the `unified_scoring_model` folder and execute the `python_install.sh` script to install Python.

### 2. Navigate to the Project Directory

Open a terminal and navigate to the `unified_scoring_model` directory.

### 3. Run the Script

Execute the `run.sh` script in the `unified_scoring_model` folder. This script will:

- Automatically install all required dependencies.
- Run the `test_single.py` script to process example matrices.

### 4. View the Results

After running the script:

- **Terminal Output**: Scoring results will be displayed directly in the terminal.  
- **Saved Results**: A `.txt` file with scoring results and `.png` images visualizing the matrices will be saved.

Both files will be located in the `unified_scoring_model` folder.

### Generate Data

Check the folder `generator`. 

Generate matrices with different visual patterns (block, off-diagonal block, star, band) and types (binary, continuous) using `reorderbench_generator`.


### Unified Scoring Models

Check the folder `unified_scoring_model`.

We build unified scoring models based on ReorderBench. These models align with the convolution- and entropy-based scoring method across all four visual patterns in both binary and continuous matrices and can also measure matrices of varying sizes. 

These models can be used to reproduce the scoring results on the ReorderBench test set or to evaluate the quality of visual patterns in any matrix.


### Reordering Models

Check the folder `reordering_model`.

By treating the matrices with index swaps as negative samples and their ground-truth matrices as positive samples, we build deep models for matrix reordering.

These models can be used to reproduce the reordering results of the ReorderBench test set or to reveal the visual patterns of a given matrix.

## Contact

If you have any problem with our code, feel free to contact reorderbench@gmail.com or open an issue.

