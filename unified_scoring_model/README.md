# Unified Scoring model

We build a unified scoring model based on the ReorderBench. This model aligns with the convolution- and entropy-based scoring method across all four visual patterns in both binary and continuous matrices and can also measure matrices of varying sizes.

## Quick start

### 1. Download data

Download the ReorderBench test set from [here](https://huggingface.co/datasets/reorderbench/ReorderBench), the unified scoring model from [here](https://huggingface.co/reorderbench/unified_scoring_model), and the source code from [here](https://github.com/reorderbench/reorderbench_code/tree/main/unified_scoring_model). 

### 2. Setup environment

```bash
# install from requirements.txt
pip3 install -r requirements.txt
```

### 3. Run

```bash
python test.py \
    --data_folder <path> \ # the path to the test set folder
    --model_path <path> \ # the path to the unified scoring model checkpoint
    --model_type <str> \ # one of convnext, res50, vgg16
```

To test on a single matrix:

```bash
python test_single.py \
    --matrix_path <path> \ # the path to the matrix in examples folder
    --model_path <path> \ # the path to the convnext-tiny unified scoring model checkpoint
```
## Steps to Reproduce the Results of the Unified Scoring Model

### Figure 15: Scoring Results for Example Matrices
The steps below will reproduce the scoring result for example matrices. This corresponds to **Figure 15** (`unified_scoring_model/result_examples.png`) in our paper.

#### 1. Prepare the Environment

Ensure you have a Linux system with Python installed. If Python is not installed, execute the `python_install.sh` script to install Python.

#### 2. Run the Script

Execute the `run_examples.sh` script in the `unified_scoring_model` folder. This script will:

- Automatically install all required dependencies.
- Run the `test_single.py` script to process example matrices.

#### 3. View the Results

After running the script:

- **Terminal Output**: Scoring results will be displayed directly in the terminal.  
- **Saved Results**: A `.txt` file with scoring results and `.png` images visualizing the matrices will be saved.

Both files will be located in the `unified_scoring_model` folder.

### Table IV: Scoring Results for ReorderBench test set

The steps below will reproduce the scoring result for the ReorderBench test set. This corresponds to **Table IV** (`unified_scoring_model/result_benchmark.png`) in our paper.

#### 1. Prepare the Environment

Ensure you have a Linux system with Python installed.

#### 2. Run the Script

Execute the `run_benchmark.sh` script in the `unified_scoring_model` folder. This script will:

- Automatically install all required dependencies and download required data and models.
- Run the `test.py` script to test the performance of deep neural networks as the unified scoring model on the ReorderBench test set.

#### 3. View the Results

After running the script (this may take several hours):

- **Saved Results**: Three `.txt` files (convnext_results.txt, vgg16_results.txt, res50_results.txt) will be saved. The scoring accuracy results for Table IV can be found at the end of each output file. As for the time results, please note that they may vary slightly depending on the machine and the presence of other running programs. As such, our script does not include the reproduction of time results.
- **Issue**: We have observed that the results for ResNet-50 differ from those reported in the paper. This discrepancy is due to an error when uploading the checkpoint for ResNet-50. Unfortunately, the model was trained during the first author's internship at Microsoft and is no longer accessible. We are currently working on retraining ResNet-50 to address this issue. However, due to the inherent variability in the training process, it may not be easy to reproduce the exact same results. This does not affect our experiments because the ConvNext model is optimal and is used in other experiments.