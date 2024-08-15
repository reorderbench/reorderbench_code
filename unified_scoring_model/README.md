# Unified Scoring model

We build a unified scoring model based on the ReorderBench. This model aligns with the convolution- and entropy-based scoring method across all four visual patterns in both binary and continuous matrices and can also measure matrices of varying sizes.

## Quick start

### 1. Download data

Download the ReorderBench test set from [here](https://huggingface.co/datasets/reorderbench/ReorderBench) and the source code from [here](https://github.com/reorderbench/reorderbench_code/tree/main/unified_scoring_model). 

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