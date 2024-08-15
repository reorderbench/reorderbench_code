# Deep Reordering Model

The extensive and diverse matrices in ReorderBench offer valuable supervision for training a deep reordering model. 
By treating the matrices with index swaps as negative samples and their ground-truth matrices as positive samples, we build a deep model for matrix reordering.

## Quick start

### 1. Download data

Download the ReorderBench test set from [here](https://huggingface.co/datasets/reorderbench/ReorderBench), the unified scoring model from [here](https://huggingface.co/reorderbench/unified_scoring_model), and the source code from [here](https://github.com/reorderbench/reorderbench_code/tree/main/reordering_model). 

### 2. Setup environment

```bash
# install from requirements.txt
pip3 install -r requirements.txt
```
### 3. Run

```bash
python test.py \
    --data_path <path> \ # the path to swap_dic.npz file
    --model_path <path> \ # the path to the deep reordering model checkpoint
    --scorer_path <path> \ # the name to the convext-tiny unified scoring model checkpoint
    --pattern_type <str> \ # one of block, offblock, star, and band
    --mat_size <int> \ # the size of the matrix, one of 100, 200, 300, 400
    --tta <int> \ # the number of noise levels for test-time augmentation, one of 1, 5, 17
    --continuous_eval \ # for continuous matrices only
```