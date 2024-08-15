# ReorderBench Generator

To ensure the representativeness and diversity of the benchmark, we first generate a set of representative matrix templates for each visual pattern. Then, based on these matrix templates, a large number of matrix variations with diverse degrees of degeneration are generated. The diversity is achieved by combining different variation methods, including index swapping and two types of typical anti-patterns: noise anti-patterns and noise-cluster anti-patterns. To accurately evaluate the quality of visual patterns in a matrix, we develop a scoring method by combining the matching capability of convolutional kernels and the disorder detection capability of entropy.

## Quick start

### 1. Setup environment

```bash
# install from requirements.txt
pip3 install -r requirements.txt
```
### 2. Run

```bash
python reorderbench_generator.py \
    --train_dir <path> \ # The path to the training data
    --train_template_num <int> \ # the number of templates to generate
    --pattern_type <str> \ # the pattern combination, one of 1000 (block), 0100 (star), 0010 (offblock), 0001 (band)
    --mat_size <int> \ # the size of the matrix, one of 100, 200, 300, 400
    --with_test \ # for generate test data
    --continuous \ # for continuous data
```