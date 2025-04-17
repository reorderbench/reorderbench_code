#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt

wget https://huggingface.co/reorderbench/unified_scoring_model/resolve/main/convnext.pth
wget https://huggingface.co/reorderbench/unified_scoring_model/resolve/main/vgg.pth
wget https://huggingface.co/reorderbench/unified_scoring_model/resolve/main/res50.pth

# Download matrices and labels
BASE_URL="https://huggingface.co/datasets/reorderbench/ReorderBench/resolve/main"
TYPES=("block" "offblock" "star" "band")
SIZES=(100 200 300 400)

# Download binary matrix data
for type in "${TYPES[@]}"; do
  for size in "${SIZES[@]}"; do
    DIR="dataset/test_${type}_binary_${size}/IndexSwap"
    mkdir -p "$DIR"
    
    wget -P "$DIR" "$BASE_URL/test_${type}_binary_${size}/IndexSwap/matrices_0.npz"
    
    if [ "$size" -eq 400 ]; then
      wget -P "$DIR" "$BASE_URL/test_${type}_binary_${size}/IndexSwap/matrices_1.npz"
    fi
    
    # Download labels
    wget -P "$DIR" "$BASE_URL/test_${type}_binary_${size}/IndexSwap/labels.npy"
  done
done

# Download continuous matrix data - all four types
for type in "${TYPES[@]}"; do
  for size in "${SIZES[@]}"; do
    DIR="dataset/test_${type}_continuous_${size}/IndexSwap"
    mkdir -p "$DIR"
    
    for i in {0..3}; do
      wget -P "$DIR" "$BASE_URL/test_${type}_continuous_${size}/IndexSwap/matrices_${i}.npz"
    done
    
    # Download labels
    wget -P "$DIR" "$BASE_URL/test_${type}_continuous_${size}/IndexSwap/labels.npy"
  done
done

python test.py --model_path convnext.pth --model_type convnext
python test.py --model_path vgg.pth --model_type vgg
python test.py --model_path res50.pth --model_type res50