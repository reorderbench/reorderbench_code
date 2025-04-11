#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt

wget https://huggingface.co/reorderbench/unified_scoring_model/resolve/main/convnext.pth

python test_single.py --matrix_path ./examples/bio-SC-TS_200.npy
python test_single.py --matrix_path ./examples/c-fat200-1_200.npy
python test_single.py --matrix_path ./examples/GD06_theory_200.npy
python test_single.py --matrix_path ./examples/net100_200_perm.npy