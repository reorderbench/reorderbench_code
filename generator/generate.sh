#!/bin/bash

# Install requirements
pip install -r requirements.txt

# Run the generator
python reorderbench_generator.py --train_dir ./dataset/block_binary_100 --train_template_num 180 --pattern_type 1000 --seed 42 --mat_size 100 --with_test
# python reorderbench_generator.py --train_dir ./dataset/block_binary_200 --train_template_num 180 --pattern_type 1000 --seed 42 --mat_size 200 --with_test
# python reorderbench_generator.py --train_dir ./dataset/block_binary_300 --train_template_num 180 --pattern_type 1000 --seed 42 --mat_size 300 --with_test
# python reorderbench_generator.py --train_dir ./dataset/block_binary_400 --train_template_num 180 --pattern_type 1000 --seed 42 --mat_size 400 --with_test
# python reorderbench_generator.py --train_dir ./dataset/block_continuous_100 --train_template_num 360 --pattern_type 1000 --seed 42 --mat_size 100 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/block_continuous_200 --train_template_num 360 --pattern_type 1000 --seed 42 --mat_size 200 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/block_continuous_300 --train_template_num 360 --pattern_type 1000 --seed 42 --mat_size 300 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/block_continuous_400 --train_template_num 360 --pattern_type 1000 --seed 42 --mat_size 400 --with_test --continuous

# python reorderbench_generator.py --train_dir ./dataset/offblock_binary_100 --train_template_num 180 --pattern_type 0100 --seed 42 --mat_size 100 --with_test
# python reorderbench_generator.py --train_dir ./dataset/offblock_binary_200 --train_template_num 180 --pattern_type 0100 --seed 42 --mat_size 200 --with_test
# python reorderbench_generator.py --train_dir ./dataset/offblock_binary_300 --train_template_num 180 --pattern_type 0100 --seed 42 --mat_size 300 --with_test
# python reorderbench_generator.py --train_dir ./dataset/offblock_binary_400 --train_template_num 180 --pattern_type 0100 --seed 42 --mat_size 400 --with_test
# python reorderbench_generator.py --train_dir ./dataset/offblock_continuous_100 --train_template_num 360 --pattern_type 0100 --seed 42 --mat_size 100 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/offblock_continuous_200 --train_template_num 360 --pattern_type 0100 --seed 42 --mat_size 200 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/offblock_continuous_300 --train_template_num 360 --pattern_type 0100 --seed 42 --mat_size 300 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/offblock_continuous_400 --train_template_num 360 --pattern_type 0100 --seed 42 --mat_size 400 --with_test --continuous

# python reorderbench_generator.py --train_dir ./dataset/star_binary_100 --train_template_num 180 --pattern_type 0010 --seed 42 --mat_size 100 --with_test
# python reorderbench_generator.py --train_dir ./dataset/star_binary_200 --train_template_num 180 --pattern_type 0010 --seed 42 --mat_size 200 --with_test
# python reorderbench_generator.py --train_dir ./dataset/star_binary_300 --train_template_num 180 --pattern_type 0010 --seed 42 --mat_size 300 --with_test
# python reorderbench_generator.py --train_dir ./dataset/star_binary_400 --train_template_num 180 --pattern_type 0010 --seed 42 --mat_size 400 --with_test
# python reorderbench_generator.py --train_dir ./dataset/star_continuous_100 --train_template_num 360 --pattern_type 0010 --seed 42 --mat_size 100 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/star_continuous_200 --train_template_num 360 --pattern_type 0010 --seed 42 --mat_size 200 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/star_continuous_300 --train_template_num 360 --pattern_type 0010 --seed 42 --mat_size 300 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/star_continuous_400 --train_template_num 360 --pattern_type 0010 --seed 42 --mat_size 400 --with_test --continuous

# python reorderbench_generator.py --train_dir ./dataset/band_binary_100 --train_template_num 180 --pattern_type 0001 --seed 42 --mat_size 100 --with_test
# python reorderbench_generator.py --train_dir ./dataset/band_binary_200 --train_template_num 180 --pattern_type 0001 --seed 42 --mat_size 200 --with_test
# python reorderbench_generator.py --train_dir ./dataset/band_binary_300 --train_template_num 180 --pattern_type 0001 --seed 42 --mat_size 300 --with_test
# python reorderbench_generator.py --train_dir ./dataset/band_binary_400 --train_template_num 180 --pattern_type 0001 --seed 42 --mat_size 400 --with_test
# python reorderbench_generator.py --train_dir ./dataset/band_continuous_100 --train_template_num 360 --pattern_type 0001 --seed 42 --mat_size 100 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/band_continuous_200 --train_template_num 360 --pattern_type 0001 --seed 42 --mat_size 200 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/band_continuous_300 --train_template_num 360 --pattern_type 0001 --seed 42 --mat_size 300 --with_test --continuous
# python reorderbench_generator.py --train_dir ./dataset/band_continuous_400 --train_template_num 360 --pattern_type 0001 --seed 42 --mat_size 400 --with_test --continuous



