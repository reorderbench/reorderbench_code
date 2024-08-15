import argparse
import os
import os.path as osp
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from skimage import measure
from tqdm import tqdm

from config import (neg_inf, noise_sample_num, noise_upper, paras_100,
                    paras_200, paras_300, paras_400, scale_factor, train_ratio)
from utils import (add_pattern, calc_ar_deviations_cost, img_write_new,
                   index_swap, pattern_type2str)

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='./dataset/band_binary_100', help='dir to save dataset')
parser.add_argument('--train_template_num', type=int, default=15)
parser.add_argument('--pattern_type', type=str, default='0001', choices=['1000', '0100', '0010', '0001'],
                    help='1000 block | 0100 off-diagonal block | 0010 star | 0001 band')
parser.add_argument('--seed', type=int, default=101)
parser.add_argument('--mat_size', type=int, default=100, choices=[100, 200, 300, 400], help='matrix size')
parser.add_argument('--with_test', action='store_true', default=False,
                    help='also generate test data, default number of test data is 0.1 of train data')
parser.add_argument('--symmetric', action='store_true', default=True, help='symmetric pattern')
parser.add_argument('--continuous', action='store_true', default=False, help='continuous pattern')
args = parser.parse_args()

NOISE_UPPER = noise_upper
NOISE_SAMPLE_NUM = noise_sample_num
SCALE_FACTOR = scale_factor
PARAS = eval('paras_' + str(args.mat_size))
MAT_SIZE = PARAS['MAT_SIZE']
SWAP_NUMS = PARAS['SWAP_NUMS']


"""
    Pattern generators
"""
##  Block
def gen_random_block_patterns(pattern_num=10, sym=True):
    min_length = 3
    def get_reversed_pattern(pattern):
        reversed_pattern = []
        for pat in pattern:
            reversed_pattern.append((MAT_SIZE - pat[0] - pat[2], MAT_SIZE - pat[0] - pat[2], pat[2], pat[2]))
        return reversed_pattern

    patterns = set()
    while len(patterns) < pattern_num:
        block_num = int(len(patterns) * PARAS['num_groups_block_upper'] // pattern_num + 1)
        num_groups = block_num
        block_starts = np.sort(
            np.random.choice(list(range(MAT_SIZE - (min_length - 1) * num_groups)), num_groups, replace=False))
        for i_s, s in enumerate(block_starts):
            block_starts[i_s] += (min_length - 1) * i_s
        chosen_block = np.random.choice(list(range(len(block_starts))), block_num, replace=False)
        chosen_block = np.sort(chosen_block)
        pattern = []
        scanned = 0
        for idx in chosen_block:
            start = block_starts[idx]
            if start < scanned: continue
            if idx != len(block_starts) - 1:
                block_size = np.random.randint(min_length, block_starts[idx + 1] - start + 1)
            else:
                block_size = np.random.randint(min_length, MAT_SIZE - start + 1)
            pattern.append((start, start, block_size, block_size))
            scanned = start + block_size
        assert len(pattern) == block_num
        rnd = np.random.rand()
        if rnd < 0.5:
            patterns.add(tuple(pattern))
        else:
            patterns.add(tuple(get_reversed_pattern(pattern)))
    patterns = list(patterns)
    for i in range(len(patterns)):
        patterns[i] = [list(p) + ['block'] for p in patterns[i]]

    return patterns[:pattern_num]

## Off-block
def gen_random_offblock_patterns(pattern_num=10, sym=True):
    # generate `pattern_num` offblock patterns
    def get_reversed_pattern(pattern):
        reversed_pattern = []
        for pat in pattern:
            if len(pat) == 3:
                reversed_pattern.append((MAT_SIZE - pat[1] - pat[2], MAT_SIZE - pat[0] - pat[2], pat[2]))
            else:
                reversed_pattern.append((MAT_SIZE - pat[1] - pat[3], MAT_SIZE - pat[0] - pat[2], pat[3], pat[2]))
        return reversed_pattern
    def gen_bipart_multi_sym(num_offblocks):
        off_blocks = []
        min_length = 2
        # split matrix into num_groups blocks
        num_groups = 2 * num_offblocks + 2
        starts = np.sort(np.random.choice(list(range(MAT_SIZE - (min_length - 1) * num_groups)), num_groups, replace=False))
        for i_s, s in enumerate(starts):
            starts[i_s] += (min_length - 1) * i_s

        lens = []
        for i in range(len(starts)):
            if i != len(starts) - 1:
                lens.append(starts[i + 1] - starts[i])
            else:
                lens.append(MAT_SIZE - starts[i])

        block_pos = np.triu(np.ones((num_groups, num_groups)), k=1)

        cur = 0
        chosen_block = []
        patterns = []
        while cur < num_offblocks and np.sum(block_pos) > 0:
            t_idx = np.random.randint(0, np.sum(block_pos))
            tmp = np.argwhere(block_pos)[t_idx]
            chosen_block.append(tmp)
            h = np.random.randint(min_length, lens[tmp[0]] + 1)
            w = np.random.randint(min_length, lens[tmp[1]] + 1)
            pattern = (starts[tmp[0]], starts[tmp[1]], h, w)
            for i in range(num_groups):
                block_pos[tmp[0]][i] = 0
                block_pos[i][tmp[0]] = 0
                block_pos[tmp[1]][i] = 0
                block_pos[i][tmp[1]] = 0

            # Allow overlap between offblocks, but not more than half, otherwise they can be merged into larger offblocks
            if tmp[0] > 0:
                max_extend = lens[tmp[0] - 1] // 2
                if max_extend > 0:
                    extend = np.random.randint(0, max_extend + 1)
                    pattern = (pattern[0] - extend, pattern[1], pattern[2] + extend, pattern[3])
                    if extend > 0:
                        block_pos[tmp[0] - 1, max(0, tmp[1] - 1):min(num_groups, tmp[1] + 2)] = 0
            if tmp[0] < min(num_groups - 1, tmp[1] - 1):
                max_extend = lens[tmp[0] + 1] // 2
                if max_extend > 0:
                    extend = np.random.randint(0, max_extend + 1)
                    pattern = (pattern[0], pattern[1], pattern[2] + extend, pattern[3])
                    if extend > 0:
                        block_pos[tmp[0] + 1, max(0, tmp[1] - 1):min(num_groups, tmp[1] + 2)] = 0
            if tmp[1] > max(0, tmp[0] + 1):
                max_extend = lens[tmp[1] - 1] // 2
                if max_extend > 0:
                    extend = np.random.randint(0, max_extend + 1)
                    pattern = (pattern[0], pattern[1] - extend, pattern[2], pattern[3] + extend)
                    if extend > 0:
                        block_pos[max(0, tmp[0] - 1):min(num_groups, tmp[0] + 2), tmp[1] - 1] = 0
            if tmp[1] < num_groups - 1:
                max_extend = lens[tmp[1] + 1] // 2
                if max_extend > 0:
                    extend = np.random.randint(0, max_extend + 1)
                    pattern = (pattern[0], pattern[1], pattern[2], pattern[3] + extend)
                    if extend > 0:
                        block_pos[max(0, tmp[0] - 1):min(num_groups, tmp[0] + 2), tmp[1] + 1] = 0
            patterns.append(pattern)
            cur += 1

        for pattern in patterns:
            off_blocks.append(pattern)
            off_blocks.append((pattern[1], pattern[0], pattern[3], pattern[2]))

        return off_blocks
    patterns = set()
    while len(patterns) < pattern_num:
        num_offblocks = int(len(patterns) * PARAS['num_groups_offblock_upper'] // pattern_num + 1)
        if sym:
            pat = gen_bipart_multi_sym(num_offblocks)
        else:
            raise NotImplementedError
        rnd = np.random.rand()
        if rnd < 0.5:
            patterns.add(tuple(pat))
        else:
            patterns.add(tuple(get_reversed_pattern(pat)))

    patterns = list(patterns)
    for i in range(len(patterns)):
        patterns[i] = [list(p) + ['offblock'] for p in patterns[i]]

    return patterns[:pattern_num]


## Star
def gen_random_star_patterns(pattern_num=10, sym=True):
    # generate `pattern_num` star patterns
    def get_reversed_pattern(pattern):
        reversed_pattern = []
        for pat in pattern:
            reversed_pattern.append((MAT_SIZE - pat[0] - pat[2], MAT_SIZE - pat[1] - pat[3], pat[2], pat[3]))
        return reversed_pattern
    def gen_multi_stars(sym, num_groups):
        # Split matrix into num_groups blocks
        # Avoid overlap
        max_width = 4
        space = int(min(2 * max_width, MAT_SIZE // num_groups - 1))
        starts = np.sort(np.random.choice(list(range(MAT_SIZE - space * num_groups)), num_groups, replace=False))
        for i_s, s in enumerate(starts):
            starts[i_s] += space * i_s

        lens = []
        for i in range(len(starts)):
            if i != len(starts) - 1:
                lens.append(starts[i + 1] - starts[i])
            else:
                lens.append(MAT_SIZE - starts[i])

        stars = []
        for i in range(len(starts)):
            start = starts[i]
            # height range
            h = np.random.randint(1, max_width + 1)
            h = int(min(h, lens[i] // 2))
            h_start = np.random.randint(start, start + lens[i] - h + 1)
            h_end = h_start + h
            w_start = np.random.randint(start, h_start + 1)
            endi = np.random.randint(i, len(starts))
            w_end = np.random.randint(h_end, start + np.sum(lens[i:endi + 1]) + 1)
            while w_end - w_start < 2 * h:
                if w_start > start: w_start -= 1
                if w_end < start + lens[i]: w_end += 1
            if sym:
                stars.append((h_start, w_start, h_end - h_start, w_end - w_start))
                stars.append((w_start, h_start, w_end - w_start, h_end - h_start))
            else:
                pass
        return stars
    patterns = set()
    while len(patterns) < pattern_num:
        num_groups = int(len(patterns) * PARAS['num_groups_star_upper'] // pattern_num + 1)
        pattern = gen_multi_stars(sym, num_groups)
        rnd = np.random.rand()
        if rnd < 0.5:
            patterns.add(tuple(pattern))
        else:
            patterns.add(tuple(get_reversed_pattern(pattern)))
    patterns = list(patterns)
    for i in range(len(patterns)):
        patterns[i] = [list(p) + ['star'] for p in patterns[i]]

    return patterns[:pattern_num]

## Band
def gen_random_band_patterns(pattern_num=10, sym=True):
    # generate `pattern_num` band patterns
    def gen_bands_multi(sym, num_half):
        bands = []
        max_width = 4
        if sym:
            diag_index = np.random.choice(range(1, MAT_SIZE - max_width * (num_half + 3)), num_half, replace=False)
            diag_index = np.sort(diag_index)
            diag_index = [d + max_width * i for i, d in enumerate(diag_index)]
            band_sel = range(len(diag_index))

            for idx in band_sel:
                width = np.random.randint(1, max_width + 1)
                diag_th = diag_index[idx]
                length = np.random.randint(2 * width, MAT_SIZE - diag_th + 1)
                start = np.random.randint(0, MAT_SIZE - diag_th - length + 1)
                x = start
                y = start + diag_th
                if diag_index[idx] == 0:
                    raise ValueError('Diag index cannot be 0')
                else:
                    bands.append((x, y, width, length))
                    bands.append((y, x, width, length))
        else:
            raise NotImplementedError
        return bands
    patterns = set()
    while len(patterns) < pattern_num:
        num_half = int(len(patterns) * PARAS['num_groups_band_upper'] // pattern_num + 1)
        patterns.add(tuple(gen_bands_multi(sym, num_half)))

    patterns = list(patterns)
    for i in range(len(patterns)):
        patterns[i] = [list(p) + ['band'] for p in patterns[i]]

    return patterns[:pattern_num]

# Generate template patterns
def gen_templates(comb, v_num, sym):
    def get_patterns(comb, v_num):
        if comb == '1000':
            return gen_random_block_patterns(v_num, sym)
        elif comb == '0100':
            return gen_random_offblock_patterns(v_num, sym)
        elif comb == '0010':
            return gen_random_star_patterns(v_num, sym)
        elif comb == '0001':
            return gen_random_band_patterns(v_num, sym)
        else:
            raise NotImplementedError
    mat = np.zeros((MAT_SIZE, MAT_SIZE))
    mat_list = {}
    patterns = get_patterns(comb, v_num)
    patterns = sorted(patterns, key=lambda x: len(x))
    for pat in patterns:
        tmp_mat = add_pattern(mat, pat, True)
        pat_num = len(pat)
        if pat_num not in mat_list:
            mat_list[pat_num] = []
        mat_list[pat_num].append({
            'ori_mat': tmp_mat,
            'mat_pattern': pat
        })

    print('gen_templates: ', len(patterns))

    return mat_list


"""
    Score calculators
"""
# Block
def calc_multi_conv_block(mask1_, patterns, mat_size=MAT_SIZE):
    # mask2 is conv kernel
    mask1 = mask1_.copy()
    mask1 = mask1.astype(int)
    sorted_patterns = sorted(patterns, key=lambda x: -x[2])
    indexes = np.arange(len(patterns))
    indexes = sorted(indexes, key=lambda x: -patterns[x][2])

    matched_pos = [None for _ in range(len(patterns))]
    scores = [None for _ in range(len(patterns))]
    L_scores = [None for _ in range(len(patterns))]

    for block in sorted_patterns:
        block_size = block[2]
        conv_result = np.zeros((mat_size - block_size + 1))
        for i in range(mat_size - block_size + 1):
            # if vis[i:i+block_size].any(): continue
            conv_result[i] = mask1[i:i + block_size, i:i + block_size].sum()
        max_conv_result = np.max(conv_result)

        pos = np.where(conv_result == max_conv_result)
        rows = pos[0]
        ridx = np.random.randint(0, len(rows))
        row, col = rows[ridx], rows[ridx]
        mask1[row:row + block_size, col:col + block_size] = neg_inf
        idx = indexes.pop(0)
        scores[idx] = np.max(conv_result) / (block_size * block_size) if max_conv_result > 0 else 0
        # L_scores[idx] = np.sqrt(np.max(conv_result)) / block_size
        matched_pos[idx] = [row, col, block_size, block_size]
    areas = [i[2] * i[3] for i in patterns]
    return {
        'scores': scores,
        'score': np.average(scores, weights=areas),
        'matched_pos': [i + ['block'] for i in matched_pos]
    }

# Off-diagonal block
def calc_multi_conv_offblock(mask1, patterns, mat_size=MAT_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def zero_out_diagonal_elements(matrix, d):
        diag_indices = np.arange(max([matrix.shape[0], matrix.shape[1]]))
        mask = np.abs(diag_indices[:, None] - diag_indices) < d
        matrix *= (1 - mask[:matrix.shape[0], :matrix.shape[1]])
        return matrix

    m1 = torch.tensor(mask1.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to(device)

    upper_pattern = []
    for offblock in patterns:
        if offblock[1] < offblock[0]: continue
        upper_pattern.append(offblock)

    pattern = sorted(upper_pattern, key=lambda x: -x[2] * x[3])
    indexes = np.arange(len(pattern))
    indexes = sorted(indexes, key=lambda x: -upper_pattern[x][2] * upper_pattern[x][3])

    matched_pos = [None for _ in range(len(pattern))]
    scores = [None for _ in range(len(pattern))]

    for offblock in pattern:
        idx = indexes.pop(0)
        x, y, h, w = offblock
        k1 = torch.ones((h, w)).unsqueeze_(0).unsqueeze_(0).to(device)
        k2 = torch.ones((w, h)).unsqueeze_(0).unsqueeze_(0).to(device)
        padding = max(h, w)

        conv_result1 = F.conv2d(m1, k1, stride=1, padding=padding).cpu().numpy()[0][0]
        conv_result2 = F.conv2d(m1, k2, stride=1, padding=padding).cpu().numpy()[0][0]
        conv_result1 = zero_out_diagonal_elements(conv_result1, h)
        conv_result2 = zero_out_diagonal_elements(conv_result2, w)
        conv_result1 = np.triu(conv_result1)
        conv_result2 = np.triu(conv_result2)
        max_conv_result1 = np.max(conv_result1)
        max_conv_result2 = np.max(conv_result2)
        if max_conv_result1 > max_conv_result2:
            score = max_conv_result1 / (w * h) if max_conv_result1 > 0 else 0
            pos = np.where(conv_result1 == max_conv_result1)
            rows = pos[0]
            cols = pos[1]
            # select the pos with min row and col
            r_idx = np.argmin(rows + cols)
            # if row < 0, h in matched_pos will be less than h in mat_pattern
            row, col = rows[r_idx] - padding, cols[r_idx] - padding
            matched_pos[idx] = (row, col, h, w)
            scores[idx] = score
            # remove matched block
            m1[0][0][max(0, row):min(row + h, mat_size), max(0, col):min(col + w, mat_size)] = neg_inf
            m1[0][0][max(0, col):min(col + w, mat_size), max(0, row):min(row + h, mat_size)] = neg_inf
        else:
            score = max_conv_result2 / (w * h) if max_conv_result2 > 0 else 0
            pos = np.where(conv_result2 == max_conv_result2)
            rows = pos[0]
            cols = pos[1]
            r_idx = np.argmin(rows + cols)
            row, col = rows[r_idx] - padding, cols[r_idx] - padding
            matched_pos[idx] = (row, col, w, h)
            scores[idx] = score
            # remove matched block
            m1[0][0][max(0, row):min(row + w, mat_size), max(0, col):min(col + h, mat_size)] = neg_inf
            m1[0][0][max(0, col):min(col + h, mat_size), max(0, row):min(row + w, mat_size)] = neg_inf
    for mp in matched_pos:
        if mp[1] > mp[0] and mp[0] + mp[2] > mp[1]:
            raise ValueError('wrong', mp)
        if mp[0] > mp[1] and mp[1] + mp[3] > mp[0]:
            raise ValueError('wrong', mp)

    all_matched_pos = []
    for pos in matched_pos:
        all_matched_pos.append(list(pos))
        all_matched_pos.append([pos[1], pos[0], pos[3], pos[2]])
    areas = [i[2] * i[3] for i in upper_pattern]
    return {
        'scores': scores,
        'score': np.average(scores, weights=areas),
        'matched_pos': [i + ['offblock'] for i in all_matched_pos]
    }

# Star
def calc_multi_conv_star(mask1, patterns, mat_size=MAT_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = torch.tensor(mask1.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to(device)

    row_pattern = []
    for star in patterns:
        if star[1] > star[0] or star[1] == star[0] and star[2] > star[3]: continue
        row_pattern.append(star)

    pattern = sorted(row_pattern, key=lambda x: (-x[3], -x[2]))
    indexes = np.arange(len(pattern))
    indexes = sorted(indexes, key=lambda x: (-row_pattern[x][3], -row_pattern[x][2]))

    matched_pos = [None for _ in range(len(pattern))]
    scores = [None for _ in range(len(pattern))]
    areas = [None for _ in range(len(pattern))]

    for star in pattern:
        idx = indexes.pop(0)

        hs, ws, h_star, w_star = star
        k = torch.ones((h_star, w_star)).unsqueeze_(0).unsqueeze_(0).to(device)

        conv_result = F.conv2d(m1, k, stride=1, padding=0).cpu().numpy()[0][0]
        best_conv_res = 0
        best_pos = [0, 0]
        for x in range(0, mat_size - h_star + 1):
            y_vals = np.arange(max(0, x + h_star - w_star), min(x, mat_size - w_star) + 1)
            tmp_res = conv_result[x, y_vals]  # - 0.5 * conv_diag[x, x]
            max_index = np.argmax(tmp_res)
            if tmp_res[max_index] > best_conv_res:
                best_conv_res = tmp_res[max_index]
                best_pos = [x, y_vals[max_index]]

        scores[idx] = best_conv_res / (w_star * h_star) if best_conv_res > 0 else 0
        matched_pos[idx] = [[best_pos[0], best_pos[1], h_star, w_star], [best_pos[1], best_pos[0], w_star, h_star]]
        areas[idx] = w_star * h_star
        # remove matched star
        m1[0][0][best_pos[0]:best_pos[0] + h_star, best_pos[1]:best_pos[1] + w_star] = neg_inf
        m1[0][0][best_pos[1]:best_pos[1] + w_star, best_pos[0]:best_pos[0] + h_star] = neg_inf

    all_matched_pos = []
    for pos in matched_pos:
        all_matched_pos.append(pos[0])
        all_matched_pos.append(pos[1])

    return {
        'scores': scores,
        'score': np.average(scores, weights=[i for i in areas]),
        'matched_pos': [i + ['star'] for i in all_matched_pos]
    }

# Band
def calc_multi_conv_band(mask1, patterns, mat_size=MAT_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = torch.tensor(mask1.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to(device)

    upper_pattern = []
    for band in patterns:
        if band[1] < band[0]: continue
        upper_pattern.append(band)

    pattern = sorted(upper_pattern, key=lambda x: (-x[3], -x[2]))
    indexes = np.arange(len(pattern))
    indexes = sorted(indexes, key=lambda x: (-upper_pattern[x][3], -upper_pattern[x][2]))

    matched_pos = [None for _ in range(len(pattern))]
    scores = [None for _ in range(len(pattern))]
    areas = [None for _ in range(len(pattern))]

    for band in pattern:
        idx = indexes.pop(0)
        x, y, w, l = band
        diag_th = y - x

        # diagonal band
        if diag_th == 0:
            scores[idx] = 1
            matched_pos[idx] = [0, 0, w, l]
            areas[idx] = l
            continue

        k = np.zeros((l, l))
        for i in range(l):
            for j in range(i, min([i + w, l])):
                k[i][j] = 1
        assert np.sum(k) == w * l - w * (w - 1) / 2
        areas[idx] = w * l - w * (w - 1) / 2

        # all diagonals
        padding = l
        k1 = torch.tensor(k, dtype=torch.float32).unsqueeze_(0).unsqueeze_(0).to(device)
        conv_result1 = F.conv2d(m1, k1, stride=1, padding=padding).cpu().numpy()[0][0]
        np.fill_diagonal(conv_result1, 0)
        conv_result1 = np.triu(conv_result1)
        best_conv_res = np.max(conv_result1)
        if best_conv_res == 0:
            scores[idx] = 0
            matched_pos[idx] = [0, diag_th, w, l]
        else:
            scores[idx] = best_conv_res / np.sum(k) if best_conv_res > 0 else 0
            pos = np.where(conv_result1 == np.max(conv_result1))
            rows = pos[0]
            cols = pos[1]
            ridx = np.random.randint(0, len(rows))
            row, col = rows[ridx] - padding, cols[ridx] - padding
            matched_pos[idx] = [row, col, w, l]

            # remove related rows and cols
            for i in range(row, row + l):
                for j in range(col + i - row, min(col + i - row + w, col + l)):
                    if i >= 0 and i < mat_size and j >= 0 and j < mat_size:
                        m1[0][0][i][j] = neg_inf

    all_matched_pos = []
    for pos in matched_pos:
        all_matched_pos.append(pos)
        if pos[0] != pos[1]:
            all_matched_pos.append([pos[1], pos[0], pos[2], pos[3]])

    return {
        'scores': scores,
        'score': np.average(scores, weights=[i for i in areas]),
        'matched_pos': [i + ['band'] for i in all_matched_pos]
    }

# Calculate the match score, disorder score and continuous score
def calc_multi_conv(mat, pattern_type, patterns, mat_size=MAT_SIZE):
    if pattern_type == 'block':
        match_res = calc_multi_conv_block(mat, patterns, mat_size)
    elif pattern_type == 'offblock':
        match_res = calc_multi_conv_offblock(mat, patterns, mat_size)
    elif pattern_type == 'star':
        match_res = calc_multi_conv_star(mat, patterns, mat_size)
    elif pattern_type == 'band':
        match_res = calc_multi_conv_band(mat, patterns, mat_size)
    else:
        raise NotImplementedError
    return match_res

def calc_multi_cost_continuous(mat, pattern_type, matched_pos, scores, mat_size=MAT_SIZE):
    scores_cost_coes = []
    if pattern_type == 'band':
        scores_cost_coes = np.ones(len(scores))
    else:
        if pattern_type == 'star' or pattern_type == 'offblock':
            matched_pos = matched_pos[::2]
        i = 0
        for pos in matched_pos:
            i += 1
            scores_cost_coe = calc_ar_deviations_cost(mat, pattern_type, pos, mat_size)
            if pattern_type == 'offblock':
                scores_cost_coe = max(scores_cost_coe,
                                      calc_ar_deviations_cost(mat, pattern_type, pos, mat_size, offdiag=True))
            scores_cost_coes.append(scores_cost_coe)
    scores_cost_coes = np.array(scores_cost_coes)
    scores = scores * scores_cost_coes
    return scores

def calc_disorder_score(mat, match_res, mat_size):
    scores = match_res['scores']
    matched_pos = match_res['matched_pos']
    swapped_noise_mat = mat
    penalty_scores = []
    entropys = []
    areas = []

    for pos_idx, pos in enumerate(matched_pos):
        assert len(pos) == 5
        pattern_type = pos[4]
        pos = pos[:4]
        if pattern_type == 'band':
            x, y, w, l = pos
            if y < x: continue
            area = w * l - w * (w - 1) / 2
            matched_pattern = np.zeros((min(x + l, mat_size) - max(0, x), min(y + l, mat_size) - max(0, y)))
            for i in range(matched_pattern.shape[0]):
                for j in range(i, min([i + w, matched_pattern.shape[1]])):
                    matched_pattern[i, j] = swapped_noise_mat[max(0, x) + i, max(0, y) + j]
        elif pattern_type == 'block':
            x, y, l, l = pos
            area = l ** 2
            matched_pattern = mat[x: x + l, y:y + l].copy()
        elif pattern_type == 'offblock':
            x, y, h, w = pos
            if y < x: continue
            area = h * w
            matched_pattern = mat[max(0, x): min(x + h, mat_size), max(0, y): min(y + w, mat_size)].copy()
        elif pattern_type == 'star':
            hs, ws, h, w = pos
            if h >= w: continue
            he = hs + h
            we = ws + w
            area = h * w
            matched_pattern = np.zeros((w, w))
            matched_pattern[hs - ws:he - ws, ws - ws:we - ws] = swapped_noise_mat[hs:he, ws:we]
            matched_pattern[ws - ws:we - ws, hs - ws:he - ws] = swapped_noise_mat[ws:we, hs:he]
        areas.append(area)
        matched_pattern_binary = matched_pattern > 0.0
        total = np.sum(matched_pattern_binary)
        labels, num_features = measure.label(matched_pattern_binary, connectivity=2, return_num=True)
        component_probabilities = []
        for label in range(1, num_features + 1):
            component_size = np.sum(labels == label)
            component_probabilities.append(component_size / total)
        component_probabilities = np.array(component_probabilities)
        entropy_value = entropy(component_probabilities, base=2)
        entropys.append(entropy_value)
    penalty_scores = scores * (1 - entropys / np.log2(areas))
    penalty_score = np.average(penalty_scores, weights=areas)
    return penalty_score, penalty_scores


"""
    Generate dataset
"""
# Phase 1: Generate matrix templates
def get_noise_cluster_set_size():
    pattern_types = ['1000', '0100', '0010', '0001']
    pattern_size = 0
    pattern_count = 0
    for pattern_type in pattern_types:
        mat_list = gen_templates(pattern_type, 450, True)
        mat_list = [item for r in mat_list.values() for item in r]
        for mat in mat_list:
            mat_pattern = mat['mat_pattern']
            pattern_count += len(mat_pattern)
            for pattern in mat_pattern:
                pattern_size = pattern_size + pattern[2] + pattern[3]
    mean_pattern_size = round(pattern_size / pattern_count)
    print(f'mean pattern size: {mean_pattern_size}')
    return mean_pattern_size

def gen_templates_dataset(variation_list, temp_dir):
    comb2mat = {}
    idx_draw = 0
    for idx, mat_item in tqdm(enumerate(variation_list)):
        pattern_num = len(mat_item['mat_pattern'])
        filename = f'{pattern_type}_{idx}_num_{pattern_num}'
        mat_item['score'] = 1.0
        comb2mat[f'{pattern_type}_{idx}_num_{pattern_num}'] = mat_item
        if idx_draw % 5 == 0:
            img_write_new(osp.join(temp_dir, filename + '.png'), mat_item['ori_mat'], scale_factor=SCALE_FACTOR)
        idx_draw += 1

    np.savez_compressed(osp.join(temp_dir, 'temp_dic.npz'), **comb2mat)

def gen_continuous_dataset(temp_dir, continuous_dir):
    def gen_random_robinson(length):
        r = np.sort(np.random.rand(length))
        diff_matrix = (r[:, np.newaxis] - r) ** 2
        d = 1 - diff_matrix
        return d

    def gen_continuous_pattern(ori_mat, pattern):
        def block_continuous(ori_mat, pattern):
            x, y, pattern_size = pattern[:3]
            mat = gen_random_robinson(pattern_size)
            ori_mat[x:x + pattern_size, y:y + pattern_size] = mat
            return ori_mat, mat

        def offblock_continuous(ori_mat, pattern):
            x, y, h, w = pattern[:4]
            mat = gen_random_robinson(max(h, w))
            mat = np.flip(mat, axis=1)
            sx, sy = max(h, w) - h, 0
            mat = mat[sx:sx + h, sy:sy + w]
            ori_mat[x:x + h, y:y + w] = mat
            ori_mat[y:y + w, x:x + h] = mat.T
            return ori_mat, mat[:h, h:]

        def band_continuous(ori_mat, pattern):
            x, y, width, length = pattern[:4]
            period = length
            base_nums = np.random.rand(period)
            for w in range(width):
                nums = np.random.rand(period)
                for row in range(x, x + length):
                    col = y + row - x + w
                    if col >= min(MAT_SIZE, y + length):
                        break
                    ori_mat[row, col] = ori_mat[row, col] * nums[row % period]
                    ori_mat[col, row] = ori_mat[row, col]
            # extract pattern
            band = np.zeros((length, length))
            for i in range(length):
                for j in range(width):
                    if i + j >= length:
                        break
                    band[i, i + j] = ori_mat[x + i, y + i + j]
            return ori_mat, band

        def star_continuous(ori_mat, pattern):
            hs, ws, h_star, w_star = pattern[:4]
            he, we = hs + h_star, ws + w_star
            hs = max(0, hs)
            ws = max(0, ws)
            he = min(he, MAT_SIZE)
            we = min(we, MAT_SIZE)
            mat = gen_random_robinson(w_star)
            ori_mat[hs:he, ws:we] = mat[hs - ws:he - ws, ws - ws:we - ws]
            ori_mat[ws:we, hs:he] = mat[ws - ws:we - ws, hs - ws:he - ws]
            return ori_mat, ori_mat[hs:he, ws:we]

        pattern_type = pattern[-1]
        eval_pattern = [int(i) for i in pattern[:-1]]
        if pattern_type == 'block':
            continuous_func = block_continuous
        elif pattern_type == 'offblock':
            continuous_func = offblock_continuous
        elif pattern_type == 'star':
            continuous_func = star_continuous
        elif pattern_type == 'band':
            continuous_func = band_continuous
        else:
            raise NotImplementedError
        ori_mat, continuous_pattern = continuous_func(ori_mat, eval_pattern)
        return ori_mat

    temp_dic = np.load(osp.join(temp_dir, 'temp_dic.npz'), allow_pickle=True)
    continuous_dic = {}
    mat_idx = 0
    for mat_name, mat_item in tqdm(temp_dic.items()):
        mat_item = mat_item.item()
        ori_mat = mat_item['ori_mat']
        mat_pattern = mat_item['mat_pattern']
        continuous_mat = ori_mat.copy()
        mat_pattern = np.array(mat_pattern)
        effectivenesses = []
        for pattern in mat_pattern:
            pattern_type = pattern[-1]
            if pattern_type == 'offblock' or pattern_type == 'band':
                if int(pattern[0]) > int(pattern[1]):
                    continue
            elif pattern_type == 'star':
                if int(pattern[2]) >= int(pattern[3]):
                    continue
            continuous_mat = gen_continuous_pattern(continuous_mat, pattern)
        if mat_idx % 5 == 0:
            img_write_new(osp.join(continuous_dir, mat_name + '.png'), continuous_mat, scale_factor=SCALE_FACTOR)
        continuous_dic[mat_name] = {
            'ori_mat': ori_mat,
            'continuous_mat': continuous_mat,
            'mat_pattern': mat_pattern,
        }
        mat_idx += 1
    np.savez_compressed(osp.join(continuous_dir, 'continuous_dic.npz'), **continuous_dic)

# Phase 2: Generate noise dataset
def gen_noise_dataset(temp_dir, noise_dir, set_size, continuous=False):
    def random_noise_mask(noise_mask, noise_num, mat_size=MAT_SIZE):
        rows, cols = np.triu_indices(mat_size, k=0)
        chosen_idx = np.random.choice(list(range(len(rows))), size=noise_num, replace=False)
        noise_mask[rows[chosen_idx], cols[chosen_idx]] = 1 - noise_mask[rows[chosen_idx], cols[chosen_idx]]
        noise_mask[cols[chosen_idx], rows[chosen_idx]] = noise_mask[rows[chosen_idx], cols[chosen_idx]]
        return noise_mask

    def noise_cluster_mask(noise_mask, mat_pattern, row_noise, set_size, mat_size=MAT_SIZE):
        noise_mat = np.zeros((mat_size, mat_size), dtype=bool)
        noise_row_patterns = []
        choices = range(mat_size)
        for _ in range(set_size):
            noise_row_pattern = np.zeros(mat_size)
            chosen_rows = np.random.choice(choices, size=row_noise, replace=False)
            noise_row_pattern[chosen_rows] = 1
            noise_row_patterns.append(noise_row_pattern)
        noise_row_patterns = np.array(noise_row_patterns)
        indexes = np.random.choice(set_size, size=mat_size, replace=True)
        noise_mat[list(range(mat_size))] = noise_row_patterns[indexes]
        noise_mat = np.maximum(noise_mat, noise_mat.T)
        noise_mask[noise_mat] = 1 - noise_mask[noise_mat]
        return noise_mask

    temp_dic = np.load(osp.join(temp_dir, 'temp_dic.npz'), allow_pickle=True)
    noise_upper = NOISE_UPPER
    noise_sample_num = NOISE_SAMPLE_NUM
    noise_dic = {}  # mat_name:{ori_mat:m, ...}
    item_idx = 0
    for mat_name, mat_item in tqdm(temp_dic.items()):
        mat_item = mat_item.item()
        to_draw = item_idx % 5 == 0
        ori_mat = mat_item['ori_mat']
        mat_pattern = mat_item['mat_pattern']
        for idx in range(noise_sample_num):
            noise_ratio = np.random.randint(0, noise_upper + 1)
            row_noise_level = np.random.randint(0, noise_upper + 1)
            noise_mat = ori_mat.copy()
            noise_mask = np.zeros((MAT_SIZE, MAT_SIZE), dtype=bool)
            noise_num = int(MAT_SIZE ** 2 * noise_ratio * 0.01 * 0.5)
            noise_mask = random_noise_mask(noise_mask, noise_num)
            row_noise = int(MAT_SIZE * row_noise_level * 0.01 * 0.5)
            noise_mask = noise_cluster_mask(noise_mask, mat_pattern, row_noise, set_size)
            noise_mat[noise_mask] = 1 - noise_mat[noise_mask]
            noise_mat_name = f'{mat_name}_noise_{idx}-{noise_ratio}-{row_noise_level}'
            noise_dic[noise_mat_name] = {
                'ori_mat': ori_mat,
                'noise_mat': noise_mat,
                'mat_pattern': mat_pattern,
            }
            if to_draw: img_write_new(osp.join(noise_dir, f'{noise_mat_name}.png'), noise_mat,
                                      scale_factor=SCALE_FACTOR)
        item_idx += 1
    np.savez_compressed(osp.join(noise_dir, 'noise_dic.npz'), **noise_dic)
    print('num after adding noise', len(noise_dic))

def gen_noise_dataset_continuous(continuous_dir, noise_dir, set_size):
    def random_noise_mask(noise_mask, noise_num, mat_size=MAT_SIZE):
        rows, cols = np.triu_indices(mat_size, k=0)
        chosen_idx = np.random.choice(list(range(len(rows))), size=noise_num, replace=False)
        noise_mask[rows[chosen_idx], cols[chosen_idx]] = 1 - noise_mask[rows[chosen_idx], cols[chosen_idx]]
        noise_mask[cols[chosen_idx], rows[chosen_idx]] = noise_mask[rows[chosen_idx], cols[chosen_idx]]
        return noise_mask

    def noise_cluster_mask(noise_mask, mat_pattern, row_noise, set_size, mat_size=MAT_SIZE):
        noise_mat = np.zeros((mat_size, mat_size), dtype=bool)
        noise_row_patterns = []
        choices = range(mat_size)
        for _ in range(set_size):
            noise_row_pattern = np.zeros(mat_size)
            chosen_rows = np.random.choice(choices, size=row_noise, replace=False)
            noise_row_pattern[chosen_rows] = 1
            noise_row_patterns.append(noise_row_pattern)
        noise_row_patterns = np.array(noise_row_patterns)
        indexes = np.random.choice(set_size, size=mat_size, replace=True)
        noise_mat[list(range(mat_size))] = noise_row_patterns[indexes]
        noise_mat = np.maximum(noise_mat, noise_mat.T)
        noise_mask[noise_mat] = 1 - noise_mask[noise_mat]
        return noise_mask

    continuous_dic = np.load(osp.join(continuous_dir, 'continuous_dic.npz'), allow_pickle=True)
    noise_upper = NOISE_UPPER
    noise_sample_num = NOISE_SAMPLE_NUM
    noise_dic = {}  # mat_name:{ori_mat:m, ...}
    item_idx = 0
    for mat_name, mat_item in tqdm(continuous_dic.items()):
        mat_item = mat_item.item()
        to_draw = item_idx % 5 == 0
        ori_mat = mat_item['ori_mat']
        continuous_mat = mat_item['continuous_mat']
        mat_pattern = mat_item['mat_pattern']
        for idx in range(noise_sample_num):
            noise_ratio = np.random.randint(0, noise_upper + 1)
            row_noise_level = np.random.randint(0, noise_upper + 1)
            noise_mat = continuous_mat.copy()
            noise_mask = np.zeros((MAT_SIZE, MAT_SIZE), dtype=bool)
            noise_num = int(MAT_SIZE ** 2 * noise_ratio * 0.01 * 0.5)
            noise_mask = random_noise_mask(noise_mask, noise_num)
            row_noise = int(MAT_SIZE * row_noise_level * 0.01 * 0.5)
            noise_mask = noise_cluster_mask(noise_mask, mat_pattern, row_noise, set_size)
            noise_rand = np.random.rand(*noise_mask.shape) * noise_mask
            noise_mat[noise_mask] = noise_rand[noise_mask]
            noise_mat[np.triu_indices(MAT_SIZE, k=1)] = noise_mat.T[np.triu_indices(MAT_SIZE, k=1)]
            noise_mat_name = f'{mat_name}_noise_{idx}-{noise_ratio}-{row_noise_level}'
            noise_dic[noise_mat_name] = {
                'ori_mat': ori_mat,
                'continuous_mat': continuous_mat,
                'noise_mat': noise_mat,
                'mat_pattern': mat_pattern,
            }
            if to_draw: img_write_new(osp.join(noise_dir, f'{noise_mat_name}.png'), noise_mat,
                                      scale_factor=SCALE_FACTOR)
        item_idx += 1
    np.savez_compressed(osp.join(noise_dir, 'noise_dic.npz'), **noise_dic)
    print('num after adding noise', len(noise_dic))

# Phase 3: Generate index-swap dataset
def gen_index_swap_dataset(noise_dir, swap_dir):
    noise_dic = np.load(osp.join(noise_dir, 'noise_dic.npz'), allow_pickle=True)
    swap_nums = SWAP_NUMS
    scores = []
    swap_dic = {}  # mat_name:{ori_mat:m, ...}
    matrices = []
    chunk_idx = 0  # chunk index of large file
    mat_num = 0
    mat_idx = 0

    for mat_name, mat_item in tqdm(noise_dic.items()):
        mat_item = mat_item.item()
        to_draw = mat_idx % 125 == 0
        ori_mat = mat_item['ori_mat']
        noise_mat = mat_item['noise_mat']
        mat_pattern = mat_item['mat_pattern']
        mat_pattern = np.array(mat_pattern)

        for swap_num in swap_nums:
            tmp_mat, tmp_perm = index_swap(noise_mat, swap_num, mat_size=MAT_SIZE)
            if len(mat_pattern[0]) == 4:
                pattern_types = [pattern_type2str(pattern_type)]
            else:
                pattern_types = list(set(mat_pattern[:, -1]))
            if len(pattern_types) == 1:
                pattern_type = pattern_types[0]
                match_res = calc_multi_conv(tmp_mat, pattern_type, np.array(mat_pattern[:, :4]).astype(int))
                match_res['score'], match_res['scores'] = calc_disorder_score(tmp_mat, match_res, MAT_SIZE)
            else:
                raise NotImplementedError
            prefix = f'{mat_name}_swap_{swap_num}_score_{str(round(match_res["score"], 2))}'
            img_path = osp.join(swap_dir, f'{prefix}.png')
            swap_dic[prefix] = {
                'swapped_noise_mat': tmp_mat.astype(bool),
                'noise_mat': noise_mat.astype(bool),
                'ori_mat': ori_mat.astype(bool),
                'mat_pattern': mat_pattern,
                'perm': tmp_perm,
                'img_path': img_path
            }

            for k, v in match_res.items():
                swap_dic[prefix][k] = v

            matrices.append(tmp_mat.astype(bool))
            if len(matrices) >= 200:
                mat_num += len(matrices)
                filename = f'{osp.join(swap_dir, "matrices")}' + f'_{chunk_idx}.npz'
                if 'score' in list(swap_dic.values())[0]:
                    scores.append(np.array([i['score'] for i in swap_dic.values()]))
                np.savez_compressed(filename, matrices=matrices)  # also save swap_dic.npz
                np.savez_compressed(osp.join(swap_dir, f'swap_dic_{chunk_idx}.npz'), **swap_dic)
                print(f'Saved chunk {chunk_idx} to {filename}')
                chunk_idx += 1
                matrices = []
                swap_dic = {}
            if to_draw:
                img_write_new(img_path, tmp_mat, scale_factor=SCALE_FACTOR)
                img_write_new(img_path[:-4] + '_ori_mat.png', ori_mat, scale_factor=SCALE_FACTOR)
                img_write_new(img_path[:-4] + '_swapped_mat.png', tmp_mat, 'hybrid', swap_dic[prefix]['matched_pos'],
                              scale_factor=SCALE_FACTOR)

        mat_idx += 1
    mat_num += len(matrices)
    if len(matrices) > 0:
        filename = f'{osp.join(swap_dir, "matrices")}' + f'_{chunk_idx}.npz'
        np.savez_compressed(filename, matrices=matrices)
        np.savez_compressed(osp.join(swap_dir, f'swap_dic_{chunk_idx}.npz'), **swap_dic)
        print(f'Saved chunk {chunk_idx} to {filename}')
        chunk_idx += 1
    if 'score' in list(swap_dic.values())[0]:
        scores.append(np.array([i['score'] for i in swap_dic.values()]))
        np.save(osp.join(swap_dir, 'labels'), np.concatenate(scores).reshape(-1, 1))
    print('num after index swap', mat_num)

def gen_index_swap_dataset_continuous(noise_dir, swap_dir):
    noise_dic = np.load(osp.join(noise_dir, 'noise_dic.npz'), allow_pickle=True)
    swap_nums = SWAP_NUMS
    scores = []
    swap_dic = {}  # mat_name:{ori_mat:m, ...}
    matrices = []
    chunk_idx = 0  # chunk index of large file
    mat_idx = 0
    mat_num = 0
    for mat_name, mat_item in tqdm(noise_dic.items()):
        mat_item = mat_item.item()
        to_draw = mat_idx % 125 == 0
        ori_mat = mat_item['ori_mat']
        noise_mat = mat_item['noise_mat']
        continuous_mat = mat_item['continuous_mat']
        mat_pattern = mat_item['mat_pattern']
        mat_pattern = np.array(mat_pattern)
        for swap_num in swap_nums:
            tmp_mat, tmp_perm = index_swap(noise_mat, swap_num, mat_size=MAT_SIZE)
            if len(mat_pattern[0]) == 4:
                pattern_types = [pattern_type2str(pattern_type)]
            else:
                pattern_types = list(set(mat_pattern[:, -1]))
            if len(pattern_types) == 1:
                pattern_type = pattern_types[0]
                tmp_mat_binary = tmp_mat > 0
                match_res = calc_multi_conv(tmp_mat_binary, pattern_type, np.array(mat_pattern[:, :4]).astype(int))
                scores = match_res['scores']
                matched_pos = match_res['matched_pos']
                scores = calc_multi_cost_continuous(tmp_mat, pattern_type, matched_pos, scores, MAT_SIZE)
                match_res['scores'] = scores
                match_res['score'], match_res['scores'] = calc_disorder_score(tmp_mat, match_res, MAT_SIZE)
            else:
                raise NotImplementedError
            prefix = f'{mat_name}_swap_{swap_num}_score_{str(round(match_res["score"], 2))}'
            img_path = osp.join(swap_dir, f'{prefix}.png')
            swap_dic[prefix] = {
                'swapped_noise_mat': tmp_mat.astype(np.float16),
                'noise_mat': noise_mat.astype(np.float16),
                'continuous_mat': continuous_mat.astype(np.float16),
                'ori_mat': ori_mat.astype(np.float16),
                'mat_pattern': mat_pattern,
                'perm': tmp_perm,
                'img_path': img_path
            }

            for k, v in match_res.items():
                swap_dic[prefix][k] = v
            matrices.append(tmp_mat.astype(np.float16))
            if len(matrices) >= 20000:
                mat_num += len(matrices)
                filename = f'{osp.join(swap_dir, "matrices")}' + f'_{chunk_idx}.npz'
                if 'score' in list(swap_dic.values())[0]:
                    scores.append(np.array([i['score'] for i in swap_dic.values()]))
                np.savez_compressed(filename, matrices=matrices)  # also save swap_dic.npz
                np.savez_compressed(osp.join(swap_dir, f'swap_dic_{chunk_idx}.npz'), **swap_dic)
                print(f'Saved chunk {chunk_idx} to {filename}')
                chunk_idx += 1
                matrices = []
                swap_dic = {}

            if to_draw:
                img_write_new(img_path, tmp_mat, scale_factor=SCALE_FACTOR)
                img_write_new(img_path[:-4] + '_ori_mat.png', ori_mat, scale_factor=SCALE_FACTOR)
                img_write_new(img_path[:-4] + '_swapped_mat.png', tmp_mat, 'hybrid',
                              swap_dic[prefix]['matched_pos'], scale_factor=SCALE_FACTOR)
        mat_idx += 1
    mat_num += len(matrices)
    if len(matrices) > 0:
        filename = f'{osp.join(swap_dir, "matrices")}' + f'_{chunk_idx}.npz'
        np.savez_compressed(filename, matrices=matrices)
        np.savez_compressed(osp.join(swap_dir, f'swap_dic_{chunk_idx}.npz'), **swap_dic)
        print(f'Saved chunk {chunk_idx} to {filename}')
        chunk_idx += 1
    if 'score' in list(swap_dic.values())[0]:
        scores.append(np.array([i['score'] for i in swap_dic.values()]))
        np.save(osp.join(swap_dir, 'labels'), np.concatenate(scores).reshape(-1, 1))        
    print('num after index swap', mat_num)


if __name__ == "__main__":
    seed = args.seed
    train_dir = args.train_dir
    pattern_type = args.pattern_type  # 1000 block | 0100 off-diagonal block | 0010 star | 0001 band
    train_template_num = args.train_template_num
    with_test = args.with_test
    test_dir = None
    symmetric = args.symmetric
    continuous = args.continuous
    np.random.seed(seed)
    set_size = get_noise_cluster_set_size()
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    print('train_dir:', train_dir)
    print('pattern_type:', pattern_type)

    train_template_dir = osp.join(train_dir, 'Templates')
    os.makedirs(train_template_dir, exist_ok=True)
    if continuous:
        train_continuous_dir = osp.join(train_dir, 'Continuous')
        os.makedirs(train_continuous_dir, exist_ok=True)
    train_noise_dir = osp.join(train_dir, 'Noise')
    os.makedirs(train_noise_dir, exist_ok=True)
    train_swap_dir = osp.join(train_dir, 'IndexSwap')
    os.makedirs(train_swap_dir, exist_ok=True)
    if with_test:
        test_template_num = int(train_template_num / train_ratio)
        print('test_template_num', test_template_num)
        test_dir = train_dir[:train_dir.rfind('/') + 1] + 'test_' + train_dir[train_dir.rfind('/') + 1:]
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        test_template_dir = osp.join(test_dir, 'Templates')
        os.makedirs(test_template_dir, exist_ok=True)
        if continuous:
            test_continuous_dir = osp.join(test_dir, 'Continuous')
            os.makedirs(test_continuous_dir, exist_ok=True)
        test_noise_dir = osp.join(test_dir, 'Noise')
        os.makedirs(test_noise_dir, exist_ok=True)
        test_swap_dir = osp.join(test_dir, 'IndexSwap')
        os.makedirs(test_swap_dir, exist_ok=True)

    if not continuous:
        if not with_test:
            train_templates = gen_templates(pattern_type, train_template_num, symmetric)
            train_template_list = [item for r in train_templates.values() for item in r]
            gen_templates_dataset(train_template_list, train_template_dir)
            gen_noise_dataset(train_template_dir, train_noise_dir, set_size)
            gen_index_swap_dataset(train_noise_dir, train_swap_dir)
        else:
            all_templates = gen_templates(pattern_type, train_template_num + test_template_num, symmetric)
            train_templates = {key: values[:len(values) * train_ratio // (train_ratio + 1)] for key, values in
                                all_templates.items()}
            test_templates = {key: values[len(values) * train_ratio // (train_ratio + 1):] for key, values in
                               all_templates.items()}
            train_template_list = [item for r in train_templates.values() for item in r]
            test_template_list = [item for r in test_templates.values() for item in r]

            gen_templates_dataset(test_template_list, test_template_dir)
            gen_noise_dataset(test_template_dir, test_noise_dir, set_size)
            gen_index_swap_dataset(test_noise_dir, test_swap_dir)

            gen_templates_dataset(train_template_list, train_template_dir)
            gen_noise_dataset(train_template_dir, train_noise_dir, set_size)
            gen_index_swap_dataset(train_noise_dir, train_swap_dir)
    else:
        if not with_test:
            train_templates = gen_templates(pattern_type, train_template_num, symmetric)
            train_template_list = [item for r in train_templates.values() for item in r]
            gen_templates_dataset(train_template_list, train_template_dir)
            gen_continuous_dataset(train_template_dir, train_continuous_dir)
            gen_noise_dataset_continuous(train_continuous_dir, train_noise_dir, set_size)
            gen_index_swap_dataset_continuous(train_noise_dir, train_swap_dir)
        else:
            all_templates = gen_templates(pattern_type, train_template_num + test_template_num, symmetric)
            train_templates = {key: values[:len(values) * train_ratio // (train_ratio + 1)] for key, values in
                                all_templates.items()}
            test_templates = {key: values[len(values) * train_ratio // (train_ratio + 1):] for key, values in
                               all_templates.items()}
            train_template_list = [item for r in train_templates.values() for item in r]
            test_template_list = [item for r in test_templates.values() for item in r]

            gen_templates_dataset(test_template_list, test_template_dir)
            gen_continuous_dataset(test_template_dir, test_continuous_dir)
            gen_noise_dataset_continuous(test_continuous_dir, test_noise_dir, set_size)
            gen_index_swap_dataset_continuous(test_noise_dir, test_swap_dir)

            gen_templates_dataset(train_template_list, train_template_dir)
            gen_continuous_dataset(train_template_dir, train_continuous_dir)
            gen_noise_dataset_continuous(train_continuous_dir, train_noise_dir, set_size)
            gen_index_swap_dataset_continuous(train_noise_dir, train_swap_dir)
