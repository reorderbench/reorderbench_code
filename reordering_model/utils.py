import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from skimage import measure

neg_inf = -999999


def scale_image(image, scale_factor=1):
    return cv2.resize(
        image,
        (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)),
        interpolation=cv2.INTER_NEAREST,
    )

def img_write_new(
    img_path,
    mat,
    pattern_type=None,
    matched_pos=None,
    scale_factor=1,
):
    # mat is 0-1 matrix
    draw_mat = mat
    draw_mat = scale_image(draw_mat, scale_factor)
    # convert draw to 3 channels
    # rgb 112e53 17,46,83
    entries_color = (83, 46, 17)
    draw_mat = np.stack(
        [
            255 - (255 - entries_color[0]) * draw_mat,
            255 - (255 - entries_color[1]) * draw_mat,
            255 - (255 - entries_color[2]) * draw_mat,
        ],
        axis=-1,
    )

    line_color = (0, 0, 225)
    grid_color = (221, 221, 220)
    cv2.line(draw_mat, (0, 0), (draw_mat.shape[0], 0), grid_color, 1)
    cv2.line(draw_mat, (0, 0), (0, draw_mat.shape[0]), grid_color, 1)
    cv2.line(
        draw_mat,
        (draw_mat.shape[0] - 1, 0),
        (draw_mat.shape[0] - 1, draw_mat.shape[0] - 1),
        grid_color,
        1,
    )
    cv2.line(
        draw_mat,
        (0, draw_mat.shape[0] - 1),
        (draw_mat.shape[0] - 1, draw_mat.shape[0] - 1),
        grid_color,
        1,
    )
    cv2.line(draw_mat, (0, 0), (draw_mat.shape[0], draw_mat.shape[0]), grid_color, 1)

    if scale_factor >= 5:
        for k in range(0, draw_mat.shape[0], scale_factor):
            cv2.line(draw_mat, (0, k), (draw_mat.shape[0], k), grid_color, 1)
            cv2.line(draw_mat, (k, 0), (k, draw_mat.shape[0]), grid_color, 1)

    def draw_block(pos):
        if len(pos) == 3:
            x, y, l = pos
            cv2.rectangle(
                draw_mat,
                (y * scale_factor, x * scale_factor),
                ((y + l) * scale_factor, (x + l) * scale_factor),
                line_color,
                2,
            )
        else:
            x, y, h, w = pos
            cv2.line(
                draw_mat,
                (y * scale_factor, x * scale_factor),
                ((y + w) * scale_factor, (x) * scale_factor),
                line_color,
                2,
            )
            cv2.line(
                draw_mat,
                ((y + w) * scale_factor, (x) * scale_factor),
                ((y + w) * scale_factor, (x + h) * scale_factor),
                line_color,
                2,
            )
            cv2.line(
                draw_mat,
                ((y + w) * scale_factor, (x + h) * scale_factor),
                ((y) * scale_factor, (x + h) * scale_factor),
                line_color,
                2,
            )
            cv2.line(
                draw_mat,
                ((y) * scale_factor, (x + h) * scale_factor),
                (y * scale_factor, x * scale_factor),
                line_color,
                2,
            )

    def draw_band(pos):
        x, y, w, l = pos
        if y >= x:
            for iw in range(w):
                cv2.line(
                    draw_mat,
                    ((y + iw) * scale_factor, (x) * scale_factor),
                    ((y + l) * scale_factor, (x + l - iw) * scale_factor),
                    line_color,
                    2,
                )
        else:
            for iw in range(w):
                cv2.line(
                    draw_mat,
                    ((y) * scale_factor, (x + iw) * scale_factor),
                    ((y + l - iw) * scale_factor, (x + l) * scale_factor),
                    line_color,
                    2,
                )

    def draw_star(pos):
        x, y, h, w = pos

        cv2.line(
            draw_mat,
            (y * scale_factor, x * scale_factor),
            (y * scale_factor, (x + h) * scale_factor),
            line_color,
            2,
        )
        cv2.line(
            draw_mat,
            (y * scale_factor, (x + h) * scale_factor),
            ((y + w) * scale_factor, (x + h) * scale_factor),
            line_color,
            2,
        )
        cv2.line(
            draw_mat,
            ((y + w) * scale_factor, (x + h) * scale_factor),
            ((y + w) * scale_factor, x * scale_factor),
            line_color,
            2,
        )
        cv2.line(
            draw_mat,
            ((y + w) * scale_factor, x * scale_factor),
            (y * scale_factor, x * scale_factor),
            line_color,
            2,
        )

    def draw_pattern(pos, pattern_type):
        if pattern_type == "block" or pattern_type == "offblock":
            draw_block(np.array(list(pos))[:4].astype(int))
        elif pattern_type == "band":
            draw_band(np.array(list(pos))[:4].astype(int))
        elif pattern_type == "star":
            draw_star(np.array(list(pos))[:4].astype(int))

    if pattern_type is not None:
        if pattern_type == "hybrid":
            for pos in matched_pos:
                assert len(pos) == 5
                draw_pattern(np.array(pos[:4]).astype(int), pos[-1])
        else:
            for pos in matched_pos:
                draw_pattern(pos, pattern_type)

    cv2.imwrite(img_path, draw_mat)

def calc_multi_conv_block(mask1_, pattern, mat_size):
    # mask2 is conv kernel
    mask1 = mask1_.copy()
    mask1 = mask1.astype(int)
    sorted_pattern = sorted(pattern, key=lambda x: -x[2])
    indexes = np.arange(len(pattern))
    indexes = sorted(indexes, key=lambda x: -pattern[x][2])

    matched_pos = [None for _ in range(len(pattern))]
    scores = [None for _ in range(len(pattern))]
    L_scores = [None for _ in range(len(pattern))]

    for block in sorted_pattern:
        block_size = block[2]
        conv_result = np.zeros((mat_size - block_size + 1))
        for i in range(mat_size - block_size + 1):
            conv_result[i] = mask1[i : i + block_size, i : i + block_size].sum()
        max_conv_result = np.max(conv_result)

        pos = np.where(conv_result == max_conv_result)
        rows = pos[0]
        ridx = np.random.randint(0, len(rows))
        row, col = rows[ridx], rows[ridx]
        mask1[row : row + block_size, col : col + block_size] = neg_inf
        idx = indexes.pop(0)
        scores[idx] = (
            np.max(conv_result) / (block_size * block_size)
            if max_conv_result > 0
            else 0
        )
        matched_pos[idx] = [row, col, block_size, block_size]
    areas = [i[2] * i[3] for i in pattern]
    return {
        "scores": scores,
        "score": np.average(scores, weights=areas),
        "matched_pos": [i + ["block"] for i in matched_pos],
    }

def calc_multi_conv_offblock(mask1, ori_pattern, mat_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def zero_out_diagonal_elements(matrix, d):
        diag_indices = np.arange(max([matrix.shape[0], matrix.shape[1]]))
        mask = np.abs(diag_indices[:, None] - diag_indices) < d
        matrix *= 1 - mask[: matrix.shape[0], : matrix.shape[1]]
        return matrix

    m1 = torch.tensor(mask1.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to(device)

    upper_pattern = []
    for offblock in ori_pattern:
        if offblock[1] < offblock[0]:
            continue
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
            m1[0][0][
                max(0, row) : min(row + h, mat_size),
                max(0, col) : min(col + w, mat_size),
            ] = neg_inf
            m1[0][0][
                max(0, col) : min(col + w, mat_size),
                max(0, row) : min(row + h, mat_size),
            ] = neg_inf
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
            m1[0][0][
                max(0, row) : min(row + w, mat_size),
                max(0, col) : min(col + h, mat_size),
            ] = neg_inf
            m1[0][0][
                max(0, col) : min(col + h, mat_size),
                max(0, row) : min(row + w, mat_size),
            ] = neg_inf
    for mp in matched_pos:
        if mp[1] > mp[0] and mp[0] + mp[2] > mp[1]:
            raise ValueError("wrong", mp)
        if mp[0] > mp[1] and mp[1] + mp[3] > mp[0]:
            raise ValueError("wrong", mp)

    all_matched_pos = []
    for pos in matched_pos:
        all_matched_pos.append(list(pos))
        all_matched_pos.append([pos[1], pos[0], pos[3], pos[2]])
    areas = [i[2] * i[3] for i in upper_pattern]
    return {
        "scores": scores,
        "score": np.average(scores, weights=areas),
        "matched_pos": [i + ["offblock"] for i in all_matched_pos],
    }

def calc_multi_conv_star(mask1, ori_pattern, mat_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = torch.tensor(mask1.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to(device)

    row_pattern = []
    for star in ori_pattern:
        if star[1] > star[0] or star[1] == star[0] and star[2] > star[3]:
            continue
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
            y_vals = np.arange(
                max(0, x + h_star - w_star), min(x, mat_size - w_star) + 1
            )
            tmp_res = conv_result[x, y_vals]  # - 0.5 * conv_diag[x, x]
            max_index = np.argmax(tmp_res)
            if tmp_res[max_index] > best_conv_res:
                best_conv_res = tmp_res[max_index]
                best_pos = [x, y_vals[max_index]]

        scores[idx] = best_conv_res / (w_star * h_star) if best_conv_res > 0 else 0
        matched_pos[idx] = [
            [best_pos[0], best_pos[1], h_star, w_star],
            [best_pos[1], best_pos[0], w_star, h_star],
        ]
        areas[idx] = w_star * h_star
        # remove matched star
        m1[0][0][
            best_pos[0] : best_pos[0] + h_star, best_pos[1] : best_pos[1] + w_star
        ] = neg_inf
        m1[0][0][
            best_pos[1] : best_pos[1] + w_star, best_pos[0] : best_pos[0] + h_star
        ] = neg_inf

    all_matched_pos = []
    for pos in matched_pos:
        all_matched_pos.append(pos[0])
        all_matched_pos.append(pos[1])

    return {
        "scores": scores,
        "score": np.average(scores, weights=[i for i in areas]),
        "matched_pos": [i + ["star"] for i in all_matched_pos],
    }

def calc_multi_conv_band(mask1, ori_pattern, mat_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = torch.tensor(mask1.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to(device)

    upper_pattern = []
    for band in ori_pattern:
        if band[1] < band[0]:
            continue
        upper_pattern.append(band)

    pattern = sorted(upper_pattern, key=lambda x: (-x[3], -x[2]))
    indexes = np.arange(len(pattern))
    indexes = sorted(
        indexes, key=lambda x: (-upper_pattern[x][3], -upper_pattern[x][2])
    )

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
        "scores": scores,
        "score": np.average(scores, weights=[i for i in areas]),
        "matched_pos": [i + ["band"] for i in all_matched_pos],
    }

def calc_multi_conv(mat, patterns, mat_size):
    pattern_type = patterns[0, 4]
    patterns = np.array(patterns[:, :4]).astype(int)
    if pattern_type == "block":
        match_res = calc_multi_conv_block(mat, patterns, mat_size)
    elif pattern_type == "offblock":
        match_res = calc_multi_conv_offblock(mat, patterns, mat_size)
    elif pattern_type == "star":
        match_res = calc_multi_conv_star(mat, patterns, mat_size)
    elif pattern_type == "band":
        match_res = calc_multi_conv_band(mat, patterns, mat_size)
    else:
        raise NotImplementedError
    return match_res

def calc_disorder_score(mat, match_res, mat_size):
    scores = match_res["scores"]
    matched_pos = match_res["matched_pos"]
    swapped_noise_mat = mat
    penalty_scores = []
    entropys = []
    areas = []

    for pos_idx, pos in enumerate(matched_pos):
        assert len(pos) == 5
        pattern_type = pos[4]
        pos = pos[:4]
        if pattern_type == "band":
            x, y, w, l = pos
            if y < x:
                continue
            area = w * l - w * (w - 1) / 2
            matched_pattern = np.zeros(
                (min(x + l, mat_size) - max(0, x), min(y + l, mat_size) - max(0, y))
            )
            for i in range(matched_pattern.shape[0]):
                for j in range(i, min([i + w, matched_pattern.shape[1]])):
                    matched_pattern[i, j] = swapped_noise_mat[
                        max(0, x) + i, max(0, y) + j
                    ]
        elif pattern_type == "block":
            x, y, l, l = pos
            area = l**2
            matched_pattern = mat[x : x + l, y : y + l].copy()
        elif pattern_type == "offblock":
            x, y, h, w = pos
            if y < x:
                continue
            area = h * w
            matched_pattern = mat[
                max(0, x) : min(x + h, mat_size), max(0, y) : min(y + w, mat_size)
            ].copy()
        elif pattern_type == "star":
            hs, ws, h, w = pos
            if h >= w:
                continue
            he = hs + h
            we = ws + w
            area = h * w
            matched_pattern = np.zeros((w, w))
            matched_pattern[hs - ws : he - ws, ws - ws : we - ws] = swapped_noise_mat[
                hs:he, ws:we
            ]
            matched_pattern[ws - ws : we - ws, hs - ws : he - ws] = swapped_noise_mat[
                ws:we, hs:he
            ]
        areas.append(area)
        matched_pattern_binary = matched_pattern > 0.0
        total = np.sum(matched_pattern_binary)
        labels, num_features = measure.label(
            matched_pattern_binary, connectivity=2, return_num=True
        )
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

def calc_ar_deviations_cost(mat, pattern_type, pos, mat_size, offdiag=False):
    def process_offset(h, w, row_sub_mat, col_sub_mat, offset):
        cost = 0
        max_cost = 0
        for i in range(h):
            sub_mat = row_sub_mat[i]
            i_ = np.clip(i + offset, 0, w - 1)
            sub_mat_upper_left = sub_mat[: i_ + 1, : i_ + 1]
            sub_mat_lower_right = sub_mat[i_:, i_:]
            cost += np.sum(np.maximum(0, -sub_mat_upper_left))
            cost += np.sum(np.maximum(0, sub_mat_lower_right))
            max_cost += np.sum(np.abs(sub_mat_upper_left)) + np.sum(
                np.abs(sub_mat_lower_right)
            )
        for j in range(w):
            sub_mat = col_sub_mat[j]
            j_ = np.clip(j - offset, 0, h - 1)
            sub_mat_upper_left = sub_mat[: j_ + 1, : j_ + 1]
            sub_mat_lower_right = sub_mat[j_:, j_:]
            cost += np.sum(np.maximum(0, -sub_mat_upper_left))
            cost += np.sum(np.maximum(0, sub_mat_lower_right))
            max_cost += np.sum(np.abs(sub_mat_upper_left)) + np.sum(
                np.abs(sub_mat_lower_right)
            )
        if np.abs(max_cost) < 1e-6:
            return 1, 1
        return cost, max_cost

    def process_offset_delta(
        h,
        w,
        offset,
        prev_cost,
        prev_max_cost,
        row_sub_mat_neg_col_cumsum,
        row_sub_mat_neg_row_cumsum,
        row_sub_mat_pos_col_cumsum,
        row_sub_mat_pos_row_cumsum,
        row_sub_mat_abs_col_cumsum,
        row_sub_mat_abs_row_cumsum,
        col_sub_mat_neg_col_cumsum,
        col_sub_mat_neg_row_cumsum,
        col_sub_mat_pos_col_cumsum,
        col_sub_mat_pos_row_cumsum,
        col_sub_mat_abs_col_cumsum,
        col_sub_mat_abs_row_cumsum,
    ):
        cost = prev_cost
        max_cost = prev_max_cost
        i_values = np.arange(h)
        prev_i_ = np.clip(i_values + offset - 1, 0, w - 1)
        i_ = np.clip(i_values + offset, 0, w - 1)

        mask = prev_i_ != i_
        cost += np.sum(
            np.where(mask, row_sub_mat_neg_col_cumsum[i_values, i_, i_], 0), axis=-1
        )
        max_cost += np.sum(
            np.where(mask, row_sub_mat_abs_col_cumsum[i_values, i_, i_], 0), axis=-1
        )

        cost -= np.sum(
            np.where(
                mask,
                row_sub_mat_pos_row_cumsum[i_values, prev_i_, -1]
                - row_sub_mat_pos_row_cumsum[i_values, prev_i_, prev_i_],
                0,
            ),
            axis=-1,
        )
        max_cost -= np.sum(
            np.where(
                mask,
                row_sub_mat_abs_row_cumsum[i_values, prev_i_, -1]
                - row_sub_mat_abs_row_cumsum[i_values, prev_i_, prev_i_],
                0,
            ),
            axis=-1,
        )

        j_values = np.arange(w)
        prev_j_ = np.clip(j_values - offset + 1, 0, h - 1)
        j_ = np.clip(j_values - offset, 0, h - 1)

        mask = prev_j_ != j_
        cost -= np.sum(
            np.where(mask, col_sub_mat_neg_col_cumsum[j_values, prev_j_, prev_j_], 0),
            axis=-1,
        )
        max_cost -= np.sum(
            np.where(mask, col_sub_mat_abs_col_cumsum[j_values, prev_j_, prev_j_], 0),
            axis=-1,
        )

        cost += np.sum(
            np.where(
                mask,
                col_sub_mat_pos_row_cumsum[j_values, j_, -1]
                - col_sub_mat_pos_row_cumsum[j_values, j_, prev_j_],
                0,
            ),
            axis=-1,
        )
        max_cost += np.sum(
            np.where(
                mask,
                col_sub_mat_abs_row_cumsum[j_values, j_, -1]
                - col_sub_mat_abs_row_cumsum[j_values, j_, prev_j_],
                0,
            ),
            axis=-1,
        )

        if np.abs(max_cost) < 1e-6:
            return 1, 1
        return cost, max_cost

    hs, ws, h, w = pos[:4]
    pattern = mat[
        max(hs, 0) : min(hs + h, mat_size), max(ws, 0) : min(ws + w, mat_size)
    ]
    h, w = pattern.shape
    best_coe = 0
    row_sub_mat = []
    if offdiag:
        pattern = np.flip(pattern, axis=1)
    for i in range(h):
        row = pattern[i, :]
        zero_idx = np.where(row == 0)[0]
        sub_mat = np.triu(np.subtract.outer(row, row))
        sub_mat[zero_idx, :] = 0
        sub_mat[:, zero_idx] = 0
        row_sub_mat.append(sub_mat)
    row_sub_mat = np.array(row_sub_mat)
    col_sub_mat = []
    for j in range(w):
        col = pattern[:, j]
        zero_idx = np.where(col == 0)[0]
        sub_mat = np.triu(np.subtract.outer(col, col))
        sub_mat[zero_idx, :] = 0
        sub_mat[:, zero_idx] = 0
        col_sub_mat.append(sub_mat)
    col_sub_mat = np.array(col_sub_mat)
    if pattern_type == "block":
        offsets = [0]
    elif pattern_type == "star":
        offsets = [hs - ws]
    elif pattern_type == "offblock":
        offsets = [_ for _ in range(-h + 1, w)]
    if pattern_type == "offblock":
        row_sub_mat_neg_col_cumsum = np.cumsum(np.maximum(0, -row_sub_mat), axis=1)
        row_sub_mat_neg_row_cumsum = np.cumsum(np.maximum(0, -row_sub_mat), axis=2)
        row_sub_mat_pos_col_cumsum = np.cumsum(np.maximum(0, row_sub_mat), axis=1)
        row_sub_mat_pos_row_cumsum = np.cumsum(np.maximum(0, row_sub_mat), axis=2)
        row_sub_mat_abs_col_cumsum = np.cumsum(np.abs(row_sub_mat), axis=1)
        row_sub_mat_abs_row_cumsum = np.cumsum(np.abs(row_sub_mat), axis=2)
        col_sub_mat_neg_col_cumsum = np.cumsum(np.maximum(0, -col_sub_mat), axis=1)
        col_sub_mat_neg_row_cumsum = np.cumsum(np.maximum(0, -col_sub_mat), axis=2)
        col_sub_mat_pos_col_cumsum = np.cumsum(np.maximum(0, col_sub_mat), axis=1)
        col_sub_mat_pos_row_cumsum = np.cumsum(np.maximum(0, col_sub_mat), axis=2)
        col_sub_mat_abs_col_cumsum = np.cumsum(np.abs(col_sub_mat), axis=1)
        col_sub_mat_abs_row_cumsum = np.cumsum(np.abs(col_sub_mat), axis=2)
        for offset in offsets:
            if offset == -h + 1:
                cost, max_cost = process_offset(h, w, row_sub_mat, col_sub_mat, offset)
                if cost / max_cost > best_coe:
                    best_coe = cost / max_cost
            else:
                cost, max_cost = process_offset_delta(
                    h,
                    w,
                    offset,
                    cost,
                    max_cost,
                    row_sub_mat_neg_col_cumsum,
                    row_sub_mat_neg_row_cumsum,
                    row_sub_mat_pos_col_cumsum,
                    row_sub_mat_pos_row_cumsum,
                    row_sub_mat_abs_col_cumsum,
                    row_sub_mat_abs_row_cumsum,
                    col_sub_mat_neg_col_cumsum,
                    col_sub_mat_neg_row_cumsum,
                    col_sub_mat_pos_col_cumsum,
                    col_sub_mat_pos_row_cumsum,
                    col_sub_mat_abs_col_cumsum,
                    col_sub_mat_abs_row_cumsum,
                )
                if cost / max_cost > best_coe:
                    best_coe = cost / max_cost
    else:
        cost, max_cost = process_offset(h, w, row_sub_mat, col_sub_mat, offsets[0])
        if cost / max_cost > best_coe:
            best_coe = cost / max_cost
    return best_coe

def calc_multi_cost_continuous(
    mat, pattern_type, matched_pos, scores, mat_size
):
    scores_cost_coes = []
    if pattern_type == "band":
        scores_cost_coes = np.ones(len(scores))
    else:
        if pattern_type == "star" or pattern_type == "offblock":
            matched_pos = matched_pos[::2]
        i = 0
        for pos in matched_pos:
            i += 1
            scores_cost_coe = calc_ar_deviations_cost(mat, pattern_type, pos, mat_size)
            if pattern_type == "offblock":
                scores_cost_coe = max(
                    scores_cost_coe,
                    calc_ar_deviations_cost(
                        mat, pattern_type, pos, mat_size, offdiag=True
                    ),
                )
            scores_cost_coes.append(scores_cost_coe)
    scores_cost_coes = np.array(scores_cost_coes)
    scores = scores * scores_cost_coes
    return scores

def calc_multi_conv_continuous(mat, patterns, mat_size):
    pattern_type = patterns[0, 4]
    mat_binary = mat > 0
    match_res = calc_multi_conv(mat_binary, patterns, mat_size)
    scores = match_res["scores"]
    matched_pos = match_res["matched_pos"]
    scores = calc_multi_cost_continuous(
        mat, pattern_type, matched_pos, scores, mat_size
    )
    match_res["scores"] = scores
    match_res["score"], match_res["scores"] = calc_disorder_score(mat, match_res)
    return match_res
