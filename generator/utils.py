import cv2
import numpy as np

"""
    Functions to plot the matrix and save the image
"""


def scale_image(image, scale_factor=10):
    return cv2.resize(
        image,
        (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)),
        interpolation=cv2.INTER_NEAREST,
    )


def img_write_new(img_path, mat, pattern_type=None, matched_pos=None, scale_factor=3):
    draw_mat = mat.astype(np.float32)
    draw_mat = scale_image(draw_mat, scale_factor)
    # RGB#112e53
    entries_color = (83, 46, 17)
    draw_mat = np.stack([255 - (255 - entries_color[0]) * draw_mat,
                         255 - (255 - entries_color[1]) * draw_mat,
                         255 - (255 - entries_color[2]) * draw_mat,
                        ], axis=-1)
    line_color = (0, 0, 225)
    grid_color = (221, 221, 220)
    cv2.line(draw_mat, (0, 0), (draw_mat.shape[0], 0), grid_color, 1)
    cv2.line(draw_mat, (0, 0), (0, draw_mat.shape[0]), grid_color, 1)
    cv2.line(draw_mat, (draw_mat.shape[0] - 1, 0), (draw_mat.shape[0] - 1, draw_mat.shape[0] - 1),
        grid_color, 1)
    cv2.line(draw_mat, (0, draw_mat.shape[0] - 1), (draw_mat.shape[0] - 1, draw_mat.shape[0] - 1),
        grid_color, 1)

    if scale_factor >= 5:
        for k in range(0, draw_mat.shape[0], scale_factor):
            cv2.line(draw_mat, (0, k), (draw_mat.shape[0], k), grid_color, 1)
            cv2.line(draw_mat, (k, 0), (k, draw_mat.shape[0]), grid_color, 1)

    def draw_block(pos):
        if len(pos) == 3:
            x, y, l = pos
            cv2.rectangle(draw_mat, (y * scale_factor, x * scale_factor), ((y + l) * scale_factor, (x + l) * scale_factor),
                line_color, 2
            )
        else:
            x, y, h, w = pos
            cv2.line(draw_mat, (y * scale_factor, x * scale_factor), ((y + w) * scale_factor, (x) * scale_factor),
                line_color, 2,
            )
            cv2.line(draw_mat, ((y + w) * scale_factor, (x) * scale_factor), ((y + w) * scale_factor, (x + h) * scale_factor),
                line_color, 2,
            )
            cv2.line(draw_mat, ((y + w) * scale_factor, (x + h) * scale_factor), ((y) * scale_factor, (x + h) * scale_factor),
                line_color, 2,
            )
            cv2.line(draw_mat, ((y) * scale_factor, (x + h) * scale_factor), (y * scale_factor, x * scale_factor),
                line_color, 2,
            )

    def draw_band(pos):
        x, y, w, l = pos
        if y >= x:
            for iw in range(w):
                cv2.line(draw_mat, ((y + iw) * scale_factor, (x) * scale_factor), ((y + l) * scale_factor, (x + l - iw) * scale_factor),
                    line_color, 2,
                )
        else:
            for iw in range(w):
                cv2.line(draw_mat, ((y) * scale_factor, (x + iw) * scale_factor), ((y + l - iw) * scale_factor, (x + l) * scale_factor),
                    line_color, 2,
                )

    def draw_star(pos):
        x, y, h, w = pos
        cv2.line(draw_mat, (y * scale_factor, x * scale_factor), (y * scale_factor, (x + h) * scale_factor),
            line_color, 2,
        )
        cv2.line(draw_mat, (y * scale_factor, (x + h) * scale_factor), ((y + w) * scale_factor, (x + h) * scale_factor),
            line_color, 2,
        )
        cv2.line(draw_mat, ((y + w) * scale_factor, (x + h) * scale_factor), ((y + w) * scale_factor, x * scale_factor),
            line_color, 2,
        )
        cv2.line(draw_mat, ((y + w) * scale_factor, x * scale_factor), (y * scale_factor, x * scale_factor),
            line_color, 2,
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


def pattern_type2str(comb):
    if comb == "1000":
        return "block"
    elif comb == "0100":
        return "star"
    elif comb == "0010":
        return "offblock"
    elif comb == "0001":
        return "band"


"""
    Basic function for adding pattern to the matrix
"""


def add_pattern(mat, pattern, copy=False):
    def add_block(mat, block):
        # Block and Off-diagonal block: (x, y, h, w)
        assert len(block) == 4
        x, y, h, w = block
        mat[x : x + h, y : y + w] = 1.0

    def add_star(mat, star):
        # Star: (x, y, h, w), for one of the two lines
        x, y, h, w = star
        mat[x : x + h, y : y + w] = 1.0

    def add_band(mat, band):
        # Band: (x, y, width, length), consider upper triangle
        n = mat.shape[0]
        x, y, width, length = band
        if y >= x:
            for row in range(x, min([n, x + length])):
                for col in range(y + row - x, min([n, y + length, y + row - x + width])):
                    mat[row][col] = 1.0
        else:
            for col in range(y, min([n, y + length])):
                for row in range(x + col - y, min([n, x + length, x + col - y + width])):
                    mat[row][col] = 1.0

    if copy:
        mat = mat.copy()
    for pat in pattern:
        assert len(pat) == 5
        if pat[-1] == "block" or pat[-1] == "offblock":
            add_block(mat, np.array(pat[:4]).astype(int))
        elif pat[-1] == "star":
            add_star(mat, np.array(pat[:4]).astype(int))
        elif pat[-1] == "band":
            add_band(mat, np.array(pat[:4]).astype(int))
        else:
            print("wrong pattern")
            raise ValueError
    return mat


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
            max_cost += np.sum(np.abs(sub_mat_upper_left)) + np.sum(np.abs(sub_mat_lower_right))
        for j in range(w):
            sub_mat = col_sub_mat[j]
            j_ = np.clip(j - offset, 0, h - 1)
            sub_mat_upper_left = sub_mat[: j_ + 1, : j_ + 1]
            sub_mat_lower_right = sub_mat[j_:, j_:]
            cost += np.sum(np.maximum(0, -sub_mat_upper_left))
            cost += np.sum(np.maximum(0, sub_mat_lower_right))
            max_cost += np.sum(np.abs(sub_mat_upper_left)) + np.sum(np.abs(sub_mat_lower_right))
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
        cost += np.sum(np.where(mask, row_sub_mat_neg_col_cumsum[i_values, i_, i_], 0), axis=-1)
        max_cost += np.sum(np.where(mask, row_sub_mat_abs_col_cumsum[i_values, i_, i_], 0), axis=-1)

        cost -= np.sum(np.where(
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


def index_swap(ori_mat, swap_num, mat_size):
    mat = ori_mat.copy()
    n = mat.shape[0]
    t = 0
    perm = np.array(list(range(mat_size)))
    while t < swap_num:
        r1 = np.random.randint(0, n - 1)
        r2 = np.random.randint(r1 + 1, n - 1 + 1)
        perm[[r1, r2]] = perm[[r2, r1]]
        mat[[r1, r2]] = mat[[r2, r1]]
        mat[:, [r1, r2]] = mat[:, [r2, r1]]
        t += 1
    return mat, perm
