import numpy as np
import cv2


def extract_scores(pred_comb, pattern_type):
    if pattern_type == "block":
        return pred_comb[:, 0]
    elif pattern_type == "offblock":
        return pred_comb[:, 1]
    elif pattern_type == "star":
        return pred_comb[:, 2]
    elif pattern_type == "band":
        return pred_comb[:, 3]


def resize_to_200(matrix):
    matrix = matrix.astype(np.float32)
    return cv2.resize(matrix, (200, 200), interpolation=cv2.INTER_AREA)


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
