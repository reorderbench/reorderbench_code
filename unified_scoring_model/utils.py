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
    
