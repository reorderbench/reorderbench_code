import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from data import TestContinuousMatrixDataset, TestMatrixDataset
from model import DAO_ResNet_Matmul_Sinkhorn, PairwiseDistanceLayer
from utils import (calc_disorder_score, calc_multi_conv,
                   calc_multi_conv_continuous, img_write_new)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str,
                    default='./output/',
                    help="output dir")
parser.add_argument("--model_path", type=str, default='./models/block_binary_100.pth',
                    help="model dir")
parser.add_argument("--data_path", type=str,
                    default='./dataset/test_block_binary_400/IndexSwap/swap_dic.npz',
                    help="data dir")
parser.add_argument("--scorer_path", type=str, default='./unified_scoring_model/convnext.pth',
                    help="scorer_path")
parser.add_argument("--tta_iter", type=int, default=20)
parser.add_argument("--model_dim", type=int, default=256)
parser.add_argument("--mat_size", type=int, default=400, choices=[100, 200, 300, 400])
parser.add_argument("--pattern_type", type=str, choices=[None, "block", "offblock", "star", "band"], default="block")
parser.add_argument("--continuous_eval", action="store_true", default=False)
parser.add_argument("--tta", type=int, default=1, choices=[1, 5, 17], help="tta noisel level number")
scorer_dim_map = {"block": 0, "offblock": 1, "star": 2, "band": 3}

# add use_scheduler
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
print(args)
json.dump(vars(args), open(os.path.join(args.output_dir, f"args.json"), "w"))

np.random.seed(101)
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = DAO_ResNet_Matmul_Sinkhorn(args.model_dim)
checkpoint = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(checkpoint)


scorer = torchvision.models.convnext_tiny(weights=None)
first_conv_layer = scorer.features[0][0]
scorer.features[0][0] = nn.Sequential(
    PairwiseDistanceLayer(),
    nn.Conv2d(2, 3, kernel_size=1, padding=0),
    first_conv_layer
)
fc_layer = nn.Linear(768, 1 if args.pattern_type is None else 4)
sigmoid = nn.Sigmoid()
scorer.classifier[2] = nn.Sequential(fc_layer, sigmoid)

checkpoint = torch.load(args.scorer_path, map_location='cpu')
scorer.load_state_dict(checkpoint)

if not args.continuous_eval:
    val_dataset = TestMatrixDataset(args.data_path)
else:
    val_dataset = TestContinuousMatrixDataset(args.data_path)


model.to(device)
model.eval()

scorer.to(device)
scorer.eval()

output_img_dir = os.path.join(args.output_dir, 'images')
os.makedirs(output_img_dir, exist_ok=True)
scores = []
time_1 = 0
time_tta = 0
def get_score(mat, patterns):
    if not args.continuous_eval:
        match_res = calc_multi_conv(mat, patterns, args.mat_size)
        match_res['score'], match_res['scores'] = calc_disorder_score(mat, match_res, args.mat_size)
    else:
        match_res = calc_multi_conv_continuous(mat, patterns, args.mat_size)
    return match_res['score']


if os.path.exists(os.path.join(args.output_dir, f"scores.pth")):
    scores = torch.load(os.path.join(args.output_dir, f"scores.pth"))
# get size of score
scored_num = len(scores)
idx = 0
noise_level = [0]
if args.tta == 5:
    noise_level = [0, 0.04, 0.08, 0.12, 0.16]
elif args.tta == 17:
    noise_level = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
                   0.16]

with torch.no_grad():
    for idx, data_tuple in tqdm(enumerate(val_dataset)):
        if idx < scored_num:
            continue
        X = data_tuple[0]
        y = data_tuple[1]
        score_upb = data_tuple[2]
        patterns = data_tuple[3]
        if len(data_tuple) == 5:
            raise NotImplementedError
        noise_masks = [torch.bernoulli(torch.ones_like(X) * noise_level[i]).bool() for i in range(len(noise_level))]
        for i in range(len(noise_level)):
            noise_masks[i][torch.triu_indices(noise_masks[i].shape[0], 1)] = noise_masks[i].permute(0, 2, 1)[
                torch.triu_indices(noise_masks[i].shape[0], 1)]
        X = X.to(device)
        masked_X = [X.clone() for _ in range(len(noise_level))]
        mat_size = masked_X[i].shape[-1]
        for i in range(len(noise_level)):
            upper_triangular_mask = torch.triu(noise_masks[i], diagonal=1)
            upper_triangular = noise_masks[i] & upper_triangular_mask
            lower_trianguler_mask = torch.tril(torch.ones((1, mat_size, mat_size), dtype=torch.bool), diagonal=-1)
            noise_masks[i][lower_trianguler_mask] = upper_triangular.permute(0, 2, 1)[lower_trianguler_mask]
            masked_X[i][noise_masks[i]] = 1 - masked_X[i][noise_masks[i]]
        masked_X = torch.cat(masked_X, dim=0)[:, None, :, :]
        masked_X = masked_X.to(device)
        start_time = time.time()
        pred_noise, perm_y, perm_p = model.infer(masked_X)
        time_1 += time.time() - start_time
        tta_perm_p = [None, perm_p]
        tta_pred_noise = [masked_X, pred_noise]
        tta_pred = [X.repeat(len(noise_level), 1, 1)]
        pred = torch.matmul(torch.matmul(perm_p, tta_pred[-1].squeeze(1)), perm_p.permute(0, 2, 1))
        tta_pred.append(pred)
        for _ in range(args.tta_iter - 1):
            pred_noise_, perm_y_, perm_p_ = model.infer(tta_pred_noise[-1][:, None, :, :])
            tta_perm_p.append(perm_p_)
            tta_pred_noise.append(pred_noise_)
            pred_ = torch.matmul(torch.matmul(perm_p_, tta_pred[-1].squeeze(1)), perm_p_.permute(0, 2, 1))
            tta_pred.append(pred_)

        tta_pred = torch.stack(tta_pred, dim=0)
        tta_pred = tta_pred.view(-1, tta_pred.shape[-2], tta_pred.shape[-1])
        tta_pred_resize_200 = np.array(
            [cv2.resize(_.squeeze(0).cpu().numpy(), (200, 200), interpolation=cv2.INTER_AREA) for _ in tta_pred])
        tta_pred_resize_200 = torch.tensor(tta_pred_resize_200).to(device)
        tta_scores = [scorer(_[None, None, :, :])[
                            0, 0 if args.pattern_type is None else scorer_dim_map[args.pattern_type]].item() for _
                        in tta_pred_resize_200]
        time_tta += time.time() - start_time
        pred_best_idx = np.argmax(tta_scores[len(noise_level):2*len(noise_level)])
        tta_best_idx = np.argmax(tta_scores)
        pred_best_mat = tta_pred[pred_best_idx]
        tta_best_mat = tta_pred[tta_best_idx]
        input_score = get_score(X[0].squeeze(0).cpu().numpy(), patterns)
        pred_score = get_score(pred_best_mat.cpu().numpy(), patterns)
        tta_best_score = get_score(tta_best_mat.cpu().numpy(), patterns)
        scores.append({'input_score': input_score, 'pred_score': pred_score, 'tta_pred_score': tta_best_score,
                        'upb_score': score_upb})
        if idx % 100 == 0 and len(scores) > 0:
            print(f"Mean score: {np.mean([np.minimum(_['tta_pred_score'] / _['upb_score'], 1) for _ in scores])}")
            print(f"Mean inference time: {time_tta / len(scores)}")
            torch.save(scores, os.path.join(args.output_dir, f"scores.pth"))
        if idx % 53 == 0:
            img_write_new(os.path.join(output_img_dir, f'{idx}_input.png'),
                            X[0].detach().cpu().numpy())
            img_write_new(os.path.join(output_img_dir, f'{idx}_gt.png'),
                            y.detach().cpu().numpy())
            img_write_new(os.path.join(output_img_dir,
                                        f'{idx}_pred_{(pred_score / score_upb):.6f}.png'),
                            pred_best_mat.detach().cpu().numpy())
            img_write_new(os.path.join(output_img_dir,
                                        f'{idx}_tta_pred_{(tta_best_score / score_upb):.6f}.png'),
                            tta_best_mat.detach().cpu().numpy())


if len(scores) != 0:
    torch.save(scores, os.path.join(args.output_dir, f"scores.pth"))
    fp = open(os.path.join(args.output_dir, f"results.txt"), "w")
    fp.write(f"Mean score: {np.mean([np.minimum(_['tta_pred_score'] / _['upb_score'], 1) for _ in scores])}\n")
    fp.write(f"Mean inference time: {time_tta / len(val_dataset)}\n")
    fp.close()
