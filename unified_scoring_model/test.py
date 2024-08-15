import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import extract_scores, resize_to_200

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_folder", type=str, default="./dataset", help="data folder"
)
parser.add_argument(
    "--model_path", type=str, default="./vgg.pth", help="model type"
)
parser.add_argument(
    "--model_type",
    type=str,
    choices=["vgg16", "convnext", "res50"],
    default="vgg16",
    help="model type",
)
args = parser.parse_args()
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([transforms.ToTensor()])


class PairwiseDistanceLayer(nn.Module):
    def __init__(self):
        super(PairwiseDistanceLayer, self).__init__()

    def forward(self, x):
        return torch.cat((x, torch.cdist(x, x)), dim=1)


class TestDataset(Dataset):
    def __init__(self, indices, matrices, labels, info=None, transform=transform):
        self.matrices = matrices[indices]
        self.labels = labels[indices]
        if info is None:
            self.info = None
        else:
            self.info = info[indices]
        assert len(self.matrices) == len(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        matrix = self.matrices[index]
        label = self.labels[index]
        if self.transform:
            matrix = self.transform(matrix)
        return matrix, label


if __name__ == "__main__":
    if args.model_type == "vgg16":
        print("vgg16")
        net = torchvision.models.vgg16(weights=None)
        first_conv_layer = net.features[0]
        net.features[0] = nn.Sequential(
            PairwiseDistanceLayer(),
            nn.Conv2d(2, 3, kernel_size=1, padding=0),
            first_conv_layer,
        )
        fc_layer = nn.Linear(4096, 4)
        sigmoid = nn.Sigmoid()
        net.classifier[6] = nn.Sequential(fc_layer, sigmoid)
    elif args.model_type == "convnext":
        print("convnext")
        net = torchvision.models.convnext_tiny(weights=None)
        first_conv_layer = net.features[0][0]
        net.features[0][0] = nn.Sequential(
            PairwiseDistanceLayer(),
            nn.Conv2d(2, 3, kernel_size=1, padding=0),
            first_conv_layer,
        )
        fc_layer = nn.Linear(768, 4)
        sigmoid = nn.Sigmoid()
        net.classifier[2] = nn.Sequential(fc_layer, sigmoid)
    elif args.model_type == "res50":
        print("resnet50")
        net = torchvision.models.resnet50(weights=None)
        first_conv_layer = net.conv1
        net.conv1 = nn.Sequential(
            PairwiseDistanceLayer(),
            nn.Conv2d(2, 3, kernel_size=1, padding=0),
            first_conv_layer,
        )
        fc_layer = nn.Linear(2048, 4)
        sigmoid = nn.Sigmoid()
        net.fc = nn.Sequential(fc_layer, sigmoid)
    else:
        print("wrong model")
        raise NotImplementedError
    patterns = ["block", "offblock", "star", "band"]
    matrix_types = ["binary", "continuous"]
    matrix_sizes = [100, 200, 300, 400]
    model = net
    model_path = args.model_path
    if not osp.exists(model_path):
        print(f"{model_path} not exists")
        raise FileNotFoundError
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    for cont in matrix_types:
        for pat in patterns:
            for size in matrix_sizes:
                hybrid_dir = os.path.join(
                    args.data_folder, f"test_{pat}_{cont}_{size}", "IndexSwap"
                )
                i = 0
                tested_num = 0
                L1loss = 0
                L2loss = 0
                labels_all = np.load(osp.join(hybrid_dir, "labels.npy"))
                while os.path.exists(osp.join(hybrid_dir, f"matrices_{i}.npz")):
                    matrices = np.load(osp.join(hybrid_dir, f"matrices_{i}.npz"))["matrices"].astype(np.float16)
                    matrices = np.array([resize_to_200(matrix) for matrix in matrices])
                    labels = labels_all[tested_num:tested_num + len(matrices)]
                    tested_num += len(matrices)
                    test_dataset = TestDataset(np.array(range(len(matrices))), matrices, labels)
                    test_loader = DataLoader(
                        test_dataset, batch_size=256, shuffle=False, num_workers=1
                    )
                    dataloader = test_loader
                    L1criterion = nn.L1Loss(reduction="sum")
                    L2criterion = nn.MSELoss(reduction="sum")

                    with torch.no_grad():
                        for X, y in tqdm(dataloader):
                            X = X.float()
                            X, y = X.to(device), y.to(device)
                            y = y.reshape(-1, 1)
                            pred_comb = model(X)
                            pred = extract_scores(pred_comb, pat).reshape(-1, 1)
                            L1loss += L1criterion(pred, y)
                            L2loss += L2criterion(pred, y)
                    
                    i += 1
                L1loss /= len(labels_all)
                L2loss /= len(labels_all)
                print(f"L1 loss: {L1loss:>5f} \n")
                print(f"L2 loss: {L2loss:>5f} \n")
                    
