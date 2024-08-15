import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils import resize_to_200


parser = argparse.ArgumentParser()
parser.add_argument(
    "--matrix_path", type=str, default="./examples/bio-SC-TS_200.npy", help="matrix path"
)
parser.add_argument(
    "--model_path", type=str, default="./convnext.pth", help="path to ConvNeXt-T model"
)
args = parser.parse_args()
print(args)
device = "cuda" if torch.cuda.is_available() else "cpu"


class PairwiseDistanceLayer(nn.Module):
    def __init__(self):
        super(PairwiseDistanceLayer, self).__init__()

    def forward(self, x):
        return torch.cat((x, torch.cdist(x, x)), dim=1)


if __name__ == "__main__":
    net = torchvision.models.convnext_tiny()
    first_conv_layer = net.features[0][0]
    net.features[0][0] = nn.Sequential(
        PairwiseDistanceLayer(),
        nn.Conv2d(2, 3, kernel_size=1, padding=0),
        first_conv_layer,
    )
    fc_layer = nn.Linear(768, 4)
    sigmoid = nn.Sigmoid()
    net.classifier[2] = nn.Sequential(fc_layer, sigmoid)

    model = net
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    matrix = resize_to_200(np.load(args.matrix_path).astype(np.float32))
    with torch.no_grad():
        output = model(
            torch.tensor(matrix).unsqueeze(0).unsqueeze(0).float().to(device)
        )
        output = output.cpu().numpy()[0]
        print(output)
