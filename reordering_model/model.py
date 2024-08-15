from typing import Callable, Optional, Type, Union

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

# from torchvision.models.vision_transformer import Encoder


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Sinkhorn(torch.nn.Module):
    # Modify ResNet18 to be the backbone for matrix input/output
    def __init__(self, sinkhorn_iter=20):
        super(ResNet_Sinkhorn, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.sinkhorn_iter = sinkhorn_iter

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer4 = self._make_layer(BasicBlock, 64, 2, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def sinkhorn(self, x):
        results = [x]
        for _ in range(self.sinkhorn_iter):
            results.append(results[-1] - torch.logsumexp(results[-1], dim=1, keepdim=True))
            results.append(results[-1] - torch.logsumexp(results[-1], dim=2, keepdim=True))
        return torch.exp(results[-1])

    def solve_assign(self, x):
        nrow = x.shape[1]

        results = []
        for mat in x:
            row_id, col_id = linear_sum_assignment(mat.detach().cpu().numpy(), maximize=True)
            # print(f"Assignment total: {mat[row_id, col_id].sum()}")
            mat = torch.nn.functional.one_hot(torch.tensor(col_id), num_classes=nrow)
            results.append(mat.unsqueeze(0))

        results = torch.cat(results, dim=0).to(x.device)
        return results.to(torch.float)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = torch.permute(y, (0, 2, 3, 1))
        y = self.fc(y)
        y = torch.squeeze(y, -1)

        y = self.sinkhorn(y)
        out = torch.matmul(torch.matmul(y, x.squeeze(1)), y.permute(0, 2, 1))

        return out

    def infer(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = torch.permute(y, (0, 2, 3, 1))
        y = self.fc(y)
        y = torch.squeeze(y, -1)

        y = self.sinkhorn(y)
        y = self.solve_assign(y)
        x = torch.matmul(torch.matmul(y, x.squeeze(1)), y.permute(0, 2, 1))

        return x


class DAO_ResNet_Matmul_Sinkhorn(torch.nn.Module):
    def __init__(self, model_dim=64, sinkhorn_iter=20):
        super(DAO_ResNet_Matmul_Sinkhorn, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.sinkhorn_iter = sinkhorn_iter
        self.model_dim = model_dim

        self.inplanes = model_dim
        self.dilation = 1
        self.groups = 1
        self.base_width = model_dim

        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, model_dim, 2)
        self.layer2 = self._make_layer(BasicBlock, model_dim, 2, stride=1)
        self.layer3 = self._make_layer(BasicBlock, model_dim, 2, stride=1)
        self.layer4 = self._make_layer(BasicBlock, model_dim, 2, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(model_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def sinkhorn(self, x):
        results = [x]
        for _ in range(self.sinkhorn_iter):
            results.append(results[-1] - torch.logsumexp(results[-1], dim=1, keepdim=True))
            results.append(results[-1] - torch.logsumexp(results[-1], dim=2, keepdim=True))
        return torch.exp(results[-1])

    def solve_assign(self, x):
        nrow = x.shape[1]

        results = []
        for mat in x:
            row_id, col_id = linear_sum_assignment(mat.detach().cpu().numpy(), maximize=True)
            # print(f"Assignment total: {mat[row_id, col_id].sum()}")
            mat = torch.nn.functional.one_hot(torch.tensor(col_id), num_classes=nrow)
            results.append(mat.unsqueeze(0))

        results = torch.cat(results, dim=0).to(x.device)
        return results.to(torch.float)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.cat((x, torch.cdist(x, x)), dim=1)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.layer1(y)
        y = self.layer2(y)

        y = torch.matmul(y, y.permute(0, 1, 3, 2))

        y = self.layer3(y)
        y = self.layer4(y)

        y = torch.permute(y, (0, 2, 3, 1))
        y = self.fc(y)
        y = torch.squeeze(y, -1)

        y = self.sinkhorn(y)
        out = torch.matmul(torch.matmul(y, x.squeeze(1)), y.permute(0, 2, 1))

        return out, y

    def infer(self, x, noise_mask=None):
        x_ = x.clone()
        if noise_mask is not None:
            x_[noise_mask] = 1 - x_[noise_mask]
        y = torch.cat((x_, torch.cdist(x_, x_)), dim=1)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.layer1(y)
        y = self.layer2(y)

        y = torch.matmul(y, y.permute(0, 1, 3, 2))

        y = self.layer3(y)
        y = self.layer4(y)

        y = torch.permute(y, (0, 2, 3, 1))
        y = self.fc(y)
        y = torch.squeeze(y, -1)

        y = self.sinkhorn(y)
        p = self.solve_assign(y)
        x = torch.matmul(torch.matmul(p, x.squeeze(1)), p.permute(0, 2, 1))

        return x, y, p

class PairwiseDistanceLayer(nn.Module):
    def __init__(self):
        super(PairwiseDistanceLayer, self).__init__()

    def forward(self, x):
        return torch.cat((x, torch.cdist(x, x)), dim=1)
