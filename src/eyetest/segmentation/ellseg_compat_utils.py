from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: Optional[bool] = True,
) -> torch.Tensor:
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    base_grid = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)


def get_predictions(output: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = output.size()
    _, indices = output.cpu().max(1)
    return indices.view(batch_size, height, width)


def normPts(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    points_out = copy.deepcopy(points)
    original_shape = points_out.shape
    points_out = points_out.reshape(-1, 2)
    points_out[:, 0] = 2 * (points_out[:, 0] / size[1]) - 1
    points_out[:, 1] = 2 * (points_out[:, 1] / size[0]) - 1
    return points_out.reshape(original_shape)


class linStack(torch.nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, bias, actBool, dp):
        super().__init__()
        layers = []
        for index in range(num_layers):
            layer = torch.nn.Linear(
                hidden_dim if index > 0 else in_dim,
                hidden_dim if index < num_layers - 1 else out_dim,
                bias=bias,
            )
            layers.append(layer)
        self.layersLin = torch.nn.ModuleList(layers)
        self.act_func = torch.nn.SELU()
        self.actBool = actBool
        self.dp = torch.nn.Dropout(p=dp)

    def forward(self, x):
        for layer in self.layersLin:
            x = self.act_func(x) if self.actBool else x
            x = layer(x)
            x = self.dp(x)
        return x


class regressionModule(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()
        in_channels = sizes["enc"]["op"][-1]
        self.max_pool = nn.AvgPool2d(kernel_size=2)
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=128, bias=True, kernel_size=(2, 3))
        self.c2 = nn.Conv2d(in_channels=128, out_channels=128, bias=True, kernel_size=3)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, bias=False)
        self.l1 = nn.Linear(32 * 3 * 5, 256, bias=True)
        self.l2 = nn.Linear(256, 10, bias=True)
        self.c_actfunc = torch.tanh
        self.param_actfunc = torch.sigmoid

    def forward(self, x, alpha):
        batch_size = x.shape[0]
        x = F.leaky_relu(self.c1(x))
        x = self.max_pool(x)
        x = F.leaky_relu(self.c2(x))
        x = F.leaky_relu(self.c3(x))
        x = x.reshape(batch_size, -1)
        x = self.l2(torch.selu(self.l1(x)))

        pupil_center = self.c_actfunc(x[:, 0:2])
        pupil_param = self.param_actfunc(x[:, 2:4])
        pupil_angle = x[:, 4]
        iris_center = self.c_actfunc(x[:, 5:7])
        iris_param = self.param_actfunc(x[:, 7:9])
        iris_angle = x[:, 9]
        return torch.cat(
            [
                pupil_center,
                pupil_param,
                pupil_angle.unsqueeze(1),
                iris_center,
                iris_param,
                iris_angle.unsqueeze(1),
            ],
            dim=1,
        )


class convBlock(nn.Module):
    def __init__(self, in_c, inter_c, out_c, actfunc):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inter_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = torch.nn.BatchNorm2d(num_features=out_c)

    def forward(self, x):
        x = self.actfunc(self.conv1(x))
        x = self.actfunc(self.conv2(x))
        return self.bn(x)
