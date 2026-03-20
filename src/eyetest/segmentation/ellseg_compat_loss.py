from __future__ import annotations

import torch
import torch.nn.functional as F

from eyetest.segmentation.ellseg_compat_utils import create_meshgrid


def get_seg2ptLoss(op, gtPts, temperature=1):
    batch_size, height, width = op.shape
    weight_map = F.softmax(op.view(batch_size, -1) * temperature, dim=1)
    xy_grid = create_meshgrid(height, width, normalized_coordinates=True)
    xloc = xy_grid[0, :, :, 0].reshape(-1).to(op.device)
    yloc = xy_grid[0, :, :, 1].reshape(-1).to(op.device)
    xpos = torch.sum(weight_map * xloc, -1, keepdim=True)
    ypos = torch.sum(weight_map * yloc, -1, keepdim=True)
    pred_pts = torch.stack([xpos, ypos], dim=1).squeeze()
    loss = F.l1_loss(pred_pts, gtPts.to(op.device), reduction="none")
    return loss, pred_pts


def conf_Loss(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Training-only loss is not supported in eyetest inference mode.")


def get_ptLoss(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Training-only loss is not supported in eyetest inference mode.")


def get_segLoss(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Training-only loss is not supported in eyetest inference mode.")


def get_seg2elLoss(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Training-only loss is not supported in eyetest inference mode.")


def get_selfConsistency(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Training-only loss is not supported in eyetest inference mode.")
