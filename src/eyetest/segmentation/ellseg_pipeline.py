from __future__ import annotations

import cv2
import numpy as np
import torch

from eyetest.config import SegmentationConfig
from eyetest.segmentation.ellseg_compat_loss import get_seg2ptLoss
from eyetest.segmentation.ellseg_compat_utils import get_predictions
from eyetest.segmentation.ellseg_helpers import ElliFit, getValidPoints, my_ellipse, ransac
from eyetest.segmentation.ellseg_model import EllSegRuntime, load_ellseg_runtime
from eyetest.segmentation.ellseg_preprocess import preprocess_frame, rescale_to_original


class EllSegSegmenter:
    def __init__(self, segmentation: SegmentationConfig, eval_on_cpu: bool = False) -> None:
        self.config = segmentation
        self.runtime: EllSegRuntime = load_ellseg_runtime(segmentation, eval_on_cpu=eval_on_cpu)

    def segment(self, frame: np.ndarray) -> dict[str, object]:
        if frame.ndim == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        frame_scaled_shifted, scale_shift = preprocess_frame(
            frame_gray,
            (self.config.input_height, self.config.input_width),
            self.config.align_width,
        )
        input_tensor = frame_scaled_shifted.unsqueeze(0).to(self.runtime.device)
        seg_map, latent, pupil_ellipse, iris_ellipse = self._evaluate_on_image(input_tensor)
        seg_map, pupil_ellipse, iris_ellipse = rescale_to_original(
            seg_map,
            pupil_ellipse,
            iris_ellipse,
            scale_shift,
            frame_gray.shape,
        )
        return {
            "seg_map": seg_map,
            "latent": latent,
            "pupil_ellipse": pupil_ellipse,
            "iris_ellipse": iris_ellipse,
        }

    def _evaluate_on_image(self, frame: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(frame.shape) == 4, "Frame must be [1, 1, H, W]"
        with torch.no_grad():
            x4, x3, x2, x1, x = self.runtime.model.enc(frame)
            latent = torch.mean(x.flatten(start_dim=2), -1)
            seg_out = self.runtime.model.dec(x4, x3, x2, x1, x)

        seg_out = seg_out.cpu()
        latent = latent.squeeze().cpu().numpy()
        seg_map = get_predictions(seg_out).squeeze().numpy()
        if not self.config.segment_iris:
            seg_map[seg_map == 1] = 0
        if not self.config.segment_pupil:
            seg_map[seg_map == 2] = 0

        use_net_ellipses = self.config.use_regressed_ellipses
        el_out = None
        if use_net_ellipses:
            feat_h, feat_w = x.shape[-2:]
            reg_h = (feat_h - 1) // 2 - 4
            reg_w = (feat_w - 2) // 2 - 4
            if not ((reg_h == 3) and (reg_w == 5)):
                use_net_ellipses = False
            else:
                el_out = self.runtime.model.elReg(x, 0).squeeze().cpu()

        if use_net_ellipses:
            zeros = torch.zeros(2)
            if self.config.segment_pupil:
                _, norm_pupil_center = get_seg2ptLoss(seg_out[:, 2, ...], zeros, temperature=4)
                norm_pupil_ellipse = torch.cat([norm_pupil_center, el_out[7:10]])
                pupil_ellipse = my_ellipse(norm_pupil_ellipse.detach().cpu().numpy()).transform(
                    np.array([[frame.shape[-1] / 2, 0, frame.shape[-1] / 2], [0, frame.shape[-2] / 2, frame.shape[-2] / 2], [0, 0, 1]])
                )[0][:-1]
            else:
                pupil_ellipse = np.array([-1, -1, -1, -1, -1])

            if self.config.segment_iris:
                _, norm_iris_center = get_seg2ptLoss(-seg_out[:, 0, ...], zeros, temperature=4)
                norm_iris_ellipse = torch.cat([norm_iris_center, el_out[2:5]])
                iris_ellipse = my_ellipse(norm_iris_ellipse.detach().cpu().numpy()).transform(
                    np.array([[frame.shape[-1] / 2, 0, frame.shape[-1] / 2], [0, frame.shape[-2] / 2, frame.shape[-2] / 2], [0, 0, 1]])
                )[0][:-1]
            else:
                iris_ellipse = np.array([-1, -1, -1, -1, -1])
            return seg_map, latent, pupil_ellipse, iris_ellipse

        seg_map_temp = seg_map.copy()
        seg_map_temp[seg_map_temp == 2] += 1
        seg_map_temp[seg_map_temp == 1] += 1
        pupil_points, iris_points = getValidPoints(seg_map_temp, isPartSeg=False)

        if self.config.segment_pupil and np.sum(seg_map_temp == 3) > 50 and type(pupil_points) is not list:
            model_pupil = (
                ElliFit(**{"data": pupil_points})
                if self.config.skip_ransac
                else ransac(pupil_points, ElliFit, 15, 40, 5e-3, 15).loop()
            )
            pupil_ellipse = np.array(model_pupil.model)
        else:
            pupil_ellipse = np.array([-1, -1, -1, -1, -1])

        if self.config.segment_iris and np.sum(seg_map_temp == 2) > 50 and type(iris_points) is not list:
            model_iris = (
                ElliFit(**{"data": iris_points})
                if self.config.skip_ransac
                else ransac(iris_points, ElliFit, 15, 40, 5e-3, 15).loop()
            )
            iris_ellipse = np.array(model_iris.model)
        else:
            iris_ellipse = np.array([-1, -1, -1, -1, -1])
        return seg_map, latent, pupil_ellipse, iris_ellipse
