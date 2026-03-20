from __future__ import annotations

from itertools import chain

import cv2
import numpy as np
from skimage import draw


class my_ellipse:
    def __init__(self, param):
        self.eps = 1e-3
        self.param = param
        self.mat = self.param2mat(self.param)

    def param2mat(self, param):
        cx, cy, a, b, theta = tuple(param)
        h_rot = rotation_2d(-theta)
        h_trans = trans_2d(-cx, -cy)
        a_inv, b_inv = 1 / a**2, 1 / b**2
        q_matrix = np.array([[a_inv, 0, 0], [0, b_inv, 0], [0, 0, -1]])
        return h_trans.T @ h_rot.T @ q_matrix @ h_rot @ h_trans

    def mat2quad(self, mat):
        a, b, c, d, e, f = mat[0, 0], 2 * mat[0, 1], mat[1, 1], 2 * mat[0, 2], 2 * mat[1, 2], mat[-1, -1]
        return np.array([a, b, c, d, e, f])

    def recover_theta(self, mat):
        a, b, c, _, _, _ = tuple(self.mat2quad(mat))
        if abs(b) <= self.eps and a <= c:
            return 0.0
        if abs(b) <= self.eps and a > c:
            return np.pi / 2
        return 0.5 * np.arctan2(b, (a - c))

    def recover_center(self, mat):
        a, b, c, d, e, _ = tuple(self.mat2quad(mat))
        tx = (2 * c * d - b * e) / (b**2 - 4 * a * c)
        ty = (2 * a * e - b * d) / (b**2 - 4 * a * c)
        return tx, ty

    def mat2param(self, mat):
        theta = self.recover_theta(mat)
        tx, ty = self.recover_center(mat)
        h_rot = rotation_2d(theta)
        h_trans = trans_2d(tx, ty)
        mat_norm = h_rot.T @ h_trans.T @ mat @ h_trans @ h_rot
        major_axis = np.sqrt(1 / mat_norm[0, 0])
        minor_axis = np.sqrt(1 / mat_norm[1, 1])
        area = np.pi * major_axis * minor_axis
        return np.array([tx, ty, major_axis, minor_axis, theta, area])

    def transform(self, transform_matrix):
        mat_trans = np.linalg.inv(transform_matrix.T) @ self.mat @ np.linalg.inv(transform_matrix)
        return self.mat2param(mat_trans), self.mat2quad(mat_trans), mat_trans


class ElliFit:
    def __init__(self, **kwargs):
        self.data = np.array([])
        self.pts_lim = 12
        for key, value in kwargs.items():
            setattr(self, key, value)
        if np.size(self.data) > self.pts_lim:
            self.model = self.fit()
            self.error = np.mean(self.fit_error(self.data))
        else:
            self.model = [-1, -1, -1, -1, -1]
            self.error = np.inf

    def fit(self):
        xm = np.mean(self.data[:, 0])
        ym = np.mean(self.data[:, 1])
        x = self.data[:, 0] - xm
        y = self.data[:, 1] - ym
        x_stack = np.stack([x**2, 2 * x * y, -2 * x, -2 * y, -np.ones((np.size(x),))], axis=1)
        y_stack = -y**2
        try:
            phi = np.matmul(np.linalg.inv(np.matmul(x_stack.T, x_stack)), np.matmul(x_stack.T, y_stack))
            x0 = (phi[2] - phi[3] * phi[1]) / (phi[0] - phi[1] ** 2)
            y0 = (phi[0] * phi[3] - phi[2] * phi[1]) / (phi[0] - phi[1] ** 2)
            term2 = np.sqrt((1 - phi[0]) ** 2 + 4 * phi[1] ** 2)
            term3 = phi[4] + y0**2 + (x0**2) * phi[0] + 2 * phi[1]
            term1 = 1 + phi[0]
            b = np.sqrt(2 * term3 / (term1 + term2))
            a = np.sqrt(2 * term3 / (term1 - term2))
            alpha = 0.5 * np.arctan2(2 * phi[1], 1 - phi[0])
            model = [x0 + xm, y0 + ym, a, b, -alpha]
        except Exception:
            model = [np.nan, np.nan, np.nan, np.nan, np.nan]
        if np.all(np.isreal(model)) and np.all(~np.isnan(model)) and np.all(~np.isinf(model)):
            return model
        return [-1, -1, -1, -1, -1]

    def fit_error(self, data):
        term1 = (data[:, 0] - self.model[0]) * np.cos(self.model[-1])
        term2 = (data[:, 1] - self.model[1]) * np.sin(self.model[-1])
        term3 = (data[:, 0] - self.model[0]) * np.sin(self.model[-1])
        term4 = (data[:, 1] - self.model[1]) * np.cos(self.model[-1])
        res = (1 / self.model[2] ** 2) * (term1 - term2) ** 2 + (1 / self.model[3] ** 2) * (term3 + term4) ** 2 - 1
        return np.abs(res)


class ransac:
    def __init__(self, data, model, n_min, mxIter, thres, n_good):
        self.data = data
        self.num_pts = data.shape[0]
        self.model = model
        self.n_min = n_min
        self.d = n_good if n_min < n_good else n_min
        self.k = mxIter
        self.t = thres
        self.bestModel = self.model(**{"data": data})

    def loop(self):
        if self.num_pts <= self.n_min:
            return self.bestModel
        for _ in range(self.k + 1):
            inlier_indices = np.random.choice(self.num_pts, self.n_min, replace=False)
            location_inlier = np.isin(np.arange(0, self.num_pts), inlier_indices)
            outlier_indices = np.where(~location_inlier)[0]
            pot_model = self.model(**{"data": self.data[location_inlier, :]})
            list_error = pot_model.fit_error(self.data[~location_inlier, :])
            inlier_num = np.size(inlier_indices) + np.sum(list_error < self.t)
            if inlier_num > self.d:
                pot_inliers = np.concatenate([inlier_indices, outlier_indices[list_error < self.t]], axis=0)
                location_pot_inliers = np.isin(np.arange(0, self.num_pts), pot_inliers)
                better_model = self.model(**{"data": self.data[location_pot_inliers, :]})
                if better_model.error < self.bestModel.error:
                    self.bestModel = better_model
        return self.bestModel


def rotation_2d(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array([[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1]])


def trans_2d(cx, cy):
    return np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1]])


def getValidPoints(label_matrix, isPartSeg=True):
    if label_matrix.max() <= 0:
        return [], []
    image = np.uint8(255 * label_matrix.astype(np.float32) / label_matrix.max())
    edges = cv2.Canny(image, 50, 100) + cv2.Canny(255 - image, 50, 100)
    rows, cols = np.where(edges)
    pupil_points = []
    iris_points = []
    for location in zip(cols, rows):
        temp = label_matrix[location[1] - 1 : location[1] + 2, location[0] - 1 : location[0] + 2]
        cond_pupil = np.any(temp == 0) or np.any(temp == 1) or temp.size == 0
        cond_iris = np.any(temp == 0) or np.any(temp == 3) or temp.size == 0 if isPartSeg else np.any(temp == 3) or temp.size == 0
        if not cond_pupil:
            pupil_points.append(np.array(location))
        if not cond_iris:
            iris_points.append(np.array(location))
    pupil_points = np.stack(pupil_points, axis=0) if pupil_points else []
    iris_points = np.stack(iris_points, axis=0) if iris_points else []
    return pupil_points, iris_points


def plot_segmap_ellpreds(image, seg_map, pupil_ellipse, iris_ellipse):
    loc_iris = seg_map == 1
    loc_pupil = seg_map == 2
    out_image = np.stack([image] * 3, axis=2)
    loc_image_non_sat = image < (255 - 100)
    out_image[..., 1] = out_image[..., 1] + 100 * loc_iris * loc_image_non_sat
    out_image[..., 0] = out_image[..., 0] + 100 * loc_pupil * loc_image_non_sat
    out_image[..., 1] = out_image[..., 1] + 100 * loc_pupil * loc_image_non_sat

    if not np.all(iris_ellipse == -1):
        rr_i, cc_i = draw.ellipse_perimeter(
            int(iris_ellipse[1]),
            int(iris_ellipse[0]),
            int(iris_ellipse[3]),
            int(iris_ellipse[2]),
            orientation=iris_ellipse[4],
        )
        rr_i = rr_i.clip(6, image.shape[0] - 6)
        cc_i = cc_i.clip(6, image.shape[1] - 6)
        out_image[rr_i, cc_i, ...] = np.array([0, 0, 255])

    if not np.all(pupil_ellipse == -1):
        rr_p, cc_p = draw.ellipse_perimeter(
            int(pupil_ellipse[1]),
            int(pupil_ellipse[0]),
            int(pupil_ellipse[3]),
            int(pupil_ellipse[2]),
            orientation=pupil_ellipse[4],
        )
        rr_p = rr_p.clip(6, image.shape[0] - 6)
        cc_p = cc_p.clip(6, image.shape[1] - 6)
        out_image[rr_p, cc_p, ...] = np.array([255, 0, 0])
    return out_image
