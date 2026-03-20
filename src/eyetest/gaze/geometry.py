from __future__ import annotations

import math

import numpy as np


def vector_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def get_vector_onto_plane(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    direction = direction / vector_norm(direction)
    normal = normal / vector_norm(normal)
    return direction - np.dot(direction, normal) * normal


def get_rotation(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
) -> np.ndarray:
    source = np.array([sx, sy, sz])
    destination = np.array([dx, dy, dz])
    h_matrix = source.T @ destination
    u_mat, _, vt_mat = np.linalg.svd(h_matrix)
    return vt_mat.T @ u_mat.T


def esti_normal_fun(
    amajor: float,
    aminor: float,
    xe: float,
    ye: float,
    theta_deg: float,
    focal_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = theta_deg / 180.0 * math.pi
    a = 1 / (amajor**2) * math.cos(theta) ** 2 + 1 / (aminor**2) * math.sin(theta) ** 2
    b = 1 / (amajor**2) * math.sin(theta) ** 2 + 1 / (aminor**2) * math.cos(theta) ** 2
    c = (1 / (amajor**2) - 1 / (aminor**2)) * math.sin(2 * theta)
    d = 1 / (amajor**2) * (
        -2 * xe * math.cos(theta) ** 2 - ye * math.sin(2 * theta)
    ) - 1 / (aminor**2) * (
        2 * xe * math.sin(theta) ** 2 - ye * math.sin(2 * theta)
    )
    e = 1 / (amajor**2) * (
        -xe * math.sin(2 * theta) - 2 * ye * math.sin(theta) ** 2
    ) + 1 / (aminor**2) * (
        xe * math.sin(2 * theta) - 2 * ye * math.cos(theta) ** 2
    )
    f = (
        (xe * math.cos(theta) + ye * math.sin(theta)) ** 2 / (amajor**2)
        + (xe * math.sin(theta) - ye * math.cos(theta)) ** 2 / (aminor**2)
        - 1
    )
    a = a / f
    b = b / f
    c = c / f
    d = d / f
    e = e / f
    e_matrix = np.array([[a, c / 2, d / 2], [c / 2, b, e / 2], [d / 2, e / 2, f]])

    a_q = a * ((-focal_length) ** 2)
    b_q = b * ((-focal_length) ** 2)
    c_q = c * ((-focal_length) ** 2)
    d_q = d * (-focal_length)
    e_q = e * (-focal_length)
    q_matrix = np.array([[a_q, c_q / 2, d_q / 2], [c_q / 2, b_q, e_q / 2], [d_q / 2, e_q / 2, 1]])

    eigenvalues, eigenvectors = np.linalg.eig(q_matrix)

    if eigenvalues[0] * eigenvalues[1] > 0:
        lamda3 = abs(eigenvalues[2])
        vec3 = eigenvectors[:, 2] if np.dot(eigenvectors[:, 2], np.array([0, 0, 1])) > 0 else -eigenvectors[:, 2]
        if abs(eigenvalues[0]) > abs(eigenvalues[1]):
            lamda1 = abs(eigenvalues[0])
            lamda2 = abs(eigenvalues[1])
            vec2 = eigenvectors[:, 1]
        else:
            lamda1 = abs(eigenvalues[1])
            lamda2 = abs(eigenvalues[0])
            vec2 = eigenvectors[:, 0]
    elif eigenvalues[0] * eigenvalues[2] > 0:
        lamda3 = abs(eigenvalues[1])
        vec3 = eigenvectors[:, 1] if np.dot(eigenvectors[:, 1], np.array([0, 0, 1])) > 0 else -eigenvectors[:, 1]
        if abs(eigenvalues[0]) > abs(eigenvalues[2]):
            lamda1 = abs(eigenvalues[0])
            lamda2 = abs(eigenvalues[2])
            vec2 = eigenvectors[:, 2]
        else:
            lamda1 = abs(eigenvalues[2])
            lamda2 = abs(eigenvalues[0])
            vec2 = eigenvectors[:, 0]
    else:
        lamda3 = abs(eigenvalues[0])
        vec3 = eigenvectors[:, 0] if np.dot(eigenvectors[:, 0], np.array([0, 0, 1])) > 0 else -eigenvectors[:, 0]
        if abs(eigenvalues[1]) > abs(eigenvalues[2]):
            lamda1 = abs(eigenvalues[1])
            lamda2 = abs(eigenvalues[2])
            vec2 = eigenvectors[:, 2]
        else:
            lamda1 = abs(eigenvalues[2])
            lamda2 = abs(eigenvalues[1])
            vec2 = eigenvectors[:, 1]

    vec1 = np.cross(vec2, vec3)

    factor_i = math.sqrt((lamda3 * (lamda1 - lamda2)) / (lamda1 * (lamda1 + lamda3)))
    factor_j = math.sqrt((lamda1 * (lamda2 + lamda3)) / (lamda3 * (lamda1 + lamda3)))
    factor_d = math.sqrt((lamda1 - lamda2) / (lamda1 + lamda3))
    factor_e = math.sqrt((lamda2 + lamda3) / (lamda1 + lamda3))

    i1 = -vec1 * factor_i + vec3 * factor_j
    d1 = -vec1 * factor_d - vec3 * factor_e
    i2 = vec1 * factor_i + vec3 * factor_j
    d2 = vec1 * factor_d - vec3 * factor_e
    return i1, i2, d1, d2, e_matrix


def rigid_transform_3d(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid_a = np.mean(matrix_a, axis=0)
    centroid_b = np.mean(matrix_b, axis=0)
    aa = matrix_a - centroid_a
    bb = matrix_b - centroid_b
    h_matrix = aa.T @ bb
    u_mat, _, vt_mat = np.linalg.svd(h_matrix)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[2, :] *= -1
        rotation = vt_mat.T @ u_mat.T
    translation = -rotation @ centroid_a + centroid_b
    return rotation, translation


def trans_camera_to_screen(screen_corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_points = np.array(screen_corners)
    screen_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [vector_norm(screen_corners[1] - screen_corners[0]), 0.0, 0.0],
            [
                vector_norm(screen_corners[1] - screen_corners[0]),
                vector_norm(screen_corners[2] - screen_corners[1]),
                0.0,
            ],
            [0.0, vector_norm(screen_corners[2] - screen_corners[1]), 0.0],
        ]
    )
    return rigid_transform_3d(camera_points, screen_points)


def line_plane_intersection(
    point_on_line: np.ndarray,
    line_direction: np.ndarray,
    point_on_plane: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    scale = (np.dot(point_on_plane, plane_normal) - np.dot(point_on_line, plane_normal)) / np.dot(
        line_direction, plane_normal
    )
    return point_on_line + scale * line_direction
