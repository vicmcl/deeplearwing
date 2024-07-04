import numpy as np
import cv2
from pathlib import Path

DATA_PATH = Path(__file__).parents[2] / "data"


def compute_curvature(image_array, n=5, smooth_value=3):
    I = (image_array * 255).astype(np.uint8)
    GX, GY = compute_gradients(I)
    I = threshold_image(I)
    C = find_contours(I)
    heatmap = compute_heatmap(C, GX, GY, n)
    if smooth_value is not None:
        heatmap = smooth_heatmap(heatmap, smmooth_value=smooth_value)
    return heatmap


def compute_gradients(image):
    GX = cv2.Scharr(image, cv2.CV_32F, 1, 0, scale=1)
    GY = cv2.Scharr(image, cv2.CV_32F, 0, 1, scale=1)
    GX = GX + 0.0001  # Avoid div by zero
    return GX, GY


def threshold_image(image):
    _, thresholded_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image


def find_contours(image):
    contours, _ = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return contours


def compute_heatmap(contours, GX, GY, n):
    heatmap = np.zeros_like(GX, dtype=float)
    for contour in contours:
        contour = contour.squeeze()
        N = len(contour)
        for i in range(N):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + n) % N]
            gx1 = GX[y1, x1]
            gy1 = GY[y1, x1]
            gx2 = GX[y2, x2]
            gy2 = GY[y2, x2]
            cos_angle = gx1 * gx2 + gy1 * gy2
            cos_angle /= np.linalg.norm((gx1, gy1)) * np.linalg.norm((gx2, gy2))
            cos_angle = min(1, cos_angle)
            angle = np.arccos(cos_angle)
            if cos_angle < 0:
                angle = np.pi - angle
            x1, y1 = contour[((2 * i + n) // 2) % N]
            heatmap[y1, x1] = angle
    return heatmap


def smooth_heatmap(heatmap, smmooth_value):
    smoothed_heatmap = cv2.GaussianBlur(
        heatmap, (smmooth_value, smmooth_value), heatmap.max()
    )
    return smoothed_heatmap
