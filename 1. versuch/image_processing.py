import cv2
from matplotlib import image
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Fehler beim Laden des Bildes: {path}")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray_image

def detect_walls(gray_image):
    _, binary = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    wall_mask = np.uint8(distance < 2.0) * 255  # Wände sind dicker als 2 Pixel

    return wall_mask