import cv2
import numpy as np

def geometrical_analysis_image(image):
    contours, hierarchy = cv2.findContours(
        image.copy(),
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, hierarchy