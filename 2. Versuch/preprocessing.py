import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Check if the image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the original grayscale image directly so thin line details
    # are not weakened by the blur before wall extraction.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    return {
        "original": img,
        "gray": gray,
        "binary": binary,
    }
