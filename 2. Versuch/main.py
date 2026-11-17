import cv2
import numpy as np
from pathlib import Path

from preprocessing import preprocess_image
from structure_rec import analyze_structures


# Bild verarbeiten
project_dir = Path(__file__).resolve().parent.parent
image_path = project_dir / "Bilder" / "Floor_Test_1.png"
images = preprocess_image(str(image_path))
structures = analyze_structures(images["binary"])

original = images["original"]
rectangles_overlay = original.copy()

for rectangle in structures["wall_rectangles"]:
    corners = np.asarray(rectangle, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(rectangles_overlay, [corners], True, (0, 255, 255), 2)


cv2.imshow("otsu_image", structures["otsu_image"])
cv2.imshow("Core input", structures["core_image"])
cv2.imshow("EdgeDrawing edges", structures["edge_image"])
cv2.imshow("Filtered wall rectangles", rectangles_overlay)


cv2.waitKey(0)
cv2.destroyAllWindows()