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


# Die Rechtecke sowohl auf dem Originalbild als auch auf der Wandmaske anzeigen.
rectangles_on_original = images["original"].copy()
rectangles_on_mask = cv2.cvtColor(
    structures["wall_mask"].copy(),
    cv2.COLOR_GRAY2BGR,
)

for rectangle_corners in structures["wall_rectangles"]:
    corners = np.asarray(rectangle_corners, dtype=np.int32)
    cv2.polylines(rectangles_on_original, [corners], True, (0, 0, 255), 2)
    cv2.polylines(rectangles_on_mask, [corners], True, (0, 0, 255), 2)


cv2.imshow("Original", images["original"])
cv2.imshow("Binary", images["binary"])
cv2.imshow("Wall mask", structures["wall_mask"])
cv2.imshow("Rectangles on original", rectangles_on_original)
cv2.imshow("Rectangles on wall mask", rectangles_on_mask)
cv2.imshow("otsu_image", structures["otsu_image"])

cv2.waitKey(0)
cv2.destroyAllWindows()
