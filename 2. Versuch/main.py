import cv2
import numpy as np
from pathlib import Path

from preprocessing import preprocess_image
from structure_rec import analyze_structures, recognize_opening_in_wall


# Bild verarbeiten
project_dir = Path(__file__).resolve().parent.parent
image_path = project_dir / "Bilder" / "Floor_Test_1.png"
images = preprocess_image(str(image_path))
structures = analyze_structures(images["binary"])

original = images["original"]
rectangles_overlay = original.copy()
gap_rectangles = recognize_opening_in_wall(
    structures["wall_rectangles"], threshold=0
)
gaps_overlay = original.copy()


def draw_gaps_on_image(image, gaps):
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    for gap in gaps:
        points = np.asarray(gap, dtype=np.int32)
        polygon = cv2.convexHull(points).reshape(-1, 1, 2)
        cv2.polylines(overlay, [polygon], True, (0, 0, 255), 2)

    return overlay


for rectangle in structures["wall_rectangles"]:
    corners = np.asarray(rectangle["corners"], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(rectangles_overlay, [corners], True, (0, 255, 255), 2)

gaps_on_binary = draw_gaps_on_image(images["binary"], gap_rectangles)
gaps_overlay = draw_gaps_on_image(gaps_overlay, gap_rectangles)

combined_overlay = rectangles_overlay.copy()
combined_overlay = draw_gaps_on_image(combined_overlay, gap_rectangles)


cv2.imshow("otsu_image", structures["otsu_image"])
cv2.imshow("Core input", structures["core_image"])
cv2.imshow("EdgeDrawing edges", structures["edge_image"])
cv2.imshow("Filtered wall rectangles", rectangles_overlay)
cv2.imshow("Gap rectangles", gaps_overlay)
cv2.imshow("Walls and gap rectangles", combined_overlay)
cv2.imshow("Gaps on binary image", gaps_on_binary)
print("Lücken-Rechtecke:", len(gap_rectangles))


cv2.waitKey(0)
cv2.destroyAllWindows()