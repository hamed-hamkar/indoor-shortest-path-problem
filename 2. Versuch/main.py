import cv2
import numpy as np
from pathlib import Path

from preprocessing import preprocess_image
from structure_rec import analyze_structures, recognize_opening_in_wall


project_dir = Path(__file__).resolve().parent.parent
image_path = project_dir / "Bilder" / "Floor_Test_1.png"
images = preprocess_image(str(image_path))
(
    wall_rectangles,
    window_rectangles,
    door_rectangles,
    rest_gap_rectangles,
    gap_analysis,
    parallel_group_lengths,
    door_window_threshold,
    classified_openings,
) = analyze_structures(
    images["binary"],
    images["original"],
)
gap_rectangles = recognize_opening_in_wall(wall_rectangles, threshold=0)

hough_lines_overlay = np.zeros_like(images["binary"], dtype=np.uint8)
hough_lines_overlay = cv2.cvtColor(
    hough_lines_overlay,
    cv2.COLOR_GRAY2BGR,
)

for gap_result in gap_analysis:
    for segment in gap_result["gap_segments"]:
        for point_a, point_b in zip(segment[:-1], segment[1:]):
            cv2.line(
                hough_lines_overlay,
                (int(point_a[0]), int(point_a[1])),
                (int(point_b[0]), int(point_b[1])),
                (0, 255, 255),
                1,
            )

for gap in gap_rectangles:
    polygon = np.asarray(gap, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        hough_lines_overlay,
        [polygon],
        True,
        (0, 0, 255),
        1,
    )

print("parallel_group_counts:", parallel_group_lengths)
print("door_window_threshold:", door_window_threshold)

cv2.imshow("Hough lines in gaps", hough_lines_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()