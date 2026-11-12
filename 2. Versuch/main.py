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


# Binärbild in ein farbiges Bild umwandeln
gaps_image = cv2.cvtColor(
    images["binary"].copy(),
    cv2.COLOR_GRAY2BGR
)

wall_rectangles_image = cv2.cvtColor(
    structures["wall_mask"].copy(),
    cv2.COLOR_GRAY2BGR,
)
for rectangle in structures["wall_rectangles"]:
    corners = np.asarray(rectangle["corners"], dtype=np.int32)
    cv2.polylines(wall_rectangles_image, [corners], True, (0, 0, 255), 1)


# Fenster und Türen mit eindeutiger Bildevidenz zusammenführen
all_gaps = structures["windows"] + structures["doors"]


# Alle Gaps einzeichnen
for gap in all_gaps:
    (x1, y1), (x2, y2) = gap

    cv2.line(
        gaps_image,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    # Anfangspunkt
    cv2.circle(gaps_image, (x1, y1), 5, (0, 255, 0), -1)

    # Endpunkt
    cv2.circle(gaps_image, (x2, y2), 5, (255, 0, 0), -1)

# Unklassifizierte Wandlücken gelb markieren, damit sie überprüfbar bleiben.
for opening in structures["unknown_openings"]:
    corners = np.asarray(opening["corners"], dtype=np.int32)
    cv2.polylines(gaps_image, [corners], True, (0, 255, 255), 2)


# Alle Zwischenergebnisse gemeinsam anzeigen. Die Anzeige ist zentral hier,
# damit keine Funktion vorher mit waitKey() den Programmablauf anhält.
cv2.imshow("Original", images["original"])
cv2.imshow("Binary", images["binary"])
cv2.imshow("Wall mask", structures["wall_mask"])
cv2.imshow("Wall rectangles", wall_rectangles_image)
cv2.imshow("Furniture candidates", structures["furniture"])
cv2.imshow("All detected gaps", gaps_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
