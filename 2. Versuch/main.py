import cv2
import numpy as np
from pathlib import Path

from preprocessing import preprocess_image
from structure_rec import analyze_structures


project_dir = Path(__file__).resolve().parent.parent
image_path = project_dir / "Bilder" / "Floor_Test_1.png"


def draw_labeled_polygon(image, corners, color, label):
    """Draw a polygon and its category label onto the result image."""
    polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], True, color, 2)

    x = int(np.min(polygon[:, 0, 0]))
    y = int(np.min(polygon[:, 0, 1]))
    cv2.putText(
        image,
        label,
        (x, max(15, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def main():
    if not image_path.exists():
        raise FileNotFoundError(f"Grundrissbild nicht gefunden: {image_path}")

    images = preprocess_image(str(image_path))
    (
        wall_rectangles,
        window_rectangles,
        door_rectangles,
        furniture_contours,
    ) = analyze_structures(images["binary"], images["original"])

    # Alle Markierungen werden direkt auf eine Kopie des Originalbilds gezeichnet.
    result_image = images["original"].copy()

    for wall in wall_rectangles:
        draw_labeled_polygon(
            result_image,
            wall["corners"],
            (0, 165, 255),
            "Wand",
        )

    for corners in door_rectangles:
        draw_labeled_polygon(result_image, corners, (0, 0, 255), "Tuer")

    for corners in window_rectangles:
        draw_labeled_polygon(result_image, corners, (0, 255, 0), "Fenster")

    # Möbelkonturen als achsenparallele Rechtecke markieren.
    for contour in furniture_contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 2 or height < 2:
            continue

        cv2.rectangle(
            result_image,
            (x, y),
            (x + width - 1, y + height - 1),
            (255, 0, 255),
            2,
        )
        cv2.putText(
            result_image,
            "Moebel",
            (x, max(15, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Ein Fenster, ein Bild; es wird keine Datei gespeichert und nichts gedruckt.
    cv2.imshow(f"Erkennung: {image_path.name}", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
