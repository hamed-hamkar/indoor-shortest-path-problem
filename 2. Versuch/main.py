"""Run structure detection, show its diagnostic images, and plan a path."""

from pathlib import Path

import cv2
import numpy as np

from open_space_analysis import (
    _point_to_grid,
    build_walkable_grid,
    grid_path_to_pixels,
    theta_star,
)
from preprocessing import preprocess_image
from structure_rec import analyze_structures, recognize_opening_in_wall


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_DIR / "Bilder" / "Floor_Test_1.png"
PATH_WINDOW = "Wegplanung - Start und Ziel anklicken"


def create_all_detections_overlay(original, walls, gaps, windows, doors, furniture):
    """Show every detected rectangle/outline together in distinct colors."""
    overlay = original.copy()
    categories = (
        (walls, (0, 165, 255)),       # orange: walls
        (gaps, (255, 255, 0)),        # cyan: unclassified wall gaps
        (doors, (0, 0, 255)),         # red: doors
        (windows, (0, 220, 0)),       # green: windows
        (furniture, (255, 0, 255)),   # magenta: furniture
    )
    for items, color in categories:
        for item in items:
            corners = item["corners"] if isinstance(item, dict) else item
            polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            if len(polygon) >= 3:
                cv2.polylines(
                    overlay, [polygon], True, color, 2, cv2.LINE_AA
                )
    return overlay


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Grundrissbild nicht gefunden: {IMAGE_PATH}")

    images = preprocess_image(str(IMAGE_PATH))
    walls, windows, doors, furniture = analyze_structures(
        images["binary"], images["original"]
    )
    gaps = recognize_opening_in_wall(walls)

    grid_step = 6
    walkable, clearance = build_walkable_grid(
        images["original"].shape,
        walls,
        windows,
        furniture,
        person_radius_px=7,
        grid_step=grid_step,
    )
    base_image = create_all_detections_overlay(
        images["original"], walls, gaps, windows, doors, furniture
    )

    selected_points = []
    display_image = base_image.copy()
    status = "Linksklick: Startpunkt, danach Zielpunkt | R: neu | Q: Ende"

    def redraw():
        nonlocal display_image
        display_image = base_image.copy()
        if len(selected_points) == 2:
            start_grid = _point_to_grid(selected_points[0], grid_step)
            goal_grid = _point_to_grid(selected_points[1], grid_step)
            path = theta_star(
                walkable,
                start_grid,
                goal_grid,
                clearance=clearance,
                preferred_clearance_px=24,
                comfort_weight=1.5,
            )
            if path:
                path_pixels = grid_path_to_pixels(
                    path, grid_step, base_image.shape
                ).reshape(-1, 1, 2)
                cv2.polylines(
                    display_image, [path_pixels], False, (0, 220, 255), 5,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display_image,
                    "Kein begehbarer Weg zwischen den Punkten",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2, cv2.LINE_AA,
                )

        if selected_points:
            cv2.circle(display_image, selected_points[0], 7, (0, 200, 0), -1)
            cv2.putText(
                display_image, "Start",
                (selected_points[0][0] + 9, selected_points[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 150, 0), 2, cv2.LINE_AA,
            )
        if len(selected_points) > 1:
            cv2.circle(display_image, selected_points[1], 7, (0, 0, 255), -1)
            cv2.putText(
                display_image, "Ziel",
                (selected_points[1][0] + 9, selected_points[1][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
            )

        # A single white status line on a solid dark strip stays readable
        # without overlapping black and white copies of the same text.
        status_top = max(0, display_image.shape[0] - 36)
        cv2.rectangle(
            display_image,
            (0, status_top),
            (display_image.shape[1], display_image.shape[0]),
            (25, 25, 25),
            thickness=-1,
        )
        cv2.putText(
            display_image, status, (12, display_image.shape[0] - 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def handle_mouse(event, x, y, flags, callback_data):
        nonlocal status
        if event == cv2.EVENT_LBUTTONDOWN:
            row, col = _point_to_grid((x, y), grid_step)
            if not (0 <= row < walkable.shape[0] and 0 <= col < walkable.shape[1]):
                status = "Punkt liegt außerhalb des begehbaren Bildbereichs."
                redraw()
                return
            if walkable[row, col] == 0:
                status = "Punkt liegt in einem Hindernis; bitte freien Boden wählen."
                redraw()
                return

            if len(selected_points) == 2:
                selected_points.clear()
            selected_points.append((x, y))
            status = "Zielpunkt anklicken" if len(selected_points) == 1 else (
                "Weg berechnet | R: neu | Q: Ende"
            )
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            selected_points.clear()
            status = "Linksklick: Startpunkt, danach Zielpunkt | R: neu | Q: Ende"
            redraw()

    # Es wird nur das Wegplanungsfenster angezeigt.
    display_height, display_width = images["original"].shape[:2]
    cv2.namedWindow(PATH_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PATH_WINDOW, display_width, display_height)
    cv2.setMouseCallback(PATH_WINDOW, handle_mouse)

    redraw()
    while True:
        cv2.imshow(PATH_WINDOW, display_image)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            selected_points.clear()
            status = "Linksklick: Startpunkt, danach Zielpunkt | R: neu | Q: Ende"
            redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
