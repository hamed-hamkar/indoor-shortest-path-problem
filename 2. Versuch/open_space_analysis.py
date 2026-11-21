"""Interactive, clearance-aware path planning on a detected floor plan."""

import heapq
import itertools
import math
from pathlib import Path

import cv2
import numpy as np

from preprocessing import preprocess_image
from structure_rec import analyze_structures


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_DIR / "Bilder" / "Floor_Test_1.png"
WINDOW_NAME = "Wegplanung - Start und Ziel anklicken"


def build_walkable_grid(
    image_shape,
    wall_rectangles,
    window_rectangles,
    furniture_outlines,
    person_radius_px=7,
    grid_step=3,
):
    """Build a conservative walkable grid, accounting for a person's width.

    Detected walls, windows, and furniture are obstacles. Door gaps are left
    open because wall detection represents the wall segments on either side
    of a gap; expanded door swing rectangles are deliberately not blocked.
    """
    height, width = image_shape[:2]
    obstacle_mask = np.zeros((height, width), dtype=np.uint8)

    for wall in wall_rectangles:
        corners = wall["corners"] if isinstance(wall, dict) else wall
        polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(obstacle_mask, polygon, 255)

    for corners in window_rectangles:
        polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(obstacle_mask, polygon, 255)

    for outline in furniture_outlines:
        polygon = np.asarray(outline, dtype=np.int32).reshape(-1, 1, 2)
        if len(polygon) >= 3:
            cv2.fillPoly(obstacle_mask, [polygon], 255)

    # Inflate obstacles so the path represents the person's center, not their
    # outer boundary. The extra radius is configurable in source-image pixels.
    if person_radius_px > 0:
        diameter = 2 * int(person_radius_px) + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (diameter, diameter)
        )
        obstacle_mask = cv2.dilate(obstacle_mask, kernel)

    # Pad to whole cells. Padding is blocked so routes cannot escape the image.
    grid_step = max(1, int(grid_step))
    grid_height = math.ceil(height / grid_step)
    grid_width = math.ceil(width / grid_step)
    padded = cv2.copyMakeBorder(
        obstacle_mask,
        0,
        grid_height * grid_step - height,
        0,
        grid_width * grid_step - width,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    blocked_cells = padded.reshape(
        grid_height, grid_step, grid_width, grid_step
    ).max(axis=(1, 3))
    walkable = (blocked_cells == 0).astype(np.uint8)

    # Distance from each walkable cell to the nearest obstacle, in source
    # pixels. Theta* uses this to prefer paths with more comfortable clearance.
    clearance = cv2.distanceTransform(walkable, cv2.DIST_L2, 5) * grid_step
    return walkable, clearance


def _line_cells(start, end):
    """Sample a grid line densely enough for collision and clearance checks."""
    row0, col0 = start
    row1, col1 = end
    steps = max(abs(row1 - row0), abs(col1 - col0)) * 2
    if steps == 0:
        return [start]

    cells = []
    for index in range(steps + 1):
        fraction = index / steps
        cell = (
            int(round(row0 + (row1 - row0) * fraction)),
            int(round(col0 + (col1 - col0) * fraction)),
        )
        if not cells or cell != cells[-1]:
            cells.append(cell)
    return cells


def _line_of_sight(walkable, start, end):
    """Return whether a segment stays in free cells without corner cutting."""
    previous = None
    height, width = walkable.shape
    for row, col in _line_cells(start, end):
        if not (0 <= row < height and 0 <= col < width):
            return False
        if walkable[row, col] == 0:
            return False
        if previous is not None:
            prev_row, prev_col = previous
            row_step = row - prev_row
            col_step = col - prev_col
            if row_step and col_step:
                if (
                    walkable[prev_row + row_step, prev_col] == 0
                    or walkable[prev_row, prev_col + col_step] == 0
                ):
                    return False
        previous = (row, col)
    return True


def _segment_cost(start, end, clearance, preferred_clearance_px, comfort_weight):
    """Measure distance plus a soft penalty for walking close to obstacles."""
    cells = _line_cells(start, end)
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    if distance == 0:
        return 0.0

    if preferred_clearance_px <= 0 or comfort_weight <= 0:
        return distance

    risks = [
        max(
            0.0,
            min(1.0, (preferred_clearance_px - clearance[row, col])
                / preferred_clearance_px),
        )
        for row, col in cells
    ]
    mean_risk = sum(risks) / len(risks)
    return distance * (1.0 + comfort_weight * mean_risk)


def theta_star(
    walkable,
    start,
    goal,
    clearance=None,
    preferred_clearance_px=24,
    comfort_weight=1.5,
):
    """Find an any-angle path with Theta*, biased toward wider open space.

    ``start`` and ``goal`` are ``(row, column)`` grid coordinates. Returns a
    list of grid coordinates, or ``None`` when no path exists.
    """
    walkable = np.asarray(walkable, dtype=np.uint8)
    if walkable.ndim != 2:
        raise ValueError("walkable muss eine zweidimensionale Maske sein")

    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    height, width = walkable.shape
    for point in (start, goal):
        row, col = point
        if not (0 <= row < height and 0 <= col < width):
            return None
        if walkable[row, col] == 0:
            return None

    if clearance is None:
        clearance = cv2.distanceTransform(walkable, cv2.DIST_L2, 5)

    def heuristic(point):
        return math.hypot(goal[0] - point[0], goal[1] - point[1])

    sequence = itertools.count()
    open_set = [(heuristic(start), 0.0, next(sequence), start)]
    g_score = {start: 0.0}
    parent = {start: start}
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    )

    while open_set:
        _, queued_cost, _, current = heapq.heappop(open_set)
        if queued_cost > g_score.get(current, math.inf):
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]

        current_row, current_col = current
        for row_offset, col_offset in neighbors:
            neighbor = (current_row + row_offset, current_col + col_offset)
            row, col = neighbor
            if not (0 <= row < height and 0 <= col < width):
                continue
            if walkable[row, col] == 0:
                continue

            # Theta*: try connecting the neighbor directly to the current
            # node's parent; this removes grid-heading constraints where clear.
            ancestor = parent[current]
            if _line_of_sight(walkable, ancestor, neighbor):
                candidate_parent = ancestor
                candidate_cost = g_score[ancestor] + _segment_cost(
                    ancestor,
                    neighbor,
                    clearance,
                    preferred_clearance_px,
                    comfort_weight,
                )
            else:
                if not _line_of_sight(walkable, current, neighbor):
                    continue
                candidate_parent = current
                candidate_cost = g_score[current] + _segment_cost(
                    current,
                    neighbor,
                    clearance,
                    preferred_clearance_px,
                    comfort_weight,
                )

            if candidate_cost >= g_score.get(neighbor, math.inf):
                continue
            g_score[neighbor] = candidate_cost
            parent[neighbor] = candidate_parent
            heapq.heappush(
                open_set,
                (
                    candidate_cost + heuristic(neighbor),
                    candidate_cost,
                    next(sequence),
                    neighbor,
                ),
            )

    return None


def grid_path_to_pixels(path, grid_step, image_shape):
    """Convert grid-cell centers to ``(x, y)`` image coordinates."""
    if not path:
        return np.empty((0, 2), dtype=np.int32)
    height, width = image_shape[:2]
    points = [
        (
            min(width - 1, col * grid_step + grid_step // 2),
            min(height - 1, row * grid_step + grid_step // 2),
        )
        for row, col in path
    ]
    return np.asarray(points, dtype=np.int32)


def _point_to_grid(point, grid_step):
    x, y = point
    return int(y // grid_step), int(x // grid_step)


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Grundrissbild nicht gefunden: {IMAGE_PATH}")

    images = preprocess_image(str(IMAGE_PATH))
    walls, windows, doors, furniture = analyze_structures(
        images["binary"], images["original"]
    )
    # Six source pixels per search cell keeps the Python implementation
    # responsive while retaining enough detail for room-scale paths.
    grid_step = 6
    walkable, clearance = build_walkable_grid(
        images["original"].shape,
        walls,
        windows,
        furniture,
        person_radius_px=7,
        grid_step=grid_step,
    )

    selected_points = []
    base_image = images["original"].copy()
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
                    display_image,
                    [path_pixels],
                    False,
                    (0, 220, 255),
                    5,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display_image,
                    "Kein begehbarer Weg zwischen den Punkten",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        if selected_points:
            cv2.circle(display_image, selected_points[0], 7, (0, 200, 0), -1)
            cv2.putText(
                display_image, "Start", (selected_points[0][0] + 9, selected_points[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 150, 0), 2, cv2.LINE_AA,
            )
        if len(selected_points) > 1:
            cv2.circle(display_image, selected_points[1], 7, (0, 0, 255), -1)
            cv2.putText(
                display_image, "Ziel", (selected_points[1][0] + 9, selected_points[1][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
            )

        cv2.putText(
            display_image,
            status,
            (12, display_image.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_image,
            status,
            (12, display_image.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def handle_mouse(event, x, y, flags, callback_data):
        nonlocal status
        if event == cv2.EVENT_LBUTTONDOWN:
            grid_point = _point_to_grid((x, y), grid_step)
            row, col = grid_point
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
            status = (
                "Zielpunkt anklicken"
                if len(selected_points) == 1
                else "Weg berechnet | R: neu | Q: Ende"
            )
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            selected_points.clear()
            status = "Linksklick: Startpunkt, danach Zielpunkt | R: neu | Q: Ende"
            redraw()

    redraw()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)
    while True:
        cv2.imshow(WINDOW_NAME, display_image)
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
