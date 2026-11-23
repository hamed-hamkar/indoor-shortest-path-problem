"""Interactive, clearance-aware path planning on a detected floor plan."""

import heapq
import itertools
import math

import cv2
import numpy as np



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


