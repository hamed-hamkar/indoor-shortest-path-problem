import cv2
import numpy as np

def analyze_structures(binary_image, original_image=None):
    wall_rectangles = extract_wall_candidates(binary_image)
    wall_gaps_pos = recognize_opening_in_wall(wall_rectangles)
    gap_line = wall_gap_interpreter(
        wall_gaps_pos,
        binary_image,
        original_image,
    )
    
    window_rectangles, door_rectangles = window_or_door(gap_line, binary_image)
    furniture_rectangles = check_for_furniture(
        binary_image,
        wall_rectangles,
        door_rectangles,
        window_rectangles,
    )


    return (
        wall_rectangles,
        window_rectangles,
        door_rectangles,
        furniture_rectangles
    )


def extract_wall_candidates(binary_image):
    if binary_image is None or binary_image.ndim != 2:
        raise ValueError(
            "binary_image in extract_wall_candidates ist kein zweidimensionales Binärbild"
        )

    distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    foreground_distances = distance[binary_image > 0]

    # Wenn forgorund_distance keine Werte hat, gibt es keine Wände
    if foreground_distances.size == 0:
        return []

    otsu_image = np.zeros_like(binary_image)

    # Otsu ist ein Verfahren zur automatischen Bestimmung eines Schwellenwertes,
    # eignet sich, wenn es wenn man Daten in 2 Kategorien unterteilen will
    # in unserem Fall, dünne und dicke Linien (Wand ist die dicke Linie, Rest ist die dünne Linie)
    # Otsu Erwartet 8- oder 16-Bit Int Werte in einem 2 Dimensionalen Array,
    # foreground_distance speichert Pixel in Int 32-Bit in einem 1 Dimensionalen Array
    # reshape(-1, 1): -1 = beliebig viele Zeilen, wie benötigt werden; 1 = es soll nur 1 Spalte geben 

    foreground_distances_u8 = (
        np.clip(foreground_distances, 0, 255)
        .reshape(-1, 1)
        .astype(np.uint8)
    )

    otsu_threshold, otsu_values = cv2.threshold(
        foreground_distances_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_image[binary_image > 0] = otsu_values.ravel()

    # Otsu kann dünnere Wände abschneiden, damit das nicht passiert, verkleinern wir den otsu_threshold
    # und sorgen dafür, dass es nicht kleiner als 1 Pixel werden kann
    core_threshold = max(1.0, float(otsu_threshold) * 0.9)
    core = np.where(distance >= core_threshold, 255, 0).astype(np.uint8)
    core_distance = distance[core > 0]

    # HoughLinesP funktioniert nicht gut mit der erkennnung von den kleinen Wänden,
    # nutze Edges, cv2.Canny(), cv2.findContours, cv2.approxPolyDP oder ed = cv2.ximgprox.createEdgeDrawing()
    # und dann ed.detectEdges() und dann ed.getSegments() und dann basierend auf den Linien-Punkten Rechtecke bilden

    segments = []
    if np.any(core):
        outlines = cv2.ximgproc.createEdgeDrawing()
        outlines.detectEdges(core)
        segments = outlines.getSegments() or []

    simplified_segments = segments_helper(segments)

    # Rechteckbildung:
    # [(50, 40), (50, 100)],            # gerader Abschnitt
    # [(20, 30), (80, 30), (80, 90)]    # Abschnitt mit einer Ecke

    wall_thickness = 3 * float(np.median(core_distance))

    line_pairs = find_parallel_pairs(simplified_segments, wall_thickness)

    rectangles = []

    for line_a, line_b in line_pairs:
        _, horizontal, pos_a, start_a, end_a = line_a
        _, _, pos_b, start_b, end_b = line_b

        start = min(start_a, start_b)
        end = max(end_a, end_b)

        if start >= end:
            continue

        if horizontal:
            top, bottom = sorted((pos_a, pos_b))
            corners = (
                (int(round(start)), int(round(top))),
                (int(round(end)), int(round(top))),
                (int(round(start)), int(round(bottom))),
                (int(round(end)), int(round(bottom)))
            )
            horizontal = True
        else:
            left, right = sorted((pos_a, pos_b))
            corners = (
                (int(round(left)), int(round(start))),
                (int(round(right)), int(round(start))),
                (int(round(left)), int(round(end))),
                (int(round(right)), int(round(end)))
            )
            horizontal = False

        rectangles.append((horizontal, corners))

    rectangles = filter_wall_rectangles(rectangles, otsu_image)
    rectangles = expand_rectangles_and_add_gap_checkers(
        rectangles, margin=3, image_shape=binary_image.shape
    )

    return rectangles


def recognize_opening_in_wall(wall_rectangles, threshold=0):
    bands = {}
    for rectangle in wall_rectangles:
        bands.setdefault(rectangle["group_id"], []).append(rectangle)

    openings = []
    for rectangles in bands.values():
        rectangles.sort(key=lambda item: item["start"])
        for first, second in zip(rectangles, rectangles[1:]):
            local_wall_thickness = (
                first["half_thickness"] + second["half_thickness"]
            )
            if min(
                first["end"] - first["start"] + 1,
                second["end"] - second["start"] + 1,
            ) < local_wall_thickness:
                continue

            gap_start = first["end"] + 1
            gap_end = second["start"] - 1
            if gap_start > gap_end:
                continue

            half_thickness = int(round(local_wall_thickness / 2)) + threshold
            centerline = int(
                round((first["centerline"] + second["centerline"]) / 2)
            )
            if first["horizontal"]:
                corners = (
                    (gap_start, centerline - 2 * half_thickness),
                    (gap_end, centerline - 2 * half_thickness),
                    (gap_end, centerline + 2 * half_thickness),
                    (gap_start, centerline + 2 * half_thickness),
                )
            else:
                corners = (
                    (centerline - 2 * half_thickness, gap_start),
                    (centerline + 2 * half_thickness, gap_start),
                    (centerline + 2 * half_thickness, gap_end),
                    (centerline - 2 * half_thickness, gap_end),
                )

            openings.append(
                tuple((int(x), int(y)) for x, y in corners)
            )

    return openings


def wall_gap_interpreter(wall_openings, binary_image, original_image=None):
    opening_regions = []
    if original_image is None:
        original_gray = None
    elif original_image.ndim == 3:
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_image

    for opening_id, opening in enumerate(wall_openings):
        polygon = np.asarray(opening, dtype=np.int32)
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        opening_regions.append({
            "opening_id": opening_id,
            "corners": opening,
            "bounds": (xs.min(), ys.min(), xs.max(), ys.max()),
        })

    results = []

    for region in opening_regions:
        left, top, right, bottom = region["bounds"]
        # Wall openings are axis-aligned rectangles, so crop the binary image
        # directly instead of masking and copying the entire image first.
        gap_roi = binary_image[top:bottom + 1, left:right + 1]
        gap_width = right - left + 1
        gap_height = bottom - top + 1
        gap_orientation = (
            "horizontal" if gap_width >= gap_height else "vertical"
        )
        wall_thickness = (
            gap_height if gap_orientation == "horizontal" else gap_width
        )
        wall_thickness = max(1, int(round(wall_thickness)))

        if not np.any(gap_roi):
            parallel_groups = []
            gap_segments = []
        else:
            if original_gray is None:
                gap_source = gap_roi
            else:
                gap_source = original_gray[
                    top:bottom + 1,
                    left:right + 1,
                ]

            gap_edges = cv2.Canny(gap_source, 50, 150)
            hough_lines = cv2.HoughLinesP(
                gap_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=15,
                minLineLength=12,
                maxLineGap=3,
            )

            simplified_segments = []
            gap_segments = []
            if hough_lines is not None:
                for x1, y1, x2, y2 in hough_lines.reshape(-1, 4):
                    segment = [
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                    ]
                    simplified_segments.append(segment)
                    gap_segments.append([
                        (int(x1) + left, int(y1) + top),
                        (int(x2) + left, int(y2) + top),
                    ])

            parallel_groups = find_parallel_groups(
                simplified_segments, wall_thickness=wall_thickness
            )

        results.append({
            "opening_id": region["opening_id"],
            "corners": region["corners"],
            "wall_thickness": wall_thickness,
            "gap_orientation": gap_orientation,
            "parallel_groups": parallel_groups,
            "gap_segments": gap_segments,
        })

    return results


def window_or_door(openings_wall, binary_image):
    window = []
    door = []
    parallel_group_lengths = [
        max((len(group) for group in opening["parallel_groups"]), default=0)
        for opening in openings_wall
    ]
    ink_points = np.column_stack(np.nonzero(binary_image))[:, ::-1]

    # Determine the count threshold once for all openings.
    threshold = classify_parallel_threshold(parallel_group_lengths)

    for opening, parallel_count in zip(openings_wall, parallel_group_lengths):
        corners = opening["corners"]
        gap_thickness = opening["wall_thickness"]
        gap_orientation = opening["gap_orientation"]

        needs_arc_check = (
            parallel_count >= 2
            and (threshold is None or parallel_count <= threshold)
        )
        arc_geometry = None
        if needs_arc_check:
            arc_geometry = detect_door_arc_geometry(
                binary_image,
                corners,
                gap_thickness,
                gap_orientation,
                ink_points,
            )
        classification = classify_opening_by_count_and_arc(
            parallel_count,
            threshold,
            arc_geometry is not None,
        )

        if classification == "door":
            swing_area_corners = expand_door_swing_area(
                corners,
                arc_geometry,
                binary_image.shape,
            )
            door.append(swing_area_corners)
        elif classification == "window":
            window.append(corners)

    return window, door



def check_for_furniture(
    binary_image,
    wall_rectangles,
    door_rectangles,
    window_rectangles,
):
    """Return foreground drawing pixels outside walls and known openings."""
    everything_except_furniture_mask = np.zeros_like(binary_image)
    rectangles = []
    for wall in wall_rectangles:
        # Wall detections are dictionaries; accept raw polygons too.
        rectangles.append(wall["corners"] if isinstance(wall, dict) else wall)
    rectangles.extend(door_rectangles)
    rectangles.extend(window_rectangles)

    for rectangle in rectangles:
        polygon = np.asarray(rectangle, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(everything_except_furniture_mask, polygon, 255)

    furniture_image = cv2.bitwise_and(
        binary_image,
        binary_image,
        mask=cv2.bitwise_not(everything_except_furniture_mask),
    )

    # Keep nested contours temporarily so the positional filter below can
    # remove furniture candidates enclosed by another candidate.
    furniture_contours, _ = cv2.findContours(
        furniture_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    furniture_contours = remove_small_contours(furniture_contours)
    furniture_contours = remove_contours_inside_other_contours(furniture_contours)

    return orthogonal_contours(furniture_contours, binary_image.shape)


def remove_small_contours(contours, min_area=50.0):
    """Discard tiny contours using their pixel area."""
    return [
        contour
        for contour in contours
        if abs(cv2.contourArea(contour)) >= min_area
    ]


def remove_contours_inside_other_contours(contours):
    """Remove contours whose bounding rectangle lies inside another one.

    This is a positional check and does not use the ``findContours`` hierarchy
    or require the inner contour to touch the enclosing contour's outline.
    """
    if not contours:
        return []

    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    contained = [False] * len(contours)

    for inner_index, (ix, iy, iw, ih) in enumerate(bounding_boxes):
        if iw <= 0 or ih <= 0:
            contained[inner_index] = True
            continue

        inner_right = ix + iw
        inner_bottom = iy + ih
        inner_box_area = iw * ih

        for outer_index, (ox, oy, ow, oh) in enumerate(bounding_boxes):
            if inner_index == outer_index or ow <= 0 or oh <= 0:
                continue

            # The outer box must be strictly larger, to avoid treating two
            # equal boxes as containing one another.
            if ow * oh <= inner_box_area:
                continue

            if ix >= ox and iy >= oy and inner_right <= ox + ow and inner_bottom <= oy + oh:
                contained[inner_index] = True
                break

    return [contour for index, contour in enumerate(contours) if not contained[index]]


def orthogonal_contours(contours, image_shape):
    """Return one horizontal/vertical outline for each connected furniture group."""
    if not contours:
        return []

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Filling all detected contours preserves connected groups while leaving
    # spatially separate furniture as separate components.
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    outer_contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    outlines = []
    for contour in outer_contours:
        boundary = contour.reshape(-1, 2)
        if len(boundary) < 3:
            continue

        # CHAIN_APPROX_NONE gives adjacent boundary pixels. For diagonal steps,
        # insert one corner pixel so every resulting edge is horizontal or
        # vertical; this keeps the complete raster outline instead of
        # simplifying away thin parts of a furniture symbol.
        orthogonal = []
        for index, point in enumerate(boundary):
            next_point = boundary[(index + 1) % len(boundary)]
            x1, y1 = map(int, point)
            x2, y2 = map(int, next_point)
            if x1 != x2 and y1 != y2:
                orthogonal.append((x2, y1))
            orthogonal.append((x2, y2))

        # Drop duplicate and collinear points to keep the polygon compact.
        cleaned = []
        for point in orthogonal:
            if not cleaned or point != cleaned[-1]:
                cleaned.append(point)
        if len(cleaned) > 2 and cleaned[0] == cleaned[-1]:
            cleaned.pop()

        changed = True
        while changed and len(cleaned) > 3:
            changed = False
            simplified = []
            count = len(cleaned)
            for index, point in enumerate(cleaned):
                previous = cleaned[(index - 1) % count]
                following = cleaned[(index + 1) % count]
                same_x = previous[0] == point[0] == following[0]
                same_y = previous[1] == point[1] == following[1]
                if same_x or same_y:
                    changed = True
                else:
                    simplified.append(point)
            if len(simplified) < 3:
                break
            cleaned = simplified

        if len(cleaned) >= 3:
            outlines.append(np.asarray(cleaned, dtype=np.int32).reshape(-1, 1, 2))

    return outlines


def segments_helper(segments, epsilon=2):
    simplified_segments = []

    for segment in segments:
        points = np.asarray(segment, dtype=np.int32).reshape(-1, 1, 2)

        if len(points) < 2:
            continue

        corners = cv2.approxPolyDP(
            points, epsilon=epsilon, closed=False
        ).reshape(-1, 2)

        simplified_segments.append([tuple(map(int, point)) for point in corners])

    return simplified_segments


def classify_parallel_threshold(parallel_counts):
    # 0 und 1 werden als leere beziehungsweise nicht verwertbare Gruppen behandelt.
    relevant_values = sorted({
        int(value)
        for value in parallel_counts
        if value >= 2
    })

    # Ohne positive Werte gibt es weder eine Tür- noch eine Fenstergrenze.
    if not relevant_values:
        return None

    # Bei genau einem Wert wird dieser Wert als Übergangspunkt verwendet.
    if len(relevant_values) == 1:
        return relevant_values[0]

    # Bei genau zwei Werten trennt ihre Mitte Tür- und Fensterbereich.
    if len(relevant_values) == 2:
        return (relevant_values[0] + relevant_values[1]) / 2

    # Bei genau drei Werten liegt der mittlere Wert auf der Grenze.
    if len(relevant_values) == 3:
        return relevant_values[1]

    # Bei vier oder mehr Werten wird der Wert mit der größten Summe seiner
    # direkten Nachbarabstände gewählt. Das beschränkt die Bewertung auf
    # die lokale Trennung und vermeidet, dass Extremwerte alle Summe bestimmen.
    maximum_separation = -1
    threshold = None
    for index, value in enumerate(relevant_values):
        neighbor_distances = []
        if index > 0:
            neighbor_distances.append(value - relevant_values[index - 1])
        if index < len(relevant_values) - 1:
            neighbor_distances.append(relevant_values[index + 1] - value)
        distance_sum = sum(neighbor_distances)
        if distance_sum > maximum_separation:
            maximum_separation = distance_sum
            threshold = value

    return threshold


def expand_door_swing_area(corners, arc_geometry, image_shape):
    """Fit the door rectangle to the opening and measured arc pixel bounds."""
    points = np.asarray(corners, dtype=np.int32).reshape(-1, 2)
    height, width = image_shape[:2]
    left = int(points[:, 0].min())
    right = int(points[:, 0].max())
    top = int(points[:, 1].min())
    bottom = int(points[:, 1].max())

    if arc_geometry is not None:
        arc_left, arc_top, arc_right, arc_bottom = arc_geometry["bounds"]
        side = arc_geometry["side"]
        left = min(left, arc_left)
        right = max(right, arc_right)
        top = min(top, arc_top)
        bottom = max(bottom, arc_bottom)

        if side == "left":
            top = max(0, top - int(abs(left - right) / 6))
            bottom = min(height, bottom + int(abs(left - right) / 6))
            left = max(0, left - int(abs(left - right) / 6))
        elif side == "right":
            top = max(0, top - int(abs(left - right) / 6))
            bottom = min(height, bottom + int(abs(left - right) / 6))
            right = min(width - 1, right + int(abs(left - right) / 6))
        elif side == "top":
            left = max(0, left - int(abs(top - bottom) / 6))
            right = min(width, right + int(abs(top - bottom) / 6))
            top = max(0, top - int(abs(top - bottom) / 6))
        elif side == "bottom":
            left = max(0, left - int(abs(top - bottom) / 6))
            right = min(width, right + int(abs(top - bottom) / 6))
            bottom = min(height - 1, bottom + int(abs(top - bottom) / 6))

    return ((left, top), (right, top), (right, bottom), (left, bottom))


def classify_opening_by_count_and_arc(parallel_count, threshold, door_arc):
    # Werte ohne verwertbare Linie gehören immer zu something_else.
    if parallel_count < 2:
        return "unknown"

    # Kein eindeutiger Grenzwert bedeutet: exakt der Einzelwert wird
    # geometrisch geprüft und nur mit einem Bogen als Tür gewertet.
    if threshold is None:
        return "door" if door_arc else "unknown"

    # Kleiner als der Grenzwert gehört zur Türseite.
    if parallel_count < threshold and door_arc:
        return "door"

    # Größer als der Grenzwert ist die Fensterseite.
    if parallel_count > threshold:
        return "window"

    # Der genaue Grenzwert muss geometrisch geprüft werden.
    return "door" if door_arc else "window"


def detect_door_arc_geometry(
    binary_image,
    corners,
    gap_thickness,
    gap_orientation,
    ink_points=None,
    angle_bin_count=18,
    max_radius_ratio=1.2,
):
    """Detect a door arc and measure its radius and image-space bounds.

    Candidate radii are derived from ink distances to possible hinges instead
    of being estimated from the gap length. The best candidate must cover at
    least 55% of the quarter-circle's angular bins.
    """
    if binary_image is None or binary_image.ndim != 2 or len(corners) < 4:
        return None

    image_height, image_width = binary_image.shape
    gap_points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    left = max(0, int(np.floor(gap_points[:, 0].min())))
    right = min(image_width - 1, int(np.ceil(gap_points[:, 0].max())))
    top = max(0, int(np.floor(gap_points[:, 1].min())))
    bottom = min(image_height - 1, int(np.ceil(gap_points[:, 1].max())))
    gap_width = right - left + 1
    gap_height = bottom - top + 1

    if gap_orientation not in ("horizontal", "vertical"):
        return None
    opening_length = gap_width if gap_orientation == "horizontal" else gap_height
    if opening_length < max(6, 2 * max(1, int(gap_thickness))):
        return None

    if ink_points is None:
        ink_points = np.column_stack(np.nonzero(binary_image))[:, ::-1]
    ink_points = np.asarray(ink_points, dtype=np.float32).reshape(-1, 2)
    if len(ink_points) == 0:
        return None

    radius_tolerance = max(1, min(3, int(round(gap_thickness * 0.1))))
    minimum_radius = max(4.0, gap_thickness * 0.5)
    maximum_radius = max(opening_length * max_radius_ratio, gap_thickness * 2.0)
    mid_x = (left + right) / 2.0
    mid_y = (top + bottom) / 2.0
    if gap_orientation == "horizontal":
        hinges = [((left, mid_y), 1), ((right, mid_y), -1)]
        arc_sides = (("top", -1), ("bottom", 1))
    else:
        hinges = [((mid_x, top), 1), ((mid_x, bottom), -1)]
        arc_sides = (("left", -1), ("right", 1))

    best_geometry = None
    best_score = (0, 0, 0, 0)

    for (hinge_x, hinge_y), along_sign in hinges:
        delta_x = ink_points[:, 0] - hinge_x
        delta_y = ink_points[:, 1] - hinge_y
        for side, side_sign in arc_sides:
            if gap_orientation == "horizontal":
                along = delta_x * along_sign
                outward = delta_y * side_sign
            else:
                along = delta_y * along_sign
                outward = delta_x * side_sign

            in_quadrant = (along >= 0) & (outward >= 0)
            candidate_points = ink_points[in_quadrant]
            along = along[in_quadrant]
            outward = outward[in_quadrant]
            distances = np.hypot(along, outward)
            usable = (distances >= minimum_radius) & (distances <= maximum_radius)
            candidate_points = candidate_points[usable]
            along = along[usable]
            outward = outward[usable]
            distances = distances[usable]
            if len(distances) < angle_bin_count:
                continue

            angle_bins = np.clip(
                (np.arctan2(outward, along) / (np.pi / 2) * angle_bin_count)
                .astype(np.int32),
                0,
                angle_bin_count - 1,
            )
            radius_bins = np.rint(distances).astype(np.int32)
            accumulator = np.zeros(
                (int(radius_bins.max()) + 1, angle_bin_count), dtype=np.uint8
            )
            np.maximum.at(accumulator, (radius_bins, angle_bins), 1)
            radial_kernel = np.ones(
                (2 * radius_tolerance + 1, 1), dtype=np.uint8
            )
            radial_support = cv2.dilate(accumulator, radial_kernel)
            coverage_by_radius = np.count_nonzero(radial_support, axis=1)
            radius_bin = int(np.argmax(coverage_by_radius))
            coverage = int(coverage_by_radius[radius_bin])
            if coverage < int(np.ceil(angle_bin_count * 0.55)):
                continue

            supported_angles = radial_support[radius_bin] > 0
            run_edges = np.flatnonzero(
                np.diff(np.concatenate(([False], supported_angles, [False])).astype(np.int8))
            )
            run_lengths = run_edges[1::2] - run_edges[::2]
            if len(run_lengths) == 0:
                continue
            longest_run_index = int(np.argmax(run_lengths))
            longest_angular_run = int(run_lengths[longest_run_index])
            if longest_angular_run < int(np.ceil(angle_bin_count * 0.55)):
                continue

            arc_angle_mask = np.zeros(angle_bin_count, dtype=bool)
            arc_angle_mask[
                run_edges[2 * longest_run_index]:run_edges[2 * longest_run_index + 1]
            ] = True
            inliers = (
                (np.abs(distances - radius_bin) <= radius_tolerance)
                & arc_angle_mask[angle_bins]
            )
            arc_points = candidate_points[inliers]
            if len(arc_points) < angle_bin_count:
                continue

            score = (longest_angular_run, coverage, len(arc_points), -radius_bin)
            if score > best_score:
                best_score = score
                best_geometry = {
                    "side": side,
                    "hinge": (float(hinge_x), float(hinge_y)),
                    "radius": float(np.median(distances[inliers])),
                    "coverage": coverage / angle_bin_count,
                    "arc_points": arc_points,
                    "bounds": (
                        int(np.floor(arc_points[:, 0].min())),
                        int(np.floor(arc_points[:, 1].min())),
                        int(np.ceil(arc_points[:, 0].max())),
                        int(np.ceil(arc_points[:, 1].max())),
                    ),
                }

    return best_geometry


def has_quarter_circle_arc(binary_image, corners, gap_thickness, gap_orientation):
    """Return whether a quarter-circle door arc can be fitted."""
    return detect_door_arc_geometry(
        binary_image, corners, gap_thickness, gap_orientation
    ) is not None


# Für die Wände
def find_parallel_pairs(simplified_segments, wall_thickness):
    lines = []

    for segment_id, segment in enumerate(simplified_segments):
        for point_a, point_b in zip(segment[:-1], segment[1:]):
            x1, y1 = point_a
            x2, y2 = point_b
            dx, dy = abs(x2 - x1), abs(y2 - y1)

            if dx == 0 and dy == 0:
                continue
            if dy < dx:
                horizontal = True
                position = (y1 + y2) / 2
                start, end = sorted((x1, x2))
            else:
                horizontal = False
                position = (x1 + x2) / 2
                start, end = sorted((y1, y2))

            lines.append((segment_id, horizontal, position, start, end))

    pairs = []
    for i, line_a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            line_b = lines[j]
            if line_a[1] != line_b[1]:
                continue

            distance = abs(line_a[2] - line_b[2])
            overlap = min(line_a[4], line_b[4]) - max(line_a[3], line_b[3])

            if 0 < distance <= wall_thickness and overlap > 0:
                pairs.append((line_a, line_b))

    return pairs


# Für die Wandöffnungen

def find_parallel_groups(simplified_segments, wall_thickness):
    lines = []

    for segment_id, segment in enumerate(simplified_segments):
        for point_a, point_b in zip(segment[:-1], segment[1:]):
            x1, y1 = point_a
            x2, y2 = point_b
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx == 0 and dy == 0:
                continue

            if dy < dx:
                horizontal = True
                position = (y1 + y2) / 2
                start, end = sorted((x1, x2))
            else:
                horizontal = False
                position = (x1 + x2) / 2
                start, end = sorted((y1, y2))

            lines.append(
                (segment_id, horizontal, position, start, end)
            )

    groups = []
    position_tolerance = max(2, float(wall_thickness))

    for horizontal in (True, False):
        orientated_lines = sorted(
            (line for line in lines if line[1] == horizontal),
            key=lambda line: line[2],
        )

        clusters = []
        for line in orientated_lines:
            if not clusters:
                clusters.append([line])
                continue

            previous = clusters[-1]
            previous_position = previous[-1][2]

            if abs(line[2] - previous_position) <= position_tolerance:
                previous.append(line)
            else:
                clusters.append([line])

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            groups.append(tuple(cluster))

    return groups


# Filtert die Wandrechtecke basierend auf Fläschenüberdeckung mit einem Bild
def filter_wall_rectangles(rectangles, wall_image, min_support=0.5):
    accepted = []
    image_height, image_width = wall_image.shape[:2]

    for horizontal, corners in rectangles:
        polygon = np.asarray(
            (corners[0], corners[1], corners[3], corners[2]),
            dtype=np.int32,
        )
        left = max(0, int(polygon[:, 0].min()))
        right = min(image_width - 1, int(polygon[:, 0].max()))
        top = max(0, int(polygon[:, 1].min()))
        bottom = min(image_height - 1, int(polygon[:, 1].max()))
        if left > right or top > bottom:
            continue

        local_polygon = polygon - np.array((left, top), dtype=np.int32)
        local_mask = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
        cv2.fillConvexPoly(local_mask, local_polygon, 255)

        inside = local_mask > 0
        area = np.count_nonzero(inside)
        if area == 0:
            continue

        wall_roi = wall_image[top:bottom + 1, left:right + 1]
        support = np.count_nonzero(wall_roi[inside] > 0) / area

        if support >= min_support:
            accepted.append((horizontal, corners))

    return accepted

def expand_rectangles_and_add_gap_checkers(rectangles, margin, image_shape):
    height, width = image_shape
    expanded = []

    for horizontal, corners in rectangles:
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        left = max(0, min(xs) - margin)
        right = min(width - 1, max(xs) + margin)
        top = max(0, min(ys) - margin)
        bottom = min(height - 1, max(ys) + margin)

        expanded.append({
            "horizontal": horizontal,
            "start": left if horizontal else top,
            "end": right if horizontal else bottom,
            "centerline": (top + bottom) / 2 if horizontal else (left + right) / 2,
            "half_thickness": (
                (bottom - top) / 2 if horizontal else (right - left) / 2
            ),
            "corners": (
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            ),
        })

    if not expanded:
        return expanded

    band_tolerance = max(
        2.0,
        float(np.median([rectangle["half_thickness"] for rectangle in expanded]))
        * 0.5,
    )
    group_id = 0
    for horizontal in (True, False):
        aligned_rectangles = sorted(
            (
                rectangle
                for rectangle in expanded
                if rectangle["horizontal"] == horizontal
            ),
            key=lambda rectangle: rectangle["centerline"],
        )
        bands = []
        for rectangle in aligned_rectangles:
            if (
                not bands
                or rectangle["centerline"] - bands[-1][0] > band_tolerance
            ):
                bands.append([rectangle["centerline"], [rectangle]])
            else:
                bands[-1][1].append(rectangle)
                bands[-1][0] = float(
                    np.median(
                        [item["centerline"] for item in bands[-1][1]]
                    )
                )

        for _, band_rectangles in bands:
            for rectangle in band_rectangles:
                rectangle["group_id"] = group_id
            group_id += 1

    return expanded
