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
    (
        window_pos,
        door_pos,
        rest_gap_pos,
        parallel_group_counts,
        door_window_threshold,
        classified_openings,
    ) = window_or_door(gap_line)

    return (
        wall_rectangles,
        window_pos,
        door_pos,
        rest_gap_pos,
        gap_line,
        parallel_group_counts,
        door_window_threshold,
        classified_openings,
    )


def extract_wall_candidates(binary_image):
    if binary_image is None or binary_image.ndim != 2:
        raise ValueError(
            "binary_image in extract_wall_candidates ist kein zweidimensionales Binärbild"
        )

    distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    foreground_distances = distance[binary_image > 0]
    otsu_image = np.zeros_like(binary_image)
    core = np.zeros_like(binary_image)

    # Wenn forgorund_distance keine Werte hat, gibt es keine Wände
    if foreground_distances.size == 0:
        return []

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

    edge_image = np.zeros_like(core)
    segments = []
    if np.any(core):
        outlines = cv2.ximgproc.createEdgeDrawing()
        outlines.detectEdges(core)
        edge_image = outlines.getEdgeImage()
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
        rectangles, margin=2, image_shape=binary_image.shape
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
    gap_mask = np.zeros_like(binary_image)
    opening_regions = []
    if original_image is None:
        original_gray = binary_image
    elif original_image.ndim == 3:
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_image

    for opening_id, opening in enumerate(wall_openings):
        polygon = np.asarray(opening, dtype=np.int32)
        cv2.fillConvexPoly(gap_mask, polygon, 255)

        xs = polygon[:, 0]
        ys = polygon[:, 1]
        opening_regions.append({
            "opening_id": opening_id,
            "corners": opening,
            "bounds": (xs.min(), ys.min(), xs.max(), ys.max()),
        })

    gap_image = cv2.bitwise_and(binary_image, gap_mask)

    results = []

    for region in opening_regions:
        left, top, right, bottom = region["bounds"]
        gap_roi = gap_image[top:bottom + 1, left:right + 1]
        gap_width = right - left + 1
        gap_height = bottom - top + 1
        gap_orientation = (
            "horizontal" if gap_width >= gap_height else "vertical"
        )
        wall_thickness = (
            gap_height if gap_orientation == "horizontal" else gap_width
        )
        wall_thickness = max(1, int(round(wall_thickness)))
        door_arc = has_quarter_circle_arc(
            binary_image,
            region["corners"],
            wall_thickness,
        )

        if not np.any(gap_roi):
            parallel_groups = []
            gap_segments = []
        else:
            if original_image is None:
                gap_source = gap_roi
            elif original_image.ndim == 3:
                original_gray = cv2.cvtColor(
                    original_image,
                    cv2.COLOR_BGR2GRAY,
                )
                gap_source = original_gray[
                    top:bottom + 1,
                    left:right + 1,
                ]
            else:
                gap_source = original_image[
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
            "door_arc": door_arc,
            "gap_segments": gap_segments,
        })

    return results


def has_quarter_circle_arc(binary_image, corners, gap_thickness):
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    gap_width = max(xs) - min(xs)
    gap_height = max(ys) - min(ys)

    margin = max(12, int(round(max(gap_width, gap_height) * 0.75)))
    height, width = binary_image.shape
    left = max(0, min(xs) - margin)
    right = min(width - 1, max(xs) + margin)
    top = max(0, min(ys) - margin)
    bottom = min(height - 1, max(ys) + margin)

    roi = binary_image[top:bottom + 1, left:right + 1]
    if roi.size == 0:
        return False

    edges = cv2.Canny(roi, 50, 150)
    min_radius = max(4, int(round(min(gap_width, gap_height) * 0.15)))
    max_radius = max(
        min_radius + 1,
        int(round(max(gap_width, gap_height) * 1.25)),
    )

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, gap_thickness),
        param1=100,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return False

    absolute_corners = [(int(x), int(y)) for x, y in corners]
    for cx, cy, radius in np.round(circles[0]).astype(int):
        circle_center = (left + int(cx), top + int(cy))

        touches_gap_corner = any(
            abs(circle_center[0] - corner_x) <= radius * 0.45
            and abs(circle_center[1] - corner_y) <= radius * 0.45
            for corner_x, corner_y in absolute_corners
        )
        if not touches_gap_corner:
            continue

        quadrant_counts = [0, 0, 0, 0]
        sample_count = 48
        for index in range(sample_count):
            angle = 2 * np.pi * index / sample_count
            sx = int(round(cx + radius * np.cos(angle)))
            sy = int(round(cy + radius * np.sin(angle)))

            if sy < 0 or sy >= edges.shape[0] or sx < 0 or sx >= edges.shape[1]:
                continue

            if edges[sy, sx] == 0:
                continue

            if sx >= cx and sy < cy:
                quadrant = 0
            elif sx < cx and sy < cy:
                quadrant = 1
            elif sx < cx and sy >= cy:
                quadrant = 2
            else:
                quadrant = 3
            quadrant_counts[quadrant] += 1

        occupied_quadrants = sum(count >= 3 for count in quadrant_counts)
        strongest_quadrant = max(quadrant_counts)
        if occupied_quadrants <= 2 and strongest_quadrant >= 5:
            return True

    return False


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


def classify_opening_by_count(parallel_count, threshold, door_arc):
    # Werte ohne verwertbare Linie gehören immer zu something_else.
    if parallel_count < 2:
        return "unknown"

    # Kein eindeutiger Grenzwert bedeutet: exakt der Einzelwert wird
    # geometrisch geprüft und nur mit einem Bogen als Tür gewertet.
    if threshold is None:
        return "door" if door_arc else "unknown"

    # Kleiner als der Grenzwert gehört zur Türseite.
    if parallel_count < threshold:
        return "door"

    # Größer als der Grenzwert ist die Fensterseite.
    if parallel_count > threshold:
        return "window"

    # Der genaue Grenzwert muss geometrisch geprüft werden.
    return "door" if door_arc else "window"


def window_or_door(openings_wall):
    window = []
    door = []
    something_else = []
    parallel_group_lengths = []

    for opening in openings_wall:
        parallel_groups = opening["parallel_groups"]
        parallel_count = max(
            (len(group) for group in parallel_groups),
            default=0,
        )
        parallel_group_lengths.append(parallel_count)

    threshold = classify_parallel_threshold(parallel_group_lengths)

    classified_openings = []

    for opening in openings_wall:
        corners = opening["corners"]
        parallel_groups = opening["parallel_groups"]
        door_arc = opening.get("door_arc", False)
        opening_id = opening["opening_id"]

        parallel_count = max(
            (len(group) for group in parallel_groups),
            default=0,
        )
        classification = classify_opening_by_count(
            parallel_count,
            threshold,
            door_arc,
        )

        if classification == "door":
            door.append(corners)
        elif classification == "window":
            window.append(corners)
        else:
            something_else.append(corners)

        classified_openings.append({
            "opening_id": opening_id,
            "corners": corners,
            "parallel_count": parallel_count,
            "door_arc": door_arc,
            "classification": classification,
        })

    return (
        window,
        door,
        something_else,
        parallel_group_lengths,
        threshold,
        classified_openings,
    )


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

    for horizontal, corners in rectangles:
        rectangel_mask = np.zeros_like(wall_image)
        polygon = np.asarray(
            (corners[0], corners[1], corners[3], corners[2]),
            dtype=np.int32,
        )
        cv2.fillConvexPoly(rectangel_mask, polygon, 255)

        inside = rectangel_mask > 0
        area = np.count_nonzero(inside)
        if area == 0:
            continue

        support = np.count_nonzero(wall_image[inside] > 0) / area

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