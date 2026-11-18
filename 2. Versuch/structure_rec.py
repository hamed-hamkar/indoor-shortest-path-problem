import cv2
import numpy as np
from itertools import combinations


def analyze_structures(binary_image):

    return extract_wall_candidates(binary_image)


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
        return {
            "wall_rectangles": [],
            "otsu_image": otsu_image,
            "core_image": core,
            "edge_image": core.copy(),
            "segments": [],
            "simplified_segments": [],
        }

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

    return {
        "wall_rectangles": rectangles,
        "otsu_image": otsu_image,
        "core_image": core,
        "edge_image": edge_image,
        "segments": segments,
        "simplified_segments": simplified_segments,
    }


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


def wall_gap_interpreter(wall_openings, binary_image):
    gap_mask = np.zeros_like(binary_image)
    opening_regions = []

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
        wall_thickness = max(1, int(round(min(right - left, bottom - top))))

        if not np.any(gap_roi):
            parallel_groups = []
        else:
            outlines = cv2.ximgproc.createEdgeDrawing()
            outlines.detectEdges(gap_roi)
            segments = outlines.getSegments() or []
            simplified_segments = segments_helper(segments)
            parallel_groups = find_parallel_groups(
                simplified_segments, wall_thickness=wall_thickness
            )

        results.append({
            "opening_id": region["opening_id"],
            "corners": region["corners"],
            "wall_thickness": wall_thickness,
            "parallel_groups": parallel_groups,
        })

    return results


def window_or_door(openings):
    window = []
    door = []
    something_else = []

    return


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
    pairs = find_parallel_pairs(simplified_segments, wall_thickness)

    pair_links = {frozenset(pair) for pair in pairs}
    lines = {line for pair in pairs for line in pair}

    groups = []
    used_lines = set()

    # Erst Dreiergruppen suchen, damit sie Vorrang vor einzelnen Paaren haben.
    triplets = []
    for combination in combinations(lines, 3):
        ordered = tuple(sorted(combination, key=lambda line: line[2]))
        first, middle, last = ordered

        if (
            first[1] == middle[1] == last[1]
            and frozenset((first, middle)) in pair_links
            and frozenset((middle, last)) in pair_links
        ):
            triplets.append(ordered)

    for triplet in triplets:
        if not any(line in used_lines for line in triplet):
            groups.append(triplet)
            used_lines.update(triplet)

    # Übrig gebliebene erkannte Paare als Zweier-Tupel ergänzen.
    for pair in pairs:
        if not any(line in used_lines for line in pair):
            groups.append(pair)
            used_lines.update(pair)

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