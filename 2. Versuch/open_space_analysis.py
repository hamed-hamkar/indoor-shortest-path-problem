import cv2
import numpy as np


def analyze_structures(binary_image):
    """Extract wall rectangles, find openings, then classify their evidence."""
    wall_mask, wall_rectangles = extract_wall_candidates(binary_image)
    openings = recognize_opening_in_wall(wall_rectangles)
    windows, doors, unknown_openings = recognize_window_and_door(binary_image, openings)
    furniture = recognize_furniture(binary_image, wall_mask, windows, doors, openings)

    return {
        "wall_mask": wall_mask,
        "wall_rectangles": wall_rectangles,
        "openings": openings,
        "windows": windows,
        "doors": doors,
        "unknown_openings": unknown_openings,
        "furniture": furniture,
    }


def extract_wall_candidates(binary_image):
    if binary_image is None or binary_image.ndim != 2:
        raise ValueError("binary_image muss ein zweidimensionales Binärbild sein")

    height, width = binary_image.shape
    distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    foreground_distances = distance[binary_image > 0]
    wall_mask = np.zeros_like(binary_image)
    if foreground_distances.size == 0:
        return wall_mask, []

    distance_u8 = np.clip(distance, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        distance_u8[binary_image > 0].reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # Otsu may place the cutoff above the cores of thinner walls. Lower it
    # proportionally so those cores remain available to Hough; keep the Otsu
    # result as the image-specific reference instead of a fixed pixel value.
    core_threshold = max(1.0, float(otsu_threshold) * 0.9)
    core = np.where(distance >= core_threshold, 255, 0).astype(np.uint8)

    # Scale line detection to the drawing size. The thickness still comes from
    # the distance map below; this length only filters short strokes.
    scale = min(height, width)
    min_line_length = max(1, int(round(scale * 0.025)))
    lines = cv2.HoughLinesP(
        core, 1, np.pi / 180,
        threshold=max(1, int(round(min_line_length * 0.35))),
        minLineLength=min_line_length,
        maxLineGap=max(1, int(round(scale * 0.01))),
    )
    if lines is None:
        return wall_mask, []

    raw_segments = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = int(x2 - x1), int(y2 - y1)
        horizontal = abs(dx) >= abs(dy)
        if horizontal:
            coord, start, end = (y1 + y2) / 2, min(x1, x2), max(x1, x2)
        else:
            coord, start, end = (x1 + x2) / 2, min(y1, y2), max(y1, y2)
        raw_segments.append((horizontal, coord, int(start), int(end)))

    # Cluster duplicate detections by perpendicular position. Keep a group id
    # so opening detection can compare only intervals on the same wall axis.
    # Hough can return neighboring centerlines for one vertical wall near
    # junctions. Cluster within roughly one core radius more than before;
    # separate parallel walls remain apart by their measured wall thickness.
    pos_tol = max(core_threshold * 2.0, scale * 0.002)
    grouped = []
    group_id = 0
    for horizontal in (True, False):
        ordered = sorted((s for s in raw_segments if s[0] == horizontal), key=lambda s: s[1])
        bands = []
        for segment in ordered:
            if not bands or segment[1] - bands[-1][0] > pos_tol:
                bands.append([segment[1], [segment]])
            else:
                bands[-1][1].append(segment)
                bands[-1][0] = float(np.median([item[1] for item in bands[-1][1]]))

        for coord, band in bands:
            intervals = []
            for start, end in sorted((s[2], s[3]) for s in band):
                # Merge overlapping or directly adjacent detections only.
                # A real gap must remain visible for opening recognition.
                if intervals and start <= intervals[-1][1] + 1:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
                else:
                    intervals.append((start, end))
            grouped.extend((horizontal, group_id, int(round(coord)), a, b) for a, b in intervals)
            group_id += 1

    wall_rectangles = []
    for horizontal, band_id, coord, start, end in grouped:
        if horizontal:
            coords = np.arange(start, end + 1, dtype=int)
            xs = np.clip(coords, 0, width - 1)
            ys = np.full(xs.shape, np.clip(coord, 0, height - 1), dtype=int)
        else:
            coords = np.arange(start, end + 1, dtype=int)
            ys = np.clip(coords, 0, height - 1)
            xs = np.full(ys.shape, np.clip(coord, 0, width - 1), dtype=int)

        radii = distance[ys, xs]
        radii = radii[radii > 0]
        if radii.size == 0:
            continue
        half_thickness = max(1, int(round(float(np.median(radii)) + core_threshold)))
        # Text fragments and small furniture strokes are commonly shorter
        # than the local wall thickness; they are not useful wall rectangles.
        if end - start + 1 < 2 * half_thickness:
            continue

        if horizontal:
            corners = [
                (int(start), int(coord - half_thickness)),
                (int(end), int(coord - half_thickness)),
                (int(end), int(coord + half_thickness)),
                (int(start), int(coord + half_thickness)),
            ]
        else:
            corners = [
                (int(coord - half_thickness), int(start)),
                (int(coord + half_thickness), int(start)),
                (int(coord + half_thickness), int(end)),
                (int(coord - half_thickness), int(end)),
            ]

        wall_rectangles.append({
            "orientation": "horizontal" if horizontal else "vertical",
            "group_id": band_id,
            "start": int(start),
            "end": int(end),
            "centerline": int(coord),
            "half_thickness": int(half_thickness),
            "corners": corners,
        })

    # A wall band should connect to at least one perpendicular wall band.
    # This rejects many text/furniture strokes that otherwise create fake gaps.
    supported_group_ids = {
        rectangle["group_id"]
        for rectangle in wall_rectangles
        if _has_perpendicular_support(rectangle, wall_rectangles)
    }
    wall_rectangles = [
        rectangle for rectangle in wall_rectangles
        if rectangle["group_id"] in supported_group_ids
    ]
    for rectangle in wall_rectangles:
        cv2.fillConvexPoly(
            wall_mask,
            np.asarray(rectangle["corners"], dtype=np.int32),
            255,
        )

    # Do not area-filter the mask here: an area-only filter can delete short
    # real walls and would make the rectangle list disagree with the mask.
    return wall_mask, wall_rectangles


def _has_perpendicular_support(rectangle, all_rectangles):
    for other in all_rectangles:
        if other["orientation"] == rectangle["orientation"]:
            continue
        if rectangle["orientation"] == "horizontal":
            horizontal, vertical = rectangle, other
        else:
            horizontal, vertical = other, rectangle

        x = vertical["centerline"]
        y = horizontal["centerline"]
        horizontal_pad = vertical["half_thickness"]
        vertical_pad = horizontal["half_thickness"]
        if (
            horizontal["start"] - horizontal_pad <= x <= horizontal["end"] + horizontal_pad
            and vertical["start"] - vertical_pad <= y <= vertical["end"] + vertical_pad
        ):
            return True
    return False


def recognize_opening_in_wall(wall_rectangles):
    """Find gaps between consecutive wall rectangles in the same aligned band.

    No edge detection or second HoughLinesP pass is needed.
    """
    bands = {}
    for rectangle in wall_rectangles:
        bands.setdefault(rectangle["group_id"], []).append(rectangle)

    openings = []
    for rectangles in bands.values():
        rectangles.sort(key=lambda item: item["start"])
        for first, second in zip(rectangles, rectangles[1:]):
            local_wall_thickness = first["half_thickness"] + second["half_thickness"]
            if min(first["end"] - first["start"] + 1,
                   second["end"] - second["start"] + 1) < local_wall_thickness:
                continue
            gap_start = first["end"] + 1
            gap_end = second["start"] - 1
            if gap_start > gap_end:
                continue

            half_thickness = int(round(local_wall_thickness / 2))
            coord = int(round((first["centerline"] + second["centerline"]) / 2))
            horizontal = first["orientation"] == "horizontal"
            if horizontal:
                corners = [
                    (gap_start, coord - half_thickness),
                    (gap_end, coord - half_thickness),
                    (gap_end, coord + half_thickness),
                    (gap_start, coord + half_thickness),
                ]
            else:
                corners = [
                    (coord - half_thickness, gap_start),
                    (coord + half_thickness, gap_start),
                    (coord + half_thickness, gap_end),
                    (coord - half_thickness, gap_end),
                ]

            openings.append({
                "orientation": first["orientation"],
                "group_id": first["group_id"],
                "start": int(gap_start),
                "end": int(gap_end),
                "centerline": coord,
                "half_thickness": half_thickness,
                "width": int(gap_end - gap_start + 1),
                "corners": [(int(x), int(y)) for x, y in corners],
            })
    return openings


def recognize_window_and_door(binary_image, openings):
    """Classify only when the raster contains positive symbol evidence.

    A continuous line near the middle of the wall band is window evidence.
    A door needs a quarter-circle arc in the room area beside the gap. Anything
    without either cue remains unknown instead of being guessed as a door.
    """
    windows, doors, unknown = [], [], []
    ink_distance = cv2.distanceTransform(cv2.bitwise_not(binary_image), cv2.DIST_L2, 5)

    for opening in openings:
        window_evidence = _has_window_center_line(binary_image, opening)
        door_score = _door_swing_score(ink_distance, opening)
        door_evidence = door_score >= 0.55
        opening["window_line_evidence"] = window_evidence
        opening["door_arc_score"] = door_score

        # A long stroke can be a window divider or a door threshold/leaf, and
        # nearby drawing details can imitate a short arc. Conflicting evidence
        # is kept for review instead of forcing a label.
        if window_evidence and door_evidence:
            opening["classification"] = "ambiguous"
            unknown.append(opening)
        elif door_evidence:
            opening["classification"] = "door"
            doors.append(_gap_endpoints(opening))
        elif window_evidence:
            opening["classification"] = "window"
            windows.append(_gap_endpoints(opening))
        else:
            opening["classification"] = "unknown"
            unknown.append(opening)
    return windows, doors, unknown


def _has_window_center_line(binary_image, opening):
    x_values = [point[0] for point in opening["corners"]]
    y_values = [point[1] for point in opening["corners"]]
    x0, x1 = max(0, min(x_values)), min(binary_image.shape[1], max(x_values) + 1)
    y0, y1 = max(0, min(y_values)), min(binary_image.shape[0], max(y_values) + 1)
    patch = binary_image[y0:y1, x0:x1]
    if patch.size == 0 or opening["width"] <= 0:
        return False

    # For a horizontal wall, a window stroke runs along x; for a vertical wall
    # it runs along y. Require support across most of the opening length.
    if opening["orientation"] == "horizontal":
        support = np.count_nonzero(patch, axis=1)
        middle = (patch.shape[0] - 1) / 2
    else:
        support = np.count_nonzero(patch, axis=0)
        middle = (patch.shape[1] - 1) / 2
    required_support = 0.6 * opening["width"]
    center_tolerance = max(1, opening["half_thickness"] * 0.5)
    return any(
        count >= required_support and abs(index - middle) <= center_tolerance
        for index, count in enumerate(support)
    )


def _door_swing_score(ink_distance, opening):
    nominal_radius = float(opening["width"])
    if nominal_radius < 3:
        return 0.0

    half = float(opening["half_thickness"])
    # Wall thickness is not the symbol stroke width. A wide tolerance made
    # nearby letters and furniture look like door arcs.
    tolerance = max(1.5, min(2.5, half * 0.25))
    horizontal = opening["orientation"] == "horizontal"
    best_score = 0.0

    # Printed door arcs are not always exactly the same radius as the wall gap,
    # and the detected jamb can be shifted by a few pixels. Search relative to
    # the measured opening and wall thickness rather than fixed pixel offsets.
    for radius in np.linspace(nominal_radius * 0.8, nominal_radius * 1.2, 9):
        sample_count = max(8, int(round(radius)))
        angles = np.linspace(0, np.pi / 2, sample_count)
        for hinge in (opening["start"], opening["end"]):
            along_sign = 1 if hinge == opening["start"] else -1
            for hinge_shift in (-half, 0, half):
                for room_sign in (-1, 1):
                    if horizontal:
                        xs = hinge + hinge_shift + along_sign * radius * np.sin(angles)
                        ys = opening["centerline"] + room_sign * radius * np.cos(angles)
                    else:
                        xs = opening["centerline"] + room_sign * radius * np.cos(angles)
                        ys = hinge + hinge_shift + along_sign * radius * np.sin(angles)

                    xi = np.rint(xs).astype(int)
                    yi = np.rint(ys).astype(int)
                    valid = (
                        (xi >= 0) & (xi < ink_distance.shape[1]) &
                        (yi >= 0) & (yi < ink_distance.shape[0])
                    )
                    if not np.any(valid):
                        continue
                    score = float(np.mean(ink_distance[yi[valid], xi[valid]] <= tolerance))
                    best_score = max(best_score, score)

    return best_score


def _gap_endpoints(opening):
    if opening["orientation"] == "horizontal":
        y = opening["centerline"]
        return (opening["start"], y), (opening["end"], y)
    x = opening["centerline"]
    return (x, opening["start"]), (x, opening["end"])


def recognize_furniture(binary_image, wall_image, windows_pos, doors_pos, openings=None):
    exclusion_mask = np.zeros_like(binary_image)
    exclusion_mask[wall_image > 0] = 255
    for gap in windows_pos + doors_pos:
        (x1, y1), (x2, y2) = gap
        cv2.rectangle(
            exclusion_mask,
            (min(x1, x2), min(y1, y2)),
            (max(x1, x2), max(y1, y2)),
            255,
            -1,
        )
    for opening in openings or []:
        cv2.fillConvexPoly(
            exclusion_mask,
            np.asarray(opening["corners"], dtype=np.int32),
            255,
        )
    return cv2.bitwise_and(binary_image, binary_image, mask=cv2.bitwise_not(exclusion_mask))


import cv2
import numpy as np


def analyze_structures(binary_image):
    """Extract wall rectangles, find openings, then classify their evidence."""
    (
        wall_rectangles,
        otsu_image,
        core_image,
        horizontal_hough_overlay,
        vertical_hough_overlay,
    ) = extract_wall_candidates(binary_image)

    return {
        "wall_rectangles": wall_rectangles,
        "otsu_image": otsu_image,
        "core_image": core_image,
        "horizontal_hough_overlay": horizontal_hough_overlay,
        "vertical_hough_overlay": vertical_hough_overlay,
    }


def extract_wall_candidates(binary_image):
    # Kontrolliert, ob binary_image ein valide Eingabe ist
    if binary_image is None or binary_image.ndim != 2 :
        raise ValueError("binary_image in extract_wall_candidates ist kein zweidimensionales Binärbild")

    height, width = binary_image.shape

    # Speichert die Distanz von den weißen Pixeln (Wert von 255) zum nächsten schwarzen Pixel (Wert von 0)
    distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    foreground_distances = distance[binary_image > 0]
    otsu_image = np.zeros_like(binary_image)
    core = np.zeros_like(binary_image)
    horizontal_hough_overlay = cv2.cvtColor(core, cv2.COLOR_GRAY2BGR)
    vertical_hough_overlay = cv2.cvtColor(core, cv2.COLOR_GRAY2BGR)

    # Wenn forgorund_distance keine Werte hat, gibt es keine Wände
    if foreground_distances.size == 0:
        return [], otsu_image, core, horizontal_hough_overlay, vertical_hough_overlay

    # Otsu ist ein Verfahren zur automatischen Bestimmung eines Schwellenwertes,
    # eignet sich, wenn es wenn man Daten in 2 Kategorien unterteilen will
    # in unserem Fall, dünne und dicke Linien (Wand ist die dicke Linie, Rest ist die dünne Linie)
    # Otsu Erwartet 8- oder 16-Bit Int Werte in einem 2 Dimensionalen Array,
    # foreground_distance speichert Pixel in Int 32-Bit in einem 1 Dimensionalen Array
    # reshape(-1, 1): -1 = beliebig viele Zeilen, wie benötigt werden; 1 = es soll nur 1 Spalte geben 

    foreground_distances_u8 = np.clip(foreground_distances, 0, 255).reshape(-1, 1).astype(np.uint8)

    otsu_threshold, otsu_image = cv2.threshold(
        foreground_distances_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # test
    spatial_otsu_image = np.zeros_like(binary_image)
    spatial_otsu_image[binary_image > 0] = otsu_image.ravel()

    # Otsu kann dünnere Wände abschneiden, damit das nicht passiert, verkleinern wir den otsu_threshold
    # und sorgen dafür, dass es nicht kleiner als 1 Pixel werden kann

    core_threshold = max(1.0, float(otsu_threshold) * 0.9)
    core = np.where(distance >= core_threshold, 255, 0).astype(np.uint8)
    horizontal_hough_overlay = cv2.cvtColor(core, cv2.COLOR_GRAY2BGR)
    vertical_hough_overlay = cv2.cvtColor(core, cv2.COLOR_GRAY2BGR)
    core_distance = distance[core > 0]

    if core_distance.size == 0:
        return [], spatial_otsu_image, core, horizontal_hough_overlay, vertical_hough_overlay

    # HoughLinesP braucht ein threshold, ein minLineLength und ein maxLineGap, diese sollten sich an jedes Bild anpassen
    # sollen also nicht hardcoded sein, sondern adaptiv sein und sich an andere Bilder anpassen
    # 1. Idee: Bildskalierung nutzen -> schlecht, weil die Werte immernoch gehardcoded sind: scale * 0.025 = 2,5% * scale
    # 2. Idee: den median oder average von den Distanzen ausrechnen -> besser, aber es bleiben Pixelreste von den Möbeln

    wall_thickness = 2 * float(np.median(core_distance))
    min_line_length = max(1, int(round(wall_thickness)))
    max_line_gap = max(1, int(round(wall_thickness * 0.25)))
    hough_threshold = max(1, int(round(min_line_length * 0.4)))

    # !!!
    # !!!
    # !!!
    # !!!
    # HoughLinesP funktioniert nicht gut mit der erkennnung von den kleinen Wänden,
    # nutze Edges, cv2.Canny(), cv2.findContours, cv2.approxPolyDP oder ed = cv2.ximgprox.createEdgeDrawing()
    # und dann ed.detectEdges() und dann ed.getSegments() und dann basierend auf den Linien-Punkten Rechtecke bilden
    # !!!
    # !!!
    # !!!
    # !!!

    # lines = cv2.HoughLinesP(core, rho=1, theta=np.pi / 180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    outlines = cv2.ximgproc.createEdgeDrawing()     # Rand detektor erstellen

    outlines.detectEdges(core)                      # Kanten im Bild erkennen

    edges = outlines.getEdgeImagage()               # erkannte Kantenbild aufrufen

    segments_outlines = outlines.getSegments()      # Kanten als Punktketten abrufen

    if outlines is None:
        return [], spatial_otsu_image, core, horizontal_hough_overlay, vertical_hough_overlay

    # HoughLinesP ist nicht perfekt, es kreiert für einzelne Wand-Linien mehrere Linien
    # diese Werden gefiltert, erstmal in horizontale und vertikale Linien
    # um später Rechtecke basierend auf diesen Linien zu bilden

    raw_segments = []
    # HoughLinesP speichert Daten in einer 1 Dimensionalen List,
    # es gibt 4 Werte, jeweils 2 Paare die Eckpunkte bilden, (x1, y1) und (x2, y2)

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = int(x2 - x1), int(y2 - y1)

        horizontal = abs(dx) >= abs(dy)
        line_overlay = horizontal_hough_overlay if horizontal else vertical_hough_overlay
        cv2.line(
            line_overlay,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            1,
        )

        # Daten so speicher, dass man aus denen ein Rechteck bilden kann
        if horizontal:
            coord, start, end = (y1 + y2) / 2, min(x1, x2), max(x1, x2)

        else:
            coord, start, end = (x1 + x2) / 2, min(y1, y2), max(y1, y2)
        
        raw_segments.append((horizontal, coord, int(start), int(end)))

    # Gruppiere doppelte Ereknnungen anhand ihrer Ausrichtung, Behalte Gruppen-ID, damit
    # Öffnungserkennung später nur Linienabschnitte vergleichen kann, die zur gleichen Achse gehören
    # Hough kann Wandverbindungen von benachbarten Mittellinien die selbe Wand_linie erkennen
    # Gruppiere diese Linien innerhalb eines etwas größeren Abstand als zuvor
    # Getrennte parallele Wände sollen weiterhin voneinander getrennt bleiben
    # In kurz, gruppiere die von Hough erkannten Liniensegments in Abschnitte, damit doppelte
    # oder leicht versetzte Treffer später nicht als viele verschiedene Wände behandelt werden

    # Toleranz fur die Position, also der maximale Abstand bei denen wwei parallele Segmente
    # noch als selbe Wandspur gelten
    pos_tol = core_threshold * 2.0

    # sortiere Lines in horizontale und vertikale(nicht horizontale) Linien
    ordered_segments = []

    for horizontal in (True, False):
        ordered = sorted((s for s in raw_segments if s[0] == horizontal), key=lambda s: s[1])
        ordered_segments.append((horizontal, ordered))
        # speichere Segmente basierend auf Position der y-Achse in eine Liste
        # dabei soll y-Achse sich anpassen
        grouped = []
        group_id = 0
        
        for horizontal, ordered in ordered_segments:
            # Wenn bands leer ist oder Segmente nicht nah genug sind,
            # kreiere neues Segment mit der Position
            bands = []


            for segment in ordered:
                # wenn bands leer ist oder der positions Wert zu weit entfernt ist, kreiere neues bands-segment
                if not bands or segment[1] - bands[-1][0] > pos_tol:
                    bands.append([segment[1], [segment]])
                
                # falls Position nah genaug ist, verbinde es mit existierendem Segmenten und berechne
                # die Position neu, da ordered soritert ist, muss es nicht durch alle Elemente gehen,
                # sondern nur das letzte Element in bands ansprechen, positions Wert muss sich deswegen anpassen,
                # weil sonst spätere Linien nicht mehr angesprochen werden, die eigentlich dazu passen, nutze median
                else:
                    bands[-1][1].append(segment)
                    # berechne median, damit es sich anpasst
                    bands[-1][0] = float(np.median([item[1] for item in bands[-1][1]]))

            for coord, band in bands:
                intervals = []
                # start und end Punkte müssen sortiert werden, da Beispielsweise für bands:
                # (True, 25, 19, 20)
                # (True, 25, 18, 30)
                # (True, 25, 10, 15)
                # kann es zu Probleme mit diesen Werten kommen, deswegen erst sortieren

                for start, end in sorted([(s[2], s[3]) for s in band]):
                    # Verbinde überlappende oder nahe Linien
                    # Echte gaps müssen trotzdem bleiben

                    # start <= intervals[-1][1] + toleranz, also start <= vorheriuges Ende + toleranz --> keine Lücke
                    if intervals and start <= intervals[-1][1] + pos_tol:
                        intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
                    # start > intervals[-1][1] + toleranz, also start > vorheriuges Ende + toleranz --> eine Lücke, neuer Eintrag
                    else: 
                        intervals.append((start, end))
                grouped.extend((horizontal, group_id, int(round(coord)), a, b) for a, b in intervals)
                group_id += 1

        # grouped kann so aussehen:
        # (True,  0,  25, 10, 30),
        # (True,  0,  25, 40, 50),
        # (True,  1, 100,  5, 45),
        # (False, 2,  70, 12, 60),
        # (False, 2,  70, 75, 90)
        # selbe group_id, bedeutet, die gleichen y-Wert/x-Wert, fals horizontal/vertikal
        # kann man aber ignorieren, muss einfach nur darüber iterieren und Rechtecke bilden, dicke muss ausgerechnet werden
        wall_rectangle = []
        for horizontal, band_id, coord, start, end in grouped:
            # breite der Rechtecke berechnen
            if horizontal:
                coords = np.arange(start, end + 1, dtype=int)
                xs = np.clip(coords, 0, width - 1)
                ys = np.full(xs.shape, np.clip(coord, 0, height -1), dtype = int)
            else:
                coords = np.arange(start, end + 1, dtype=int)
                ys = np.clip(coords, 0, height - 1)
                xs = np.full(ys.shape, np.clip(coord, 0, width - 1), dtype=int)

            rec_distance = distance[ys, xs]
            rec_distance = rec_distance[rec_distance > 0]
            if rec_distance.size == 0:
                continue
            half_thickness = max(1, int(round(float(np.mean(rec_distance)) + core_threshold)))

            if end - start < 2 * half_thickness:
                continue

            if horizontal:
                corners = [
                    (int(start), int(coord - half_thickness)),
                    (int(end), int( coord - half_thickness)),
                    (int(end), int(coord + half_thickness)),
                    (int(start), int(coord + half_thickness)),
                ]
            else:
                corners = [
                    (int(coord - half_thickness), int(start)),
                    (int(coord + half_thickness), int(start)),
                    (int(coord + half_thickness), int(end)),
                    (int(coord - half_thickness), int(end)),
                ]

            wall_rectangle.append(corners)
                
    return (
        wall_rectangle,
        spatial_otsu_image,
        core,
        horizontal_hough_overlay,
        vertical_hough_overlay,
    )
