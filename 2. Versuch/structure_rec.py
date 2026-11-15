import cv2
import numpy as np


def analyze_structures(binary_image):
    """Extract wall rectangles, find openings, then classify their evidence."""
    wall_mask, wall_rectangles = extract_wall_candidates(binary_image)

    return {
        "wall_mask": wall_mask,
        "wall_rectangles": wall_rectangles,
    }


def extract_wall_candidates(binary_image):
    # Kontrolliert, ob binary_image ein valide Eingabe ist
    if binary_image is None or binary_image.ndim != 2 :
        raise ValueError("binary_image in extract_wall_candidates ist kein zweidimensionales Binärbild")

    height, width = binary_image.shape

    # Speichert die Distanz von den weißen Pixeln (Wert von 255) zum nächsten schwarzen Pixel (Wert von 0)
    distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    foreground_distances = distance[binary_image > 0]
    wall_mask = np.zeros_like(binary_image)

    # Wenn forgorund_distance keine Werte hat, gibt es keine Wände
    if foreground_distances.size == 0:
        return wall_mask, []

    # Otsu ist ein Verfahren zur automatischen Bestimmung eines Schwellenwertes,
    # eignet sich, wenn es wenn man Daten in 2 Kategorien unterteilen will
    # in unserem Fall, dünne und dicke Linien (Wand ist die dicke Linie, Rest ist die dünne Linie)
    # Otsu Erwartet 8- oder 16-Bit Int Werte in einem 2 Dimensionalen Array,
    # foreground_distance speichert Pixel in Int 32-Bit in einem 1 Dimensionalen Array
    # reshape(-1, 1): -1 = beliebig viele Zeilen, wie benötigt werden; 1 = es soll nur 1 Spalte geben 

    foreground_distances_u8 = np.clip(foreground_distances, 0, 255).reshape(-1, 1).astype(np.uint8)

    otsu_threshold, _ = cv2.threshold(
        foreground_distances_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Otsu kann dünnere Wände abschneiden, damit das nicht passiert, verkleinern wir den otsu_threshold
    # und sorgen dafür, dass es nicht kleiner als 1 Pixel werden kann

    core_threshold = max(1.0, float(otsu_threshold) * 0.9)
    core = np.where(distance >= core_threshold, 255, 0).astype(np.uint8)
    core_distance = distance[core > 0]

    if core_distance.size == 0:
        return wall_mask, []

    # HoughLinesP braucht ein threshold, ein minLineLength und ein maxLineGap, diese sollten sich an jedes Bild anpassen
    # sollen also nicht hardcoded sein, sondern adaptiv sein und sich an andere Bilder anpassen
    # 1. Idee: Bildskalierung nutzen -> schlecht, weil die Werte immernoch gehardcoded sind: scale * 0.025 = 2,5% * scale
    # 2. Idee: den median oder average von den Distanzen ausrechnen -> besser, aber es bleiben Pixelreste von den Möbeln

    wall_thickness = 2 * float(np.median(core_distance))
    min_line_length = max(1, int(round(wall_thickness)))
    max_line_gap = max(1, int(round(wall_thickness * 0.25)))
    hough_threshold = max(1, int(round(min_line_length * 0.5)))

    lines = cv2.HoughLinesP(core, rho=1, theta=np.pi / 180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None:
        return wall_mask, []

    # HoughLinesP ist nicht perfekt, es kreiert für einzelne Wand-Linien mehrere Linien
    # diese Werden gefiltert, erstmal in horizontale und vertikale Linien
    # um später Rechtecke basierend auf diesen Linien zu bilden

    raw_segments = []
    # HoughLinesP speichert Daten in einer 1 Dimensionalen List,
    # es gibt 4 Werte, jeweils 2 Paare die Eckpunkte bilden, (x1, y1) und (x2, y2)

    for x1, y1, x2, y2 in lines:
        dx, dy = int(x2 - x1), int(y2 - y1)

        # dabei kann x1 > x2 sein, also sollte man das erst sortieren
        horizontal = abs(dx) >= abs(dy)

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
                coords = np.aragne(start, end + 1, dtype=int)
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

            if end - start + 1 < 2 * half_thickness:
                continue

            if horizontal:
                corners = [
                    (int(start), int(coord - half_thickness)),
                    (int(end), int( coord - half_thickness)),
                    (int(start), int(coord + half_thickness)),
                    (int(end), int(coord + half_thickness))
                ]
            else:
                corners = [
                    int(coord - half_thickness), int(start),
                    int(coord - half_thickness), int(end),
                    int(coord + half_thickness), int(start),
                    int(coord + half_thickness), int(end)
                ]
            wall_rectangle.append(corners)
                
    return wall_mask, wall_rectangle