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

    # HoughLinesP braucht ein threshold, ein minLineLength und ein maxLineGap, diese sollten sich an jedes Bild anpassen
    # sollen also nicht hardcoded sein, sondern adaptiv sein und sich an andere Bilder anpassen
    # 1. Idee: Bildskalierung nutzen -> schlecht, weil die Werte immernoch gehardcoded sind: scale * 0.025 = 2,5% * scale
    # 2. Idee: den median oder average von den Distanzen ausrechnen -> besser, aber es bleiben Pixelreste von den Möbeln

    lines = cv2.HoughLinesP(core, rho=1, theta=np.pi / 180, threshold= core_threshold, minLineLength=)

    



    return