import cv2

def mouse_callback(event, x, y, flags, param):

    grid = param["grid"]
    tile_size = param["tile_size"]

    # Pixelkoordinaten in Grid-Koordinaten umrechnen
    col = x // tile_size
    row = y // tile_size

    rows, cols = grid.shape

    if 0 <= row < rows and 0 <= col < cols:
        #Nur wenn die STRG-Taste gedrückt wird, die Kachel ändern
        if flags  & cv2.EVENT_FLAG_CTRLKEY:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Strg + Linksklick: Tile als begehbar markieren
                grid[row, col] = 0
                print(f"Kachel [{row}, {col}] = Begehbar")
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Strg + Rechtsklick: Tile als Wand markieren
                grid[row, col] = 1
                print(f"Kachel [{row}, {col}] = Wand")

    else:
        print("Klick außerhalb des Grids")