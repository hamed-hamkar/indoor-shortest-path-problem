import cv2


def draw_grid(image, grid, tile_size, start=None, goal=None, visited=None, forward_path=None):
    if visited is None:
        visited = set()

    if forward_path is None:
        forward_path = {}
    
    overlay = image.copy()

    rows, cols = grid.shape

    #Alle Tiles des gefundenen Pfads sammeln
    path_tiles = set(forward_path.keys()) | set(forward_path.values())

    for row in range(rows):
        for col in range(cols):
            x1 = col * tile_size
            y1 = row * tile_size
            x2 = x1 + tile_size
            y2 = y1 + tile_size

            if grid[row, col] == 1:
                color = (0, 0, 255)  # Rot für Wände

            elif (row, col) == start:
                color = (255, 0, 0)  # Blau für Startpunkt
                print(f"Startpunkt: [{row}, {col}]")

            elif (row, col) == goal:
                color = (0, 255, 255)  # Gelb für Zielpunkt
                print(f"Zielpunkt: [{row}, {col}]")

            elif (row, col) in path_tiles:
                color = (255, 0, 255)  # Magenta für den gefundenen Pfad

            elif (row, col) in visited:
                color = (0, 255, 255)  # Cyan für besuchte Kacheln

            else:
                color = (0, 255, 0)  # Grün für freie Flächen

            #Tile einfärben
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            #Schwarze Umrandung für jedes Tile
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), 1)

    alpha = 0.3  # Transparenzwert

    display = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return display