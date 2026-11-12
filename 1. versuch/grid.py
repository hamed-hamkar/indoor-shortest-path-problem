import numpy as np

def create_grid(wall_mask, tile_size):
    height, width = wall_mask.shape

    rows = height // tile_size
    cols = width // tile_size

    grid = np.zeros((rows, cols), dtype=int)

    for row in range(rows):
        for col in range(cols):

            x1 = col * tile_size
            y1 = row * tile_size

            x2 = x1 + tile_size
            y2 = y1 + tile_size

            tile = wall_mask[y1:y2, x1:x2]

            # Anteil der schwarzen Pixel berechnen
            wall_percentage = np.count_nonzero(tile == 0) / tile.size

            if wall_percentage > 0.2:
                grid[row, col] = 1  # Wand
            else:
                grid[row, col] = 0  # Begehbar

    return grid