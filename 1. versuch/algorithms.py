
import heapq


def astar(grid, start, goal):

    print("A1: A* gestartet", flush=True)

    rows, cols = grid.shape

    # Start und Ziel überprüfen
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return {}, set()

    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return {}, set()

    if grid[start] == 1 or grid[goal] == 1:
        return {}, set()

    # Start und Ziel sind identisch
    if start == goal:
        return {}, {start}

    # Kosten initialisieren
    g_score = {
        (row, col): float('inf')
        for row in range(rows)
        for col in range(cols)
    }

    g_score[start] = 0

    # Priority Queue
    open_set = []

    heapq.heappush(
        open_set,
        (h(start, goal), start)
    )

    # Vorgänger speichern
    came_from = {}

    # Untersuchte Tiles
    visited = set()

    # Bewegungsrichtungen
    directions = [
        (-1, 0),  # Norden
        (1, 0),   # Süden
        (0, -1),  # Westen
        (0, 1)    # Osten
    ]

    print("A2: Initialisierung abgeschlossen", flush=True)

    while open_set:

        current_f, current = heapq.heappop(open_set)

        if current in visited:
            continue

        visited.add(current)

        # Ziel erreicht
        if current == goal:
            print("A3: Ziel gefunden", flush=True)
            break

        # Nachbarn untersuchen
        for dr, dc in directions:

            neighbor = (
                current[0] + dr,
                current[1] + dc
            )

            # Grenzen überprüfen
            if not (0 <= neighbor[0] < rows and
                    0 <= neighbor[1] < cols):
                continue

            # Wände überspringen
            if grid[neighbor] == 1:
                continue

            # Bereits untersuchte Tiles überspringen
            if neighbor in visited:
                continue

            # Kosten berechnen
            tentative_g = g_score[current] + 1

            if tentative_g < g_score[neighbor]:

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + h(neighbor, goal)

                heapq.heappush(
                    open_set,
                    (f_score, neighbor)
                )

    # Prüfen, ob Ziel erreicht wurde
    if goal not in came_from:
        print("A4: Kein Pfad gefunden", flush=True)
        return {}, visited

    # Pfad rekonstruieren
    forward_path = {}
    current = goal

    while current != start:

        previous = came_from[current]

        forward_path[previous] = current

        current = previous

    print("A5: Pfad rekonstruiert", flush=True)
    print("Besuchte Tiles:", len(visited), flush=True)
    print("Pfadlänge:", len(forward_path), flush=True)

    return forward_path, visited


def h(tile1, tile2):
    # Manhattan-Distanz
    return (
        abs(tile1[0] - tile2[0]) +
        abs(tile1[1] - tile2[1])
    )


# def dijkstra(grid, start, goal):



# def bfs(grid, start, goal):



# def dfs(grid, start, goal):



# def double_dijkstra(grid, start, goal):