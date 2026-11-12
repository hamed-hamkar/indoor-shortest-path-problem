from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout

from image_processing import load_image, detect_walls
from grid import create_grid
from visualization import draw_grid
from image_utils import cv2_to_qpixmap
from map_widget import InteractiveMapLabel
from algorithms import astar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # GUI aus Qt Designer laden
        uic.loadUi("main_window.ui", self)

        # --- Daten vorbereiten ---
        self.tile_size = 20
        self.image_path = "Bilder/Multi-Room-Floor-Plan-Template-edit-online-1123609580.png"

        self.image, self.gray_image = load_image(self.image_path)
        self.wall_mask = detect_walls(self.gray_image)
        self.grid = create_grid(self.wall_mask, self.tile_size)
        self.start = None
        self.goal = None
        self.visited = set()
        self.forward_path = {}


        # --- Interaktives Karten-Widget in den mapContainer einsetzen ---
        self.map_label = InteractiveMapLabel(self.mapContainer)
        self.map_label.tile_size = self.tile_size
        self.map_label.tile_clicked.connect(self.on_tile_clicked)

        layout = QVBoxLayout(self.mapContainer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.map_label)

        # --- Buttons verbinden ---
        self.btn_wall.clicked.connect(lambda: self.set_mode("wall"))
        self.btn_free.clicked.connect(lambda: self.set_mode("free"))
        self.btn_start.clicked.connect(lambda: self.set_mode("start"))
        self.btn_goal.clicked.connect(lambda: self.set_mode("goal"))
        self.btn_reset.clicked.connect(self.reset_map)
        self.btn_astar.clicked.connect(self.run_astar)

        # --- Startzustand ---
        self.set_mode("wall")
        self.update_map_display()

    def set_mode(self, mode: str):
        self.current_mode = mode
        self.map_label.current_mode = mode
        self.lblStatus.setText(f"Modus: {mode}")

    def run_astar(self):

        print("1. A* gestartet")

        if self.start is None or self.goal is None:
            self.lblStatus.setText("Bitte zuerst Start und Ziel setzen.")
            return

        print("2. Start:", self.start)
        print("3. Goal:", self.goal)
        print("4. Grid Shape:", self.grid.shape)

        self.visited.clear()
        self.forward_path.clear()

        print("5. Starte Algorithmus")

        self.forward_path, self.visited = astar(
            self.grid,
            self.start,
            self.goal
        )

        print("6. Algorithmus abgeschlossen")
        print("Visited:", len(self.visited))
        print("Forward Path:", len(self.forward_path))

        if not self.forward_path and self.start != self.goal:
            self.lblStatus.setText("Kein Pfad gefunden!")
        else:
            self.lblStatus.setText("A* abgeschlossen!")

        print("7. Starte Visualisierung")

        self.update_map_display()

        print("8. Visualisierung abgeschlossen")

    def on_tile_clicked(self, row: int, col: int):
        rows, cols = self.grid.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return

        if self.current_mode == "start":
            if self.grid[row, col] == 1:
                self.lblStatus.setText("Startpunkt kann nicht auf einer Wand gesetzt werden.")
                return

            self.start = (row, col)
            self.lblStatus.setText(f"Startpunkt gesetzt: [{row}, {col}]")
            print("GUI Start:", self.start)

        if self.current_mode == "goal":
            if self.grid[row, col] == 1:
                self.lblStatus.setText("Zielpunkt kann nicht auf einer Wand gesetzt werden.")
                return

            self.goal = (row, col)
            self.lblStatus.setText(f"Zielpunkt gesetzt: [{row}, {col}]")
            print("GUI Goal:", self.goal)

        if self.current_mode == "wall":
            if (row, col) == self.start:
                self.start = None

            if (row, col) == self.goal:
                self.goal = None

            self.grid[row, col] = 1

        if self.current_mode == "free":
            self.grid[row, col] = 0

        self.update_map_display()

    def update_map_display(self):
        display_image = draw_grid(self.image, self.grid, self.tile_size, self.start, self.goal, self.visited, self.forward_path)
        pixmap = cv2_to_qpixmap(display_image)
        h, w = self.image.shape[:2]
        self.map_label.set_image(pixmap, w, h)

    def reset_map(self):
        self.grid = create_grid(self.wall_mask, self.tile_size)
        self.lblStatus.setText("Map zurückgesetzt")
        self.update_map_display()