from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap


class InteractiveMapLabel(QLabel):
    """Zeigt das Grundrissbild an und meldet Klicks als (row, col)."""

    tile_clicked = pyqtSignal(int, int)  # row, col

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tile_size = 20
        self.image_width = 0
        self.image_height = 0
        self.current_mode = "wall"
        self._display_pixmap: QPixmap | None = None
        self._offset_x = 0
        self._offset_y = 0
        self._scaled_width = 1
        self._scaled_height = 1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def set_image(self, pixmap: QPixmap, img_width: int, img_height: int):
        self.image_width = img_width
        self.image_height = img_height
        self._display_pixmap = pixmap
        self._update_pixmap()

    def _update_pixmap(self):
        if self._display_pixmap is None:
            return
        scaled = self._display_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._offset_x = (self.width() - scaled.width()) // 2
        self._offset_y = (self.height() - scaled.height()) // 2
        self._scaled_width = max(scaled.width(), 1)
        self._scaled_height = max(scaled.height(), 1)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            super().mousePressEvent(event)
            return

        pos = event.position()
        x = pos.x() - self._offset_x
        y = pos.y() - self._offset_y

        if not (0 <= x < self._scaled_width and 0 <= y < self._scaled_height):
            super().mousePressEvent(event)
            return

        img_x = int(x * self.image_width / self._scaled_width)
        img_y = int(y * self.image_height / self._scaled_height)

        col = img_x // self.tile_size
        row = img_y // self.tile_size

        self.tile_clicked.emit(row, col)
        super().mousePressEvent(event)