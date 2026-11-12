import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def cv2_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(
        rgb_image.data, w, h, bytes_per_line,
        QImage.Format.Format_RGB888
    ).copy()  # .copy() ist entscheidend, sonst zeigt das QImage auf flüchtigen Speicher
    return QPixmap.fromImage(qt_image)