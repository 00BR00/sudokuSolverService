#sudoku_detection.py
import cv2
import numpy as np


def find_largest_contour(img):
    """Encuentra el contorno más grande, que debería ser el Sudoku."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0] if contours else None

def get_sudoku_grid(img, contour):
    """Transforma la imagen para obtener una vista cuadrada del Sudoku."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = height = 450  # Tamaño de salida deseado
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(np.array(box, dtype="float32"), dst_pts)
    return cv2.warpPerspective(img, m, (width, height))

def split_into_cells(img):
    """Divide la imagen en 81 celdas y resalta cada celda con un borde."""
    rows = np.vsplit(img, 9)
    cells = []
    for row in rows:
        row_cells = np.hsplit(row, 9)
        for cell in row_cells:
            # Resalta la celda con un borde blanco grueso
            cell_with_border = cv2.copyMakeBorder(cell, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cells.append(cell_with_border)
    return cells
