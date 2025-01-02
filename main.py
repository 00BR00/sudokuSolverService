import cv2
import numpy as np
from utils import preProcess, biggestContour, reorder, splitBoxes, getPrediction, displayNumbers, initializePredectionModel
import sudoku_solver

# Inicializa el modelo globalmente
model = initializePredectionModel()

def process_image(image_path=None, image=None):
    """
    Procesa una imagen de Sudoku y devuelve el resultado.
    :param image_path: Ruta de la imagen.
    :param image: Imagen en forma de matriz (opcional si no se proporciona una ruta).
    :return: Diccionario con los resultados o error.
    """
    try:
        # Leer la imagen
        if image_path:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen en la ruta: {image_path}")
        elif image is not None:
            img = image
        else:
            raise ValueError("Se necesita una imagen o una ruta de imagen.")

        # Preprocesar
        img = cv2.resize(img, (450, 450))
        imgThreshold = preProcess(img)

        # Encontrar contornos
        contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest, _ = biggestContour(contours)
        if biggest.size == 0:
            return {"error": "No se encontró un contorno válido."}

        # Transformar perspectiva
        biggest = reorder(biggest)
        matrix = cv2.getPerspectiveTransform(
            np.float32(biggest),
            np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        )
        imgWarpColored = cv2.warpPerspective(img, matrix, (450, 450))
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        # Dividir la imagen en celdas
        boxes = splitBoxes(imgWarpColored)
        numbers = getPrediction(boxes, model)
        sudoku_array = np.array(numbers).reshape(9, 9)

        # Resolver el Sudoku
        if sudoku_solver.resolver_sudoku(sudoku_array):
            return {"solution": sudoku_array.tolist()}
        else:
            return {"error": "El Sudoku no tiene solución."}
    except Exception as e:
        return {"error": str(e)}
