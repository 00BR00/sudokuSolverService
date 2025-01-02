from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from main import process_image
from sudoku_solver import resolver_sudoku, es_valido
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)
CORS(app)

# Ruta al modelo Keras
MODEL_PATH = r'D:\PROGRAMACION\REACT NATIVE\sudoku_solver\finally solver sudoku\models\model.keras'

# Cargar el modelo Keras una vez
model = load_model(MODEL_PATH)

@app.route('/solve', methods=['POST'])
def solve_sudoku():
    """
    Procesa una imagen de Sudoku, resuelve el Sudoku y devuelve la solución.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    # Leer la imagen cargada
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Procesar la imagen
    result = process_image(image=img)
    if "error" in result:
        return jsonify(result), 400

    solution = result.get("solution", None)
    if solution is not None:
        return jsonify({"solution": solution}), 200
    else:
        return jsonify({"error": "No se pudo resolver el Sudoku."}), 400

@app.route('/validate', methods=['POST'])
def validate_sudoku():
    """
    Valida si un número puede colocarse en una posición específica del Sudoku.
    """
    data = request.json
    if not data or "sudoku" not in data or "row" not in data or "col" not in data or "num" not in data:
        return jsonify({"error": "Datos incompletos. Se requiere 'sudoku', 'row', 'col' y 'num'."}), 400

    sudoku = data["sudoku"]
    row, col, num = data["row"], data["col"], data["num"]

    # Validar si el número puede colocarse en la celda
    if es_valido(sudoku, row, col, num):
        return jsonify({"valid": True}), 200
    else:
        return jsonify({"valid": False}), 200

@app.route('/example', methods=['GET'])
def example_sudoku():
    """
    Devuelve un Sudoku de ejemplo.
    """
    example = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    return jsonify({"example": example}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 8080)
