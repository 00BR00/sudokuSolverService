# test
import cv2
import numpy as np
import tensorflow
from keras.models import load_model

# Cargar el modelo guardado
model = load_model(r"D:\PROGRAMACION\REACT NATIVE\sudoku_solver\finally solver sudoku\models\model.keras")

def preprocess_image(img_path):
    # Cargar la imagen en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionar la imagen a 28x28 píxeles
    img = cv2.resize(img, (28, 28))
    # Invertir colores si el fondo es negro y los dígitos son blancos
    img = cv2.bitwise_not(img)
    # Normalizar la imagen para que esté entre 0 y 1
    img = img / 255.0
    # Añadir dimensión para el lote y el canal de color
    img = img.reshape(1, 28, 28, 1)
    return img

# Ruta de la imagen real
img_path = r"D:\PROGRAMACION\REACT NATIVE\sudoku_solver\opencv python\MODEL\original\9\9-14.png"

# Preprocesar y realizar la predicción
preprocessed_image = preprocess_image(img_path)
prediction = model.predict(preprocessed_image)

# Obtener el dígito con la probabilidad más alta
predicted_digit = np.argmax(prediction)
print(f"Dígito predicho: {predicted_digit}")