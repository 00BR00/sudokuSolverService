#utils.py
import cv2
import numpy as np
import tensorflow
from keras.models import load_model

# 1. Cargar el modelo
def initializePredectionModel():
    model = load_model(r'D:\PROGRAMACION\REACT NATIVE\sudoku_solver\finally solver sudoku\models\model.keras')
    return model

# 2. Preprocesar la imagen
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Imagen en escala de grises
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Desenfoque gaussiano
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # Umbral adaptativo
    return imgThreshold

# 3. Reordenar puntos para la transformación de perspectiva
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# 4. Encontrar el contorno más grande
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# 5. Dividir la imagen en 81 cuadros
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

# 6. Obtener predicción del modelo
def getPrediction(boxes, model):
    result = []
    for image in boxes:
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)  # Ajuste a (1, 28, 28, 1)
        
        # Obtener predicción
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)  # Corrección
        probabilityValue = np.amax(predictions)
        print(classIndex, probabilityValue)
        
        # Guardar el resultado si la probabilidad es alta
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


# 7. Mostrar los números en la imagen
def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for x in range(9):
        for y in range(9):
            if numbers[(y * 9) + x] != 0:
                cv2.putText(img, str(numbers[(y * 9) + x]),
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return img

# 8. Apilar todas las imágenes en una ventana
def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)
    
    return ver
