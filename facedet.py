import cv2
import dlib

# Carga la imagen
img = cv2.imread('rostro.jpg')

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Carga el detector de rostros de Dlib
detector = dlib.get_frontal_face_detector()

# Detecta rostros en la imagen
rostros = detector(gray, 1)

# Carga el predictor de características faciales de Dlib
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Itera sobre los rostros detectados
for rostro in rostros:
    # Extrae las características faciales
    facial_landmarks = predictor(gray, rostro)
