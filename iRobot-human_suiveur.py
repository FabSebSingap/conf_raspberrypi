import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform
from gpiozero import Motor

# --- CONFIGURATION DES MOTEURS (L298N avec PWM) ---
ENA = 18
IN1 = 23
IN2 = 24

ENB = 22
IN3 = 17
IN4 = 27

motor_left = Motor(forward=IN1, backward=IN2, enable=ENA)
motor_right = Motor(forward=IN3, backward=IN4, enable=ENB)

# --- PARAMÈTRES DE DISTANCE ET VISION (Mode Proche < 30cm) ---
KNOWN_WIDTH = 15.0   # Largeur moyenne d'un visage en cm
FOCAL_LENGTH = 550   # Focale étalonnée

TARGET_DIST_CM = 30.0  # Distance maximale tolérée (le robot essaie de rester en dessous)
MIN_DIST_CM = 15.0     # Seuil de sécurité minimal pour reculer

# Vesses de rotation et de déplacement adaptées au suivi rapproché
SPEED_TURN = 0.25
SPEED_MOVE = 0.4

# --- VÉRIFICATION DU MODÈLE ONNX ---
model_file = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(model_file):
    print(f"ERREUR : Le fichier '{model_file}' est introuvable !")
    exit(1)

# Chargement du réseau de neurones YuNet
net = cv2.FaceDetectorYN.create(
    model_file, "", (320, 320), 
    score_threshold=0.45, 
    nms_threshold=0.3, 
    top_k=5000
)

# --- INITIALISATION DE LA CAMÉRA (640x480) ---
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    transform=Transform(hflip=0, vflip=1)
)
picam2.configure(config)
picam2.set_controls({"ExposureValue": 2.0})
picam2.start()

print("Robot Suiveur Rapproché (< 30 cm) démarré !")
print("Appuyez sur Ctrl+C pour quitter...")

def arreter_robots():
    motor_left.stop()
    motor_right.stop()

derniere_boite = None
compteur_perte = 0
MAX_PERTE_FRAMES = 5  

# Pré-calcul de la table Gamma pour la luminosité
gamma = 0.5  
invGamma = 1.0 / gamma
gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

try:
    while True:
        # 1. Capture de l'image
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 2. Traitement d'image (Gamma + CLAHE)
        frame_gamma = cv2.LUT(frame_bgr, gamma_table)
        lab = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame_bgr_traite = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        h_img, w_img, _ = frame_bgr_traite.shape
        net.setInputSize((w_img, h_img))
        
        # 3. Détection des visages
        _, faces = net.detect(frame_bgr_traite)
        
        fresh_detection = False
        best_w = 0
        best_box = None
        
        if faces is not None:
            for face in faces:
                confidence = face[-1]
                if confidence > 0.45:
                    fresh_detection = True
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    if w > best_w:
                        best_w = w
                        best_box = (x, y, w, h)

        # Gestion de la mémoire (lissage de perte de frame)
        en_memoire = False
        if fresh_detection and best_box is not None:
            derniere_boite = best_box
            compteur_perte = 0
            humain_trouve = True
        elif derniere_boite is not None and compteur_perte < MAX_PERTE_FRAMES:
            best_box = derniere_boite
            compteur_perte += 1
            humain_trouve = True
            en_memoire = True 
        else:
            derniere_boite = None
            humain_trouve = False

        # 4. Comportement du robot
        if humain_trouve and best_box is not None:
            x, y, w, h = best_box
            
            # Calcul de la distance en cm basée sur la largeur du visage
            distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / w
            
            face_center_x = x + (w / 2)
            image_center_x = w_img / 2
            offset_x = face_center_x - image_center_x  
            
            DEADZONE = 100  # Marge de tolérance horizontale pour 640px
            
            # PRIORITÉ 1 : Centrage horizontal
            if offset_x < -DEADZONE:
                motor_left.backward(speed=SPEED_TURN)
                motor_right.forward(speed=SPEED_TURN)
            elif offset_x > DEADZONE:
                motor_left.forward(speed=SPEED_TURN)
                motor_right.backward(speed=SPEED_TURN)
            else:
                # PRIORITÉ 2 : Gestion de la distance rapprochée (< 30 cm)
                if distance_cm > TARGET_DIST_CM:
                    # Trop loin (> 30 cm), le robot avance pour se rapprocher
                    motor_left.forward(speed=SPEED_MOVE)
                    motor_right.forward(speed=SPEED_MOVE)
                elif distance_cm < MIN_DIST_CM:
                    # Trop près (< 15 cm), sécurité pour reculer
                    motor_left.backward(speed=0.3)
                    motor_right.backward(speed=0.3)
                else:
                    # Zone idéale (entre 15 cm et 30 cm) : position stable atteinte
                    arreter_robots()
        else:
            arreter_robots()

except KeyboardInterrupt:
    print("\nArrêt d'urgence demandé.")

finally:
    arreter_robots()
    try:
        picam2.stop()
        picam2.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("Programme arrêté proprement.")