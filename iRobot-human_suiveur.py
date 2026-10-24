import time
import os
import cv2
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

# --- PARAMÈTRES DE DISTANCE ET VISION ---
KNOWN_WIDTH = 15.0  # Largeur moyenne d'un visage en cm
FOCAL_LENGTH = 550  # Votre focale étalonnée pour l'InnoMaker

# --- VÉRIFICATION DU MODÈLE ONNX ---
model_file = "face_detection_yunet_2023mar.onnx"

if not os.path.exists(model_file):
    print(f"ERREUR : Le fichier '{model_file}' est introuvable dans ce dossier !")
    exit(1)

# Chargement du réseau de neurones YuNet
net = cv2.FaceDetectorYN.create(
    model_file, "", (300, 300), 
    score_threshold=0.6, 
    nms_threshold=0.3, 
    top_k=5000
)

# --- INITIALISATION DE LA CAMÉRA ---
picam2 = Picamera2()
config = picam2.create_video_configuration(transform=Transform(hflip=1, vflip=1))
picam2.configure(config)

picam2.start()

print("Robot Suiveur Intelligent démarré ! Appuyez sur 'q' ou Ctrl+C pour quitter...")

def arreter_robots():
    motor_left.stop()
    motor_right.stop()

try:
    while True:
        # 1. Capture de l'image
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        h_img, w_img, _ = frame_bgr.shape
        net.setInputSize((w_img, h_img))
        
        # 2. Détection
        _, faces = net.detect(frame_bgr)
        
        humain_trouve = False
        best_w = 0
        best_box = None
        
        # 3. Analyse des visages détectés
        if faces is not None:
            for face in faces:
                confidence = face[-1]
                if confidence > 0.6:
                    humain_trouve = True
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    if w > best_w:
                        best_w = w
                        best_box = (x, y, w, h)

        # 4. Comportement du robot selon vos règles
        if humain_trouve and best_box is not None:
            x, y, w, h = best_box
            
            # Calcul de la distance
            distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / w
            distance_m = distance_cm / 100.0
            
            # Calcul du positionnement horizontal (centre du visage vs centre de la caméra)
            face_center_x = x + (w / 2)
            image_center_x = w_img / 2
            offset_x = face_center_x - image_center_x  # Négatif = à gauche, Positif = à droite
            
            print(f"Humain à {distance_m:.2f} m | Décalage X: {offset_x:.0f}px")
            
            DEADZONE = 45  # Marge de tolérance en pixels pour considérer que le visage est au centre
            
            # Règle 1 & 2 : Si l'humain est à gauche ou à droite, le robot tourne sur place pour le centrer
            if offset_x < -DEADZONE:
                print(" -> Humain sur la gauche, je tourne vers la gauche.")
                motor_left.backward(speed=0.25)
                motor_right.forward(speed=0.25)
                
            elif offset_x > DEADZONE:
                print(" -> Humain sur la droite, je tourne vers la droite.")
                motor_left.forward(speed=0.25)
                motor_right.backward(speed=0.25)
                
            else:
                # Règle 3 : L'humain est bien au centre, on gère la zone de distance (entre 1m et 2m)
                if distance_m > 2.0:
                    print(" -> Humain à plus de 2m, j'avance pour me rapprocher.")
                    motor_left.forward(speed=0.35)
                    motor_right.forward(speed=0.35)
                elif distance_m < 1.0:
                    print(" -> Humain à moins de 1m, je recule.")
                    motor_left.backward(speed=0.35)
                    motor_right.backward(speed=0.35)
                else:
                    print(" -> Parfait : Centré et entre 1m et 2m, je m'arrête.")
                    arreter_robots()
                
            # Dessiner le rectangle vert autour du visage
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        else:
            # Aucun humain : Le robot tourne sur lui-même pour chercher
            print("Aucun humain en vue... Recherche en cours.")
            #motor_left.forward(speed=0.2)
            #motor_right.backward(speed=0.2)

        # Affichage de la vidéo
        cv2.imshow("Robot Suiveur", frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nArrêt du programme...")
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nArrêt d'urgence demandé.")

finally:
    arreter_robots()
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
    print("Programme arrêté proprement.")