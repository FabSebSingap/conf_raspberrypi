import time
import os
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

# --- PARAMÈTRES DE DISTANCE ET VISION (YOLO - Corps entier) ---
KNOWN_WIDTH = 50.0  # Largeur moyenne des épaules d'un humain en cm
FOCAL_LENGTH = 600  # Focale indicative (à ajuster si besoin selon votre caméra)

# --- SAISIE DES PARAMÈTRES PAR L'UTILISATEUR AU DÉMARRAGE ---
print("=== CONFIGURATION DU ROBOT SUIVEUR (YOLOv8) ===")
try:
    min_cm = float(input("Entrez la distance MINIMUM en cm (ex: 50) : "))
    max_cm = float(input("Entrez la distance MAXIMUM en cm (ex: 100) : "))
    speed_left_user = float(input("Entrez la vitesse pour tourner à GAUCHE (ex: 0.2) : "))
    speed_right_user = float(input("Entrez la vitesse pour tourner à DROITE (ex: 0.2) : "))
    
    show_input = input("Activer l'affichage vidéo dans une fenêtre (O/N) ? [O] : ").strip().lower()
    show_video = False if show_input in ['n', 'non', 'no', 'false'] else True
    
    if min_cm >= max_cm:
        print("Erreur : La distance minimum doit être inférieure au maximum. Valeurs par défaut appliquées (50cm - 100cm).")
        min_cm, max_cm = 50.0, 100.0
        
    speed_left_user = max(0.0, min(1.0, speed_left_user))
    speed_right_user = max(0.0, min(1.0, speed_right_user))
    
except ValueError:
    print("Entrée invalide. Valeurs par défaut appliquées.")
    min_cm, max_cm = 50.0, 100.0
    speed_left_user, speed_right_user = 0.2, 0.2
    show_video = True

MIN_DIST_M = min_cm / 100.0
MAX_DIST_M = max_cm / 100.0

print(f"\n-> Zone cible : {min_cm} cm à {max_cm} cm")
print(f"-> Vitesse rotation GAUCHE : {speed_left_user}")
print(f"-> Vitesse rotation DROITE : {speed_right_user}")
print(f"-> Affichage vidéo : {'Activé' if show_video else 'Désactivé'}\n")

# --- VÉRIFICATION DU MODÈLE YOLOv8 ONNX ---
model_file = "yolov8n.onnx"

if not os.path.exists(model_file):
    print(f"ERREUR : Le fichier '{model_file}' est introuvable !")
    print("Téléchargez-le avec la commande : wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx")
    exit(1)

# Chargement du modèle via OpenCV DNN
net = cv2.dnn.readNetFromONNX(model_file)

# --- INITIALISATION DE LA CAMÉRA ---
picam2 = Picamera2()
config = picam2.create_video_configuration(transform=Transform(hflip=0, vflip=1))
picam2.configure(config)

picam2.set_controls({"ExposureValue": 1.0})
picam2.start()

print("Robot Suiveur (YOLOv8 Personne) démarré !")
print("Appuyez sur Ctrl+C (ou 'q' si affichage actif) pour quitter...")

def arreter_robots():
    motor_left.stop()
    motor_right.stop()

try:
    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h_img, w_img, _ = frame_bgr.shape

        # Préparation de l'image pour YOLOv8 (format 640x640)
        blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        # Analyse des sorties YOLOv8 (Format: [1, 84, 8400])
        outputs = np.squeeze(outputs)
        outputs = outputs.T  # Passage en [8400, 84]

        best_w = 0
        best_box = None
        humain_trouve = False

        for row in outputs:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            # ID 0 dans COCO dataset = "person" (Personne)
            if class_id == 0 and confidence > 0.50:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Conversion des coordonnées du modèle 640x640 vers la taille réelle de l'image
                x = int((cx - w / 2) * w_img / 640)
                y = int((cy - h / 2) * h_img / 640)
                w = int(w * w_img / 640)
                h = int(h * h_img / 640)

                if w > best_w:
                    best_w = w
                    best_box = (x, y, w, h)
                    humain_trouve = True

        # Comportement du robot
        if humain_trouve and best_box is not None:
            x, y, w, h = best_box
            
            if show_video:
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / w
            distance_m = distance_cm / 100.0
            
            person_center_x = x + (w / 2)
            image_center_x = w_img / 2
            offset_x = person_center_x - image_center_x
            
            print(f"Personne à {distance_m:.2f} m | Décalage X: {offset_x:.0f}px")
            
            DEADZONE = 350  # Marge de tolérance horizontale
            
            if offset_x < -DEADZONE:
                print(" -> Personne décalée à gauche, je tourne vers la gauche.")
                motor_left.backward(speed=speed_left_user)
                motor_right.forward(speed=speed_left_user)
                
            elif offset_x > DEADZONE:
                print(" -> Personne décalée à droite, je tourne vers la droite.")
                motor_left.forward(speed=speed_right_user)
                motor_right.backward(speed=speed_right_user)
                
            else:
                if distance_m > MAX_DIST_M:
                    print(f" -> Trop loin (> {max_cm}cm), j'avance.")
                    motor_left.forward(speed=0.5)
                    motor_right.forward(speed=0.5)
                elif distance_m < MIN_DIST_M:
                    print(f" -> Trop près (< {min_cm}cm), je recule.")
                    motor_left.backward(speed=0.4)
                    motor_right.backward(speed=0.4)
                else:
                    print(f" -> Zone idéale ({min_cm}cm - {max_cm}cm) et centré, position stable.")
                    arreter_robots()
        else:
            print("Aucune personne en vue... Arrêt complet.")
            arreter_robots()

        if show_video:
            cv2.imshow("Robot Suiveur (YOLOv8)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nArrêt d'urgence demandé.")

finally:
    arreter_robots()
    picam2.stop()
    picam2.close()
    if show_video:
        cv2.destroyAllWindows()
    print("Programme arrêté proprement.")