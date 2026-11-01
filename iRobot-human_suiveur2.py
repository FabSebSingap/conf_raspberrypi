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

# Chargement du réseau de neurones YuNet (Seuil de confiance abaissé à 0.45 pour plus de tolérance)
net = cv2.FaceDetectorYN.create(
    model_file, "", (300, 300), 
    score_threshold=0.45, 
    nms_threshold=0.3, 
    top_k=5000
)

# --- INITIALISATION DE LA CAMÉRA ---
picam2 = Picamera2()
#config = picam2.create_video_configuration(transform=Transform(hflip=1, vflip=1))
config = picam2.create_video_configuration(transform=Transform(hflip=0, vflip=1)) # hflip mis à 0
picam2.configure(config)

# Compensation d'exposition pour atténuer le contre-jour (valeur positive)
picam2.set_controls({"ExposureValue": 1.0})

picam2.start()

print("Robot Suiveur Intelligent (Optimisé Contre-jour & Immobilité) démarré !")
print("Appuyez sur 'q' ou Ctrl+C pour quitter...")

def arreter_robots():
    motor_left.stop()
    motor_right.stop()

# --- INITIALISATION DE LA MÉMOIRE (Délai de grâce) ---
derniere_boite = None
compteur_perte = 0
MAX_PERTE_FRAMES = 10  # Nombre d'images où l'on garde en mémoire le dernier visage vu

try:
    while True:
        # 1. Capture de l'image
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # --- AMÉLIORATION CONTRE-JOUR (CLAHE) ---
        # Conversion en LAB pour appliquer l'égalisation adaptative uniquement sur la luminosité (canal L)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        frame_bgr_traite = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # ----------------------------------------
        
        h_img, w_img, _ = frame_bgr_traite.shape
        net.setInputSize((w_img, h_img))
        
        # 2. Détection sur l'image traitée
        _, faces = net.detect(frame_bgr_traite)
        
        humain_trouve = False
        best_w = 0
        best_box = None
        
        # 3. Analyse des visages détectés
        if faces is not None:
            for face in faces:
                confidence = face[-1]
                if confidence > 0.45:
                    humain_trouve = True
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    if w > best_w:
                        best_w = w
                        best_box = (x, y, w, h)

        # Gestion de la mémoire (Délai de grâce si le visage saute brièvement à l'arrêt)
        if humain_trouve and best_box is not None:
            derniere_boite = best_box
            compteur_perte = 0  # On réinitialise le compteur
        elif derniere_boite is not None and compteur_perte < MAX_PERTE_FRAMES:
            best_box = derniere_boite
            compteur_perte += 1
            humain_trouve = True
            print(f" -> [Mémoire] Visage non détecté sur cette frame ({compteur_perte}/{MAX_PERTE_FRAMES})")
        else:
            derniere_boite = None

        # 4. Comportement du robot
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
            
            DEADZONE = 200  # Marge de tolérance en pixels pour considérer le visage au centre
            
            # Règle de centrage (rotation sur place)
            if offset_x < -DEADZONE:
                print(" -> Humain sur la gauche, je tourne vers la gauche.")
                motor_left.backward(speed=0.3)
                motor_right.forward(speed=0.3)
                
            elif offset_x > DEADZONE:
                print(" -> Humain sur la droite, je tourne vers la droite.")
                motor_left.forward(speed=0.3)
                motor_right.backward(speed=0.3)
                
            else:
                # Gestion de la distance (si l'humain est bien centré)
                # --- RÈGLES DE DISTANCE ---
                if distance_m > 0.8:
                    print(" -> Humain à plus de 50 cm, j'attends (pas de déclenchement).")
                    arreter_robots()
                elif distance_m < 0.4:
                    print(" -> Humain à moins de 20 cm, je m'arrête (trop près !).")
                    arreter_robots()
                else:
                    print(" -> Humain entre 20 cm et 50 cm, j'avance pour me rapprocher.")
                    motor_left.forward(speed=0.5)
                    motor_right.forward(speed=0.5)
            
            # Dessiner le rectangle vert autour du visage sur l'image traitée
            cv2.rectangle(frame_bgr_traite, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        else:
            print("Aucun humain en vue... Recherche en cours.")
            arreter_robots()

        # Affichage de la vidéo (nécessite un écran ou VNC)
        cv2.imshow("Robot Suiveur", frame_bgr_traite)
        
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