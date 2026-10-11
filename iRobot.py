import cv2
import numpy as np
from time import sleep
from gpiozero import Motor


# Initialisation du Moteur A (gauche)
moteur_a = Motor(forward=23, backward=24, enable=18)

# Initialisation du Moteur B (droit)
moteur_b = Motor(forward=17, backward=27, enable=22)


from picamera2 import Picamera2, Preview

# 1. Initialisation de la caméra Innomaker via Picamera2

picam2 = Picamera2()

# configure une résolution modérée (640x480) pour garantir une bonne fluidité (FPS)
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
    
print("Caméra démarrée. Recherche du stylo vert en cours...")
print("Appuyez sur la touche 'q' de votre clavier pour quitter.")

vert_bas = np.array([35, 50, 50])
vert_haut = np.array([85, 255, 255])

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, vert_bas, vert_haut)
            
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            plus_grand = max(contours, key=cv2.contourArea)
            if cv2.contourArea(plus_grand) > 500:
                (x, y), rayon = cv2.minEnclosingCircle(plus_grand)
                cv2.circle(frame_bgr, (int(x), int(y)), int(rayon), (0, 255, 0), 2)
                cv2.putText(frame_bgr, "Stylo vert detecte !", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print("Les deux moteurs avancent...")
                moteur_a.forward(speed=0.8)
                moteur_b.forward(speed=0.8)
                print("Arrêt complet.")
                sleep(0.5)
                moteur_a.stop()
                moteur_b.stop()

            # Affichage de l'image
            cv2.imshow("Detection Stylo Vert - Pi 5", frame_bgr)

            # Attente indispensable pour actualiser la fenêtre (1 milliseconde)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Programme arrêté.")
