import cv2
import numpy as np
from picamera2 import Picamera2, Preview


picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
    
print("Caméra démarrée avec calcul de distance.")
print("Appuyez sur la touche 'q' dans la fenêtre vidéo pour quitter.")


# Seuils HSV pour le vert
vert_bas = np.array([35, 50, 50])
vert_haut = np.array([85, 255, 255])

# --- PARAMÈTRES DE CALIBRATION DE DISTANCE ---
# 1. Le diamètre réel de votre stylo en centimètres (ex: 1.0 cm)
DIAMETRE_REEL_CM = 1.0 
    
# 2. La constante de focale (à ajuster selon votre caméra et résolution).
# Formule : CONSTANTE_FOCALE = (Taille_en_pixels * Distance_connue_en_cm) / Diametre_reel_cm
# Par exemple, si à 50 cm de distance le stylo mesure 40 pixels de diamètre : 40 * 50 = 2000
CONSTANTE_FOCALE = 2000.0  

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, vert_bas, vert_haut)
            
        # Nettoyage du masque
        masque = cv2.erode(masque, None, iterations=2)
        masque = cv2.dilate(masque, None, iterations=2)

        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        if contours:
            plus_grand = max(contours, key=cv2.contourArea)
            if cv2.contourArea(plus_grand) > 500:
                # Trouver le cercle englobant
                (x, y), rayon = cv2.minEnclosingCircle(plus_grand)
                center = (int(x), int(y))
                    
                # Le diamètre du cercle en pixels
                diametre_pixels = rayon * 2

                # Calcul de la distance en centimètres
                if diametre_pixels > 0:
                    distance_cm = (DIAMETRE_REEL_CM * CONSTANTE_FOCALE) / diametre_pixels
                else:
                    distance_cm = 0

                # Dessin du cercle et du centre sur l'image
                cv2.circle(frame_bgr, center, int(rayon), (0, 255, 0), 2)
                cv2.circle(frame_bgr, center, 5, (0, 0, 255), -1)
                    
                # Affichage du texte avec la distance arrondie
                texte = f"Stylo vert - Dist: {distance_cm:.1f} cm"
                cv2.putText(frame_bgr, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Affichage de la fenêtre vidéo
            cv2.imshow("Detection Stylo Vert & Distance", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Programme arrêté.")