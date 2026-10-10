import time
from picamera2 import Picamera2, Preview

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)
picam2.start()

print("Aperçu vidéo démarré. Appuyez sur Entrée dans le terminal pour quitter...")
try:
    input() # Le programme s'arrête ici jusqu'à ce que vous pressiez Entrée
finally:
    picam2.stop()
    picam2.close()
    print("Programme arrêté proprement.")