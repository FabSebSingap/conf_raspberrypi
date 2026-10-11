from time import sleep
from gpiozero import Motor

# Initialisation du Moteur A (gauche)
moteur_a = Motor(forward=23, backward=24, enable=18)

# Initialisation du Moteur B (droit)
moteur_b = Motor(forward=17, backward=27, enable=22)

print("Les deux moteurs avancent...")
moteur_a.forward(speed=0.8)
moteur_b.forward(speed=0.8)
sleep(3)

print("Virage : Moteur A avant, Moteur B arrière...")
moteur_a.forward(speed=0.6)
moteur_b.backward(speed=0.6)
sleep(2)

print("Marche arrière des deux moteurs...")
moteur_a.backward(speed=0.7)
moteur_b.backward(speed=0.7)
sleep(3)

print("Arrêt complet.")
moteur_a.stop()
moteur_b.stop()