import torch

""" 
Ce code permet de vérifier si votre ordinateur utilise bien votre carte graphique poru entrainer le modèle
C'est possible d'entrainer le modèle sans mais c'est 3/4 fois plus long
"""
if torch.cuda.is_available():
    print("Le GPU est détecté.")
else:
    print("Le GPU n'est pas détecté.")
    
devices = torch.cuda.device_count()
if devices > 0:
    print(f"{devices} périphérique(s) GPU détecté(s).")
else:
    print("Aucun périphérique GPU détecté.")
    
device_name = torch.cuda.get_device_name(0)
print(f"Le GPU utilisé par Torch est: {device_name}")