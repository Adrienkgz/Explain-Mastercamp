from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from data_processing import get_all_datas, get_sample_training
from customclass import CustomDataset, CustomTrainer


"""
Y'a des explications rapides de comment fonctionne les réseaux de neurones et comment tout marche en gros en bas + normalement tout le code est commenté sur tous les fichiers
"""
### Paramètres de l'entrainement #####

"""     
1er paramètre : label_level : C'est le niveau de précision du label que l'on souhaite prédire 
Par exemple si on a un label_level de 0 alors on prédit juste la première lettre (A, B, C, D, E, F, G, H, Y), si on a un label_level de 1, on prédit les 3 première charactère (A01, A02, ...., Y22), etc ...
""" 
LABEL_LEVEL = 1

""" 
2e paramètre : Start : c'est l'index du premier chunk que l'on souhaite prendre
Un chunk est un "morceau du fichier csv" qui contient des textes et des labels, Un chunk est égal à 1000 éléments et il y'a 50 chunks au total. Pour l'entrainement, on entraine pas sur les 10 derniers chunks pour se laisser la marge de tester dessus
"""

START = 10

""" 
3e paramètre : NBRE_CHUNKS : c'est le nombre de chunks que l'on souhaite prendre pour l'entrainement
Par exemple, si start = 0 et NBRE_CHUNKS = 5, alors on prend les 5 premiers chunks pour l'entrainement
"""

NBRE_CHUNKS = 5

""" 
4e paramètre : NBRE_EPOCHS : c'est le nombre d'epochs que l'on souhaite faire, Pour résumer, un epoch signifie que le modèle a vu tout les textes une fois
"""

NBRE_EPOCHS = 3

""" 
5e paramètre : BATCH_SIZE : c'est le nombre de prédictions que le modèle va faire avant de mettre à jour les poids

"""

BATCH_SIZE = 8

""" 
6e paramètre : USE_PRETRAINED_MODEL : c'est un booléen qui permet de choisir si on utilise un modèle pré-entrainé ou si on utilise un modèle qu'on a commencé à entrainer

"""

USE_PRETRAINED_MODEL = False

""" 
7e paramètre : TAILLE_ECHANTILLON : c'est la taille de l'échantillon que l'on va prendre pour l'entrainement
Sinon, il y'a beaucoup trop de texte par chunks et le temps d'entrainement sera beaucoup trop long
"""
TAILLE_ECHANTILLON = 6000

##########################################

# On récupère les données depuis le fichier csv sous la forme de deux listes : une pour les textes et une pour les labels
texts, labels = get_all_datas('EFREI - LIPSTIP - 50k elements EPO.csv', LABEL_LEVEL, START, NBRE_CHUNKS, use_fenetre_text=True)

# On affiche le nombre de textes récupéré
print(f"Nombre de textes original : {len(texts)}, Nombre de labels original : {len(labels)}")

# On prend un échantillon de taille TAILLE_ECHANTILLON
texts, labels = get_sample_training(texts, labels, TAILLE_ECHANTILLON)

# On charge le tokenizer
# Le tokenizer sert à transformer les textes en tokens (mots) et à "leur donner un index" pour que le modèle puisse les comprendre
tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')

# Selon le level que l'utilisateur a choisi on change le nombre de sortie du modele ( le nombre de labels que le modèle peut prédire)
if LABEL_LEVEL == 0:
    NBRE_LABELS = 9
elif LABEL_LEVEL == 1:
    NBRE_LABELS = 139
else:
    raise ValueError("Le label_level doit être 0 ou 1, parce qu'on a pas codé la suite")

# On charge le model
# Soit on charge le model pré-entrainé Scibert soit on peut prendre un modele qu'on a commencé à entrainé
if USE_PRETRAINED_MODEL:
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=NBRE_LABELS)
else:
    model = BertForSequenceClassification.from_pretrained('folder_model_lvl_1/results\checkpoint-4800', num_labels=NBRE_LABELS)
    

# On "tokenise" les textes, on les transforme en index que le modèle pourra comprendre
inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

# Pas très important, on transforme les labels en tensor d'une librairie de deep learning pour améliorer les performances
labels = torch.tensor(labels)

# On calcule une variable intermédiaire
nbre_texts = len(texts)

# On crée un dataset pour l'entrainement et un dataset pour l'évaluation en utilisant la class customDataset que l'on peut retrouver dans le fichier customclass.py
train_dataset = CustomDataset(texts[:int(0.8*nbre_texts)], labels[:int(0.8*nbre_texts)], tokenizer, max_length=512)
eval_dataset = CustomDataset(texts[int(-0.2*nbre_texts):], labels[int(-0.2*nbre_texts):], tokenizer, max_length=512)

# On définit les arguments d'entrainements
training_args = TrainingArguments(
    output_dir='./results', # Dossier où on sauvegarde le modèle pendant l'entrainement au cas ou il y'est un problème
    num_train_epochs=NBRE_EPOCHS, # Nombre d'epochs
    per_device_train_batch_size=BATCH_SIZE, # Nombre de textes voit à chaque steps lorsqu'il s'entraine
    per_device_eval_batch_size=64, # Pareil qu'avant mais quand il évalue
    warmup_steps=1, 
    weight_decay=0.0025, 
    logging_dir='./logs', # Dans quel dossier on écrit les logs = les résultats de l'entrainement
    save_strategy='epoch', # On sauvegarde le modèle à la fin de chaque epoch si 'epoch', à la fin de chaque batch si 'steps', jamais si 'no'
    logging_steps=100,  # On écrit les résultats de l'entrainement tout les ... steps
    evaluation_strategy='no',  # Quand le modèle évalue, à la fin d'un epoch si 'epoch' , jamais si 'no', à la fin d'un batch si 'steps'
    save_total_limit=6,  # On garde les 3 derniers modèles sauvegardés
)


# Fonction qui est censé calculé la précision du modèle mais ne marche pas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    
# On configure un Trainer qui est une classe issu d'une librairie qui va éviter qu'on code tout explicitement
# En plus c'est plus efficace
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# On lance le trainer
trainer.train()

# On sauvegarde
trainer.save_model('./saved_model')

""" 
Pour comprendre la rétropropagation et comment le modèle apprend:

Algorithme de rétropropagation
La rétropropagation permet de former des réseaux neuronaux artificiels. Lorsque des réseaux neuronaux artificiels se forment, les valeurs des poids sont assignées de manière aléatoire. 
L’utilisateur fixe des poids aléatoires parce qu’il ne connaît pas les valeurs correctes. Lorsque la valeur est différente de celle du réseau de rétrodiffusion attendu, il faut la considérer comme une erreur. 
L’algorithme est réglé de telle sorte que le modèle change les paramètres chaque fois que la sortie n’est pas celle attendue. L’erreur a un rapport avec les réseaux neuronaux artificiels. 
Ainsi, lorsque le paramètre change, l’erreur change également jusqu’à ce que le réseau neuronal trouve la sortie souhaitée en calculant la descente du gradient.

Descente de gradient
Lorsque l’algorithme apprend de l’erreur, il commence à trouver le minimum local. Il trouve un minimum local en partant en négatif du point actuel du gradient. 
Par exemple, si vous êtes bloqué sur une montagne entourée de brouillard qui vous empêche de voir, vous aurez besoin d’un moyen de descendre. Cependant, lorsque vous ne pouvez pas voir le chemin, 
vous pouvez trouver le minimum local dont vous pouvez disposer. Cela signifie que vous estimerez le chemin par la méthode de la descente par gradient. Par cette méthode, vous devinerez la pente en regardant 
la position actuelle de la montagne où vous vous trouvez. Ensuite, vous descendrez de la montagne en procédant dans le sens de la descente. Supposons que vous utilisiez un outil de mesure pour évaluer la pente.
Vous aurez besoin de moins de temps pour atteindre l’extrémité de la montagne.


Dans cet exemple :

Vous êtes l’algorithme de rétropropagation,
Le chemin que vous utiliserez pour descendre est celui des réseaux neuronaux artificiels,
La pente est la supposition que l’algorithme fera,
L’outil de mesure est le calcul que l’algorithme utilisera pour calculer la pente.
Votre direction sera la pente
Le temps nécessaire pour descendre de la montagne est le taux d’apprentissage de l’algorithme de rétropropagation.

Note écrite par moi :
C'est un peu différent dans le cas des modèles qu'on utilise parce qu'il y'a plein de couches et pas forcément que des réseaux de neurones simple mais l'idée est la même
si vous voulez vraiment tout comprendre bas y'a ça ( un peu un cours qui reprend tout sur les modèles de language, la tokenisation etc ou y'a pas mal de truc ou vous coder des modèles de languages):
https://huggingface.co/learn/nlp-course/chapter1/1?fw=pt

C'est mieux si vous avez déjà vu ce que c'est un réseau de neurones avant de lire ça, mais en vrai ça passe

"""