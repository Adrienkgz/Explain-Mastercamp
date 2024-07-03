from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, Trainer, T5ForSequenceClassification, AutoModelForSeq2SeqLM
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
import torch
import numpy as np
from data_processing import get_all_datas, get_sample_training, get_confusion_matrix
from customclass import CustomDataset, CustomTrainer, TextClassificationDataset
from transformers import EarlyStoppingCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from listes_labels import list_label_level_0, list_label_level_1
"""
Y'a des explications rapides de comment fonctionne les réseaux de neurones et comment tout marche en gros en bas + normalement tout le code est commenté sur tous les fichiers
"""
### Paramètres de l'entrainement #####

"""     
1er paramètre : label_level : C'est le niveau de précision du label que l'on souhaite prédire 
Par exemple si on a un label_level de 0 alors on prédit juste la première lettre (A, B, C, D, E, F, G, H, Y), si on a un label_level de 1, on prédit les 3 première charactère (A01, A02, ...., Y22), etc ...
""" 
LABEL_LEVEL = 0

""" 
2e paramètre : Start : c'est l'index du premier chunk que l'on souhaite prendre
Un chunk est un "morceau du fichier csv" qui contient des textes et des labels, Un chunk est égal à 1000 éléments et il y'a 50 chunks au total. Pour l'entrainement, on entraine pas sur les 10 derniers chunks pour se laisser la marge de tester dessus
"""

START = 0

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

BATCH_SIZE = 2

""" 
6e paramètre : USE_PRETRAINED_MODEL : c'est un booléen qui permet de choisir si on utilise un modèle pré-entrainé ou si on utilise un modèle qu'on a commencé à entrainer

"""

USE_PRETRAINED_MODEL = True

""" 
7e paramètre : TAILLE_ECHANTILLON : c'est la taille de l'échantillon que l'on va prendre pour l'entrainement
Sinon, il y'a beaucoup trop de texte par chunks et le temps d'entrainement sera beaucoup trop long
"""
TAILLE_ECHANTILLON = 6000

NO_CHUNKS = True

##########################################

# On récupère les données depuis le fichier csv sous la forme de deux listes : une pour les textes et une pour les labels
texts, labels = get_all_datas('EFREI - LIPSTIP - 50k elements EPO balanced train.csv', LABEL_LEVEL, START, NBRE_CHUNKS, NO_CHUNKS, model_preparation='t5',  use_fenetre_text=False, only_claim=True, taille_fenetre=400, intervalle_fenetre = 100)
print(len(texts), len(labels))
# On affiche le nombre de textes récupéré
print(f"Nombre de textes original : {len(texts)}, Nombre de labels original : {len(labels)}")

# On prend un échantillon de taille TAILLE_ECHANTILLON
#texts, labels = get_sample_training(texts, labels, TAILLE_ECHANTILLON)

# On charge le tokenizer
# Le tokenizer sert à transformer les textes en tokens (mots) et à "leur donner un index" pour que le modèle puisse les comprendre
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")


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
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
else:
    model = T5ForConditionalGeneration.from_pretrained('saved_model', num_labels=NBRE_LABELS)

# On calcule une variable intermédiaire
nbre_texts = len(texts)

# On crée un dataset pour l'entrainement et un dataset pour l'évaluation en utilisant la class customDataset que l'on peut retrouver dans le fichier customclass.py
train_dataset = TextClassificationDataset(texts[:int(0.9*nbre_texts)], labels[:int(0.9*nbre_texts)], tokenizer, max_length=512)
eval_dataset = TextClassificationDataset(texts[int(-0.02*nbre_texts):], labels[int(-0.02*nbre_texts):], tokenizer, max_length=512)

# On définit les arguments d'entrainements
training_args = TrainingArguments(
    output_dir='./results', # Dossier où on sauvegarde le modèle pendant l'entrainement au cas ou il y'est un problème
    overwrite_output_dir=True, # On écrase le dossier de sauvegarde si il existe déjà
    num_train_epochs=NBRE_EPOCHS, # Nombre d'epochs
    per_device_train_batch_size=BATCH_SIZE, # Nombre de textes voit à chaque steps lorsqu'il s'entraine
    per_device_eval_batch_size=BATCH_SIZE, # Pareil qu'avant mais quand il évalue
    warmup_steps=0, 
    weight_decay=0.001, 
    logging_dir='./logs', # Dans quel dossier on écrit les logs = les résultats de l'entrainement
    save_strategy='steps', # On sauvegarde le modèle à la fin de chaque epoch si 'epoch', à la fin de chaque batch si 'steps', jamais si 'no'
    logging_steps=100,  # On écrit les résultats de l'entrainement tout les ... steps
    eval_strategy='steps',  # Quand le modèle évalue, à la fin d'un epoch si 'epoch' , jamais si 'no', à la fin d'un batch si 'steps'
    eval_steps=1000,  # On évalue le modèle tout les ... steps
    save_steps=1000,  # On sauvegarde le modèle tout les ... steps
    save_total_limit=6,  # On garde les n derniers modèles sauvegardés
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    fp16=True,  # On utilise le half precision pour accélérer l'entrainement
    fp16_opt_level = 'O1',  # On utilise le half precision pour accélérer l'entrainement
    fp16_full_eval= True,  # On utilise le half precision pour accélérer l'entrainement
    half_precision_backend=True,
    eval_accumulation_steps=5
)


# Fonction qui est censé calculé la précision du modèle mais ne marche pas
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=-1)

    # Décoder les prédictions et les étiquettes en utilisant le tokenizer
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    y_pred, y_true = [], []
    for prediction, label in zip(predictions, labels):
        prediction_indices = [list_label_level_0.index(pred) if pred in list_label_level_0 else -1 for pred in prediction.split()]
        label_indices = [list_label_level_0.index(label) for label in label.split()]
        prediction_indices = prediction_indices[:len(label_indices)]

        # On supprime les indices -1 dans les prédictions
        prediction_indices = [i for i in prediction_indices if i != -1]

        # On convertis en vecteur one-hot
        label_one_hot = np.zeros(len(list_label_level_0))
        prediction_one_hot = np.zeros(len(list_label_level_0))
        for i in label_indices:
            label_one_hot[i] = 1
        for i in prediction_indices:
            prediction_one_hot[i] = 1

        # On ajoute le one-hot dans une liste
        y_pred.append(prediction_one_hot)
        y_true.append(label_one_hot)

    # Calculer les métriques d'évaluation
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    tp, tn, fp, fn = get_confusion_matrix(y_true, y_pred)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'perso': tp-fp
    }

# On crée un callback qui va arreter l'entrainement si la loss de l'eveluation ne diminue pas pendant 3 epochs
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

# On configure un Trainer qui est une classe issu d'une librairie qui va éviter qu'on code tout explicitement
# En plus c'est plus efficace
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
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


Amélioration possible
Régularization
Early stopping
Optimisation des hyperparamètres
Equilibrage des données
l1 et l2 regularization
"""