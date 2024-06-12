import random
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from listes_labels import list_label_level_0, list_label_level_1
from transformers import BertTokenizerFast

def get_all_datas(file_path, label_level, begin, nbre_chunks, use_fenetre_text = False, flatten = True, transform_html_in_text = True, only_description = False):
    """ Cet fonction est un peu la fonction principale pour récupérer les données et les mettres
    dans la forme que l'on veut pour qu'on puisse après l'exploiter

    Args:
        file_path (str): lien du fichier ou il y'a toutes les données
        label_level (int): Précision du label que l'on souhaite prédire ( voir explication dans la fonction get_outputs)
        begin (int): Chunk de début
        end (int): Chunk de fin

    Returns:
        inputs: la liste de texte de la description
        outputs: une liste qui correspond à la probabilité d'appartenance à chaque label ( voir fonction get_outputs )
    """
    # On récupère la liste de chunks ( voir explication dans finetuning.py)
    chunks = pd.read_csv(file_path, chunksize=1000, low_memory=True)
    
    # Sert uniquement à faire une barre de chargement parce que c'est stylé
    progress_bar = tqdm(total=nbre_chunks, desc="Processing chunks")
    
    # Partie pour passer les chunks que l'on ne veut pas
    for _ in range(begin):
        next(chunks)
    
    # Partie pour récupérer les données
    #Initialisation de variables
    i = 0
    outputs, inputs = [], []
    list_nbre_fenetre_par_texte = None
    for chunk in chunks:
        # Pour chaque chunks, on récupère uniquement les colonnes qui nous intéresse
        # On transforme le dataframe panda en tableau fixe numpy pour pouvoir le manipuler
        util_data = chunk[['CPC', 'claim', 'description']].to_numpy()
        
        # On récupère les labels et les inputs avec les fonctions get_outputs et get_inputs, 
        # que l'on a coder en bas et on les ajoutes à la suite dans les variables correspondantes
        if use_fenetre_text:
            input_temp, list_nbre_fenetre_par_texte = get_list_text_fenetre(util_data, flatten, transform_html_in_text, only_description=only_description)
            inputs += input_temp
        else:
            inputs += get_inputs(util_data, transform_html_in_text)
            
        outputs += get_outputs(util_data, label_level, flatten, list_nbre_fenetre_par_texte)
        
        # On met à jour la barre de chargement pour lui dire qu'on a fini un chunk
        progress_bar.update(1)
        
        # On incrémente i pour savoir ou on en est dans les chunks
        # Si on est au chunk ou il fallait s'arreter on sort de la boucle
        i += 1
        if i == nbre_chunks:
            break
    
    # On ferme la barre de chargement
    progress_bar.close()
    
    return inputs, outputs


def get_outputs(datas, label_level, flatten, list_nbre_fenetre_par_texte = None):
    """
    Get the labels from the data. 
    - 0 is the first level (A, B, C, ...)
    - 1 is the second level (A01, A02, A03, ...)
    - 2 is the third level (A01B, A01C, ...)
    - 3 is the fourth level (A01B1, A01B3, ...)
    
    Parameters
    ----------
    data : numpy.ndarray
        The data to get the labels from.
    label_level : int
        The level of the labels to get.
        
    Returns
    -------
    numpy.ndarray
        The labels.
    """
    # Partie pour récupérer les labels
    labels = []
    for i, data in enumerate(datas):
        data[0] = convert_str_labels_to_list(data[0])
        if label_level == 0: # Si le niveau de label est 0 on récupère le premier caractère de chaque label
            label_0 = []
            for label in data[0]:
                first_letter = label[0]
                if first_letter not in label_0:
                    label_0.append(first_letter)
            if (not list_nbre_fenetre_par_texte is None) and flatten:
                for _ in range(list_nbre_fenetre_par_texte[i]):
                    labels.append(tuple(label_0))
            else:
                labels.append(tuple(label_0))
        elif label_level == 1: # Si le niveau de label est 1 on récupère les trois premiers caractères de chaque label
            label_1 = []
            for label in data[0]:
                first_three_letters = label[:3]
                if first_three_letters not in label_1:
                    label_1.append(first_three_letters)
            if (not list_nbre_fenetre_par_texte is None) and flatten:
                for _ in range(list_nbre_fenetre_par_texte[i]):
                    labels.append(tuple(label_1))
            else:
                labels.append(tuple(label_1))
        elif label_level == 2: # Si le niveau de label est 2 on récupère les quatre premiers caractères de chaque label
            label_2 = []
            for label in data[0]:
                first_four_letters = label[:4]
                if first_four_letters not in label_2:
                    label_2.append(first_four_letters)
            for _ in range(list_nbre_fenetre_par_texte[i]):
                labels.append(tuple(label_2))
        elif label_level == 3: # Si le niveau de label est 3 on récupère tout les caractère avant le '-' de chaque label
            for _ in range(list_nbre_fenetre_par_texte[i]):
                labels.append((x.split('-')[0] for x in data[0]))
        else:
            raise ValueError("label_level must be 0, 1, 2 or 3")
        
    # Partie pour transformer les labels en tuple compréhensible par le modèle
    ### Exemple : labels = [['A', 'B'], ['A', 'C'], ['B', 'C']] => outputs = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
    
    if label_level == 0:
        outputs = []
        for label in labels:
            output = [1.0 if l in label else 0.0 for l in list_label_level_0]
            outputs.append(output)
    elif label_level == 1:
        outputs = []
        for label in labels:
            output = [1.0 if l in label else 0.0 for l in list_label_level_1]
            outputs.append(output)
    else:
        raise ValueError("label_level must be 0 or 1 because we dont have coded the other levels yet.")
    return outputs

# Définition de la fonction convert_str_labels_to_list en local dans la fonction get_labels
    
def convert_str_labels_to_list(data):
    """
    Convert the string labels to a list of labels.
    
    Parameters
    ----------
    data : str
        The string to convert.
        
    Returns
    -------
    list
        The list of labels.
    """
    return data.replace('[', '').replace(']', '').replace(' ', '').replace("'", '').split(',')

def get_inputs(datas, transform_html_in_text):
    """
    Get the inputs from the data.
    
    Parameters
    ----------
    data : numpy.ndarray
        The data to get the inputs from.
        
    Returns
    -------
    numpy.ndarray
        The inputs.
    """
    #return ((get_text_from_html_doc(data[1]), get_text_from_html_doc(data[2])) for data in datas)
    return [get_text_from_html_doc(data[1]) for data in datas] if transform_html_in_text else datas[:, 2]

def get_text_from_html_doc(html_doc):
        """
        Formats the given claim by removing any HTML tags and returning the plain text.

        Parameters:
        claim (str): The claim to be formatted.

        Returns:
        str: The formatted claim text.
        """
        soup_html = BeautifulSoup(html_doc, 'lxml')
        return soup_html.get_text() 

#### Ensemble de fonctions utilisé pour le fenetrage du texte 

def get_list_text_fenetre(datas, flatten, transform_html_in_text, only_description = False):
    list_texts, list_nbre_fenetre_par_texte = [], []
    
    # Tokenize all the texts at once
    descriptions = [get_text_from_html_doc(data[1]) if transform_html_in_text else data[1] for data in datas]
    if not only_description:
        claims = [get_text_from_html_doc(data[2]) if transform_html_in_text else data[2] for data in datas]
        texts = [f"Here is a description : {description}. And now it's the claim : {claim}" for description, claim in zip(descriptions, claims)]
    else:
        texts = [description for description in descriptions]
    texts_split = [text.split() for text in texts]
    
    for text_tokens in texts_split:
        list_text = []
        nbre_fenetre_pour_ce_texte = 0        
        for i in range(0, len(text_tokens), 450):
            if flatten:
                list_texts.append(' '.join(text_tokens[i:i+490]))
            else:
                list_text.append(' '.join(text_tokens[i:i+490]))
            nbre_fenetre_pour_ce_texte += 1
        if not flatten:        
            list_texts.append(list_text)
        list_nbre_fenetre_par_texte.append(nbre_fenetre_pour_ce_texte)
    return list_texts, list_nbre_fenetre_par_texte
        
def get_sample_training(texts, labels, taille_echantillon):
    """
    Get a sample of the training data.
    
    Parameters
    ----------
    texts : list
        The texts to get the sample from.
    labels : list
        The labels to get the sample from.
    taille_echantillon : int
        The size of the sample.
        
    Returns
    -------
    tuple
        The sample of texts and labels.
    """
    lists_index_aleatoire = random.sample(range(len(texts)), taille_echantillon)
    return [texts[i] for i in lists_index_aleatoire], [labels[i] for i in lists_index_aleatoire]