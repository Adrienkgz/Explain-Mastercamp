from data_processing import get_all_datas
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import data_processing
from listes_labels import list_label_level_0
from transformers_interpret import SequenceClassificationExplainer
from data_processing import get_text_from_html_doc

class PatentPredicterAI:
    def __init__(self, level, filepath_model = 'model lvl 0 - balanced', use_gpu = True, batch_size = 8, verbose = True):
        # Messages d'erreur
        if level not in [0, 1, 2, 3]: raise ValueError('level must be in [0, 1, 2, 3]')
        
        # Initialisation des paramètres 
        self.level = level # Le niveau de prédiction
        self.device = torch.device('cuda') if (use_gpu and torch.cuda.is_available()) else torch.device('cpu') # Paramètrage du device et si le modèle est chargé sur le processeur ou la carte graphique
        self.batch_size = batch_size # Taille des batchs de prédiction ( par exemple si batch_size = 8 alors lors de la prédiciton il prédira les éléments 8 par 8)
        self.filepath_model = filepath_model
        self.tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased') # Tokenizer pour transformer les textes en entrée du modèle on prend celui de Scibert
        self.verbose = verbose # Paramètre pour que le programme affiche des messages lors de son exécution
        self.model = self.initialise_model() # Chargement du modèle
    
    def initialise_model(self):
        model = BertForSequenceClassification.from_pretrained(self.filepath_model)
        model.to(self.device)
        if self.verbose:
            print(f"Chargement du modèle sur le device : {self.device}")
        return model
    
    def test(self, chunk, nbre_prediction = 100, use_fenetre_text = True, only_description = False, method = '3*avg', only_claim = False):
        inputs, outputs = get_all_datas(file_path='EFREI - LIPSTIP - 50k elements EPO.csv', 
                                        label_level=self.level, 
                                        begin=chunk,
                                        dont_use_chunks=False,
                                        nbre_chunks=1,
                                        use_fenetre_text=use_fenetre_text,
                                        only_description=only_description,
                                        flatten=False,
                                        only_claim=only_claim)
        
        # Initialisation des métriques et compteurs
        i = 0
        y_true, y_pred = [], []
        for input in tqdm(inputs[:nbre_prediction], desc='Prédiction en cours'):
            # Prédiction des classes
            list_classes_prédites = self.predict(input = input, get_dictionnary_with_confidence=False, method = method) # On prédit les classes pour le texte
            
            y_pred_temp = [1 if j in list_classes_prédites else 0 for j in range(len(outputs[i]))] # On convertis les classes prédites en liste de 0 et de 1
            y_true_temp = [1 if j == 1 else 0 for j in outputs[i]] # On convertis les classes réelles en liste de 0 et de 1
            
            # On convertis l'outputs en liste de classe ( outputs est une liste de probabilité d'appartenance à chaque classe soit une liste de 0 et de 1 qui par exemple s'il y'a un 1 à l'indice 3 alors la classe 3 est celle du texte et ainsi de suite)
            list_classes_reels = [j for j, value in enumerate(outputs[i]) if value == 1]
            i += 1
            y_pred.append(y_pred_temp)
            y_true.append(y_true_temp)
        # On calcule les métriques
        f1_micro = f1_score(y_true, y_pred, average='micro')
        print(f'F1 micro : {f1_micro:.5f}')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        print(f'F1 weigted : {f1_weighted:.5f}')
        exact_match_ratio = sum([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])/len(y_true)
        print(f'Exact match ratio : {exact_match_ratio:.5f}')
        
        
    def predict(self, input, method, get_dictionnary_with_confidence=False):
        # Prétraitement du texte pour enlever les balises HTML si nécessaire
        input_cleaned = get_text_from_html_doc(input) if '<' in input and '>' in input else input
        input_cleaned = input_cleaned[:510]

        # Tokenisation et préparation de l'input pour le modèle
        input_for_this_text = self.tokenizer(input_cleaned, padding=True, truncation=True, max_length=512, return_tensors='pt')
        output_for_this_text = torch.tensor([])

        # Calcul des logits par le modèle sur les données tokenisées
        with torch.no_grad():
            for i in range(0, len(input_for_this_text['input_ids']), self.batch_size):
                batch = {
                    'input_ids': input_for_this_text['input_ids'][i:i+self.batch_size],
                    'attention_mask': input_for_this_text['attention_mask'][i:i+self.batch_size]
                }
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_output = self.model(**batch)
                output_for_this_text = torch.cat((output_for_this_text, batch_output.logits.cpu()))

        # Application de la fonction sigmoïde sur les logits pour obtenir des probabilités
        probs = torch.sigmoid(output_for_this_text)

        # Transformations des probabilités en classes prédites selon la méthode spécifiée
        if 'avgfen' in method:
            predictions = self.transform_to_classes_probabilities_to_classes_avg_fen(probs, float(method.split('*')[0]), get_dictionnary_with_confidence)
        elif 'avg' in method:
            predictions = self.transform_to_classes_probabilities_to_classes_avg(probs, float(method.split('*')[0]), get_dictionnary_with_confidence)
        elif 'maxfen' in method:
            predictions = self.transform_to_classes_probabilities_to_classes_max_fenetre(probs, float(method.split('*')[0]), get_dictionnary_with_confidence)
        elif 'max' in method:
            predictions = self.transform_to_classes_probabilities_to_classes_max(probs, float(method.split('*')[0]), get_dictionnary_with_confidence)
        elif 'threshold' in method:
            predictions = self.transform_to_classes_probabilities_to_classes_over_threshold(probs, float(method.split('*')[0]), get_dictionnary_with_confidence)
        elif 'thresholdfen' in method:
            predictions = self.transform_to_classes_probabilities_to_classes_over_threshold_fen(probs, float(method.split('*')[0]), get_dictionnary_with_confidence)
        else:
            raise ValueError(f"'{method}' is not implemented")

        # Explication des contributions des mots via SequenceClassificationExplainer
        explainer = SequenceClassificationExplainer(self.model, self.tokenizer)
        word_attributions = explainer(input_cleaned)

        return input_cleaned,predictions, word_attributions

    def transform_to_classes_probabilities_to_classes_avg(self, probs, coefficient_de_sureté, get_dictionnary_with_confidence):
        avg_pred = torch.mean(probs, dim=0) # On calcule la moyenne des probabilités pour chaque classe ( on transforme la liste de probabilité pour chaque classe pour chaque fenêtre en une liste de probabilité pour chaque classe pour le texte entier)
        avg_pred_value_mean = torch.mean(avg_pred) # On calcule la valeur moyenne des probabilités de la classe pour le texte entier
        dict_classes_prédites_avec_coefficients_de_confiance = {} # On initialise le dictionnaire qui va contenir les classes prédites avec leur coefficient de confiance si on le souhaite ( si get_dictionnary_with_confidence = True )
        list_classes_prédites_avec_coefficients_de_confiance = [] # On initialise la liste qui va contenir les classes prédites avec leur coefficient de confiance si on le souhaite ( si get_dictionnary_with_confidence = False )
        for i in range(len(avg_pred)): # On parcourt les classes
            if avg_pred[i] > coefficient_de_sureté*avg_pred_value_mean: # Si la probabilité de la classe est supérieure à un certain coefficient de sureté multiplié par la moyenne des probabilités
                dict_classes_prédites_avec_coefficients_de_confiance[i] = avg_pred[i] # Alors on ajoute au dictionnaire en mettant la classe en clé et la probabilité en valeur
                list_classes_prédites_avec_coefficients_de_confiance.append(i) # On ajoute la classe à la liste
        return dict_classes_prédites_avec_coefficients_de_confiance if get_dictionnary_with_confidence else list_classes_prédites_avec_coefficients_de_confiance # On renvoie le dictionnaire ou la liste en fonction de get_dictionnary_with_confidence
        
    def transform_to_classes_probabilities_to_classes_max(self, probs, coefficient_de_sureté, get_dictionnary_with_confidence):
        avg_pred = torch.mean(probs, dim=0) # On calcule la moyenne des probabilités pour chaque classe ( on transforme la liste de probabilité pour chaque classe pour chaque fenêtre en une liste de probabilité pour chaque classe pour le texte entier)
        max_pred = torch.max(avg_pred) # On calcule la valeur maximale des probabilités de la classe pour le texte entier
        dict_classes_prédites_avec_coefficients_de_confiance = {} # On initialise le dictionnaire qui va contenir les classes prédites avec leur coefficient de confiance si on le souhaite ( si get_dictionnary_with_confidence = True )
        list_classes_prédites_avec_coefficients_de_confiance = [] # On initialise la liste qui va contenir les classes prédites avec leur coefficient de confiance si on le souhaite ( si get_dictionnary_with_confidence = False )
        for i in range(len(avg_pred)): # On parcourt les classes
            if avg_pred[i] > coefficient_de_sureté*max_pred:
                dict_classes_prédites_avec_coefficients_de_confiance[i] = avg_pred[i]
                list_classes_prédites_avec_coefficients_de_confiance.append(i)
        return dict_classes_prédites_avec_coefficients_de_confiance if get_dictionnary_with_confidence else list_classes_prédites_avec_coefficients_de_confiance
    
    def transform_to_classes_probabilities_to_classes_max_fenetre(self, probs, coefficient_de_sureté, get_dictionnary_with_confidence):
        # A modifier pour éviter la répétition de code additionner le coefficient de confiance dans le dictionnaire
        for prob in probs:
            max_pred = torch.max(prob)
            dict_classes_prédites_avec_coefficients_de_confiance = {}
            list_classes_prédites_avec_coefficients_de_confiance = []
            for i in range(len(prob)):
                if prob[i] > coefficient_de_sureté*max_pred:
                    dict_classes_prédites_avec_coefficients_de_confiance[i] = prob[i]
                    list_classes_prédites_avec_coefficients_de_confiance.append(i)
        return list(set(dict_classes_prédites_avec_coefficients_de_confiance)) if get_dictionnary_with_confidence else list_classes_prédites_avec_coefficients_de_confiance
        
    def transform_to_classes_probabilities_to_classes_avg_fen(self, probs, coefficient_de_sureté, get_dictionnary_with_confidence):
        # Pareil
        for prob in probs:
            mean_pred = torch.mean(prob)
            dict_classes_prédites_avec_coefficients_de_confiance = {}
            list_classes_prédites_avec_coefficients_de_confiance = []
            for i in range(len(prob)):
                if prob[i] > coefficient_de_sureté*mean_pred:
                    dict_classes_prédites_avec_coefficients_de_confiance[i] = prob[i]
                    list_classes_prédites_avec_coefficients_de_confiance.append(i)
        return list(set(dict_classes_prédites_avec_coefficients_de_confiance)) if get_dictionnary_with_confidence else list_classes_prédites_avec_coefficients_de_confiance
    
    def transform_to_classes_probabilities_to_classes_over_threshold(self, probs, threshold, get_dictionnary_with_confidence):
        for prob in probs:
            dict_classes_prédites_avec_coefficients_de_confiance = {}
            list_classes_prédites_avec_coefficients_de_confiance = []
            for i in range(len(prob)):
                if prob[i] > threshold:
                    dict_classes_prédites_avec_coefficients_de_confiance[i] = prob[i]
                    list_classes_prédites_avec_coefficients_de_confiance.append(i)
        return list(set(dict_classes_prédites_avec_coefficients_de_confiance)) if get_dictionnary_with_confidence else list_classes_prédites_avec_coefficients_de_confiance
    
    def transform_to_classes_probabilities_to_classes_over_threshold_fen(self, probs, threshold, get_dictionnary_with_confidence):
        for prob in probs:
            dict_classes_prédites_avec_coefficients_de_confiance = {}
            list_classes_prédites_avec_coefficients_de_confiance = []
            for i in range(len(prob)):
                if prob[i] > threshold:
                    dict_classes_prédites_avec_coefficients_de_confiance[i] = prob[i]
                    list_classes_prédites_avec_coefficients_de_confiance.append(i)
        return list(set(dict_classes_prédites_avec_coefficients_de_confiance)) if get_dictionnary_with_confidence else list_classes_prédites_avec_coefficients_de_confiance
            
    def quick_predict(self, input):
        # Nettoyer le texte si nécessaire
        input_cleaned = get_text_from_html_doc(input) if '<' in input and '>' in input else input

        # Tokenisation et préparation de l'input pour le modèle
        input_for_this_text = self.tokenizer(input_cleaned, padding=True, truncation=True, max_length=512, return_tensors='pt')
        output_for_this_text = torch.tensor([])

        # Calcul des logits par le modèle sur les données tokenisées
        with torch.no_grad():
            for i in range(0, len(input_for_this_text['input_ids']), self.batch_size):
                batch = {
                    'input_ids': input_for_this_text['input_ids'][i:i+self.batch_size],
                    'attention_mask': input_for_this_text['attention_mask'][i:i+self.batch_size]
                }
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_output = self.model(**batch)
                output_for_this_text = torch.cat((output_for_this_text, batch_output.logits.cpu()))

        # Application de la fonction sigmoïde sur les logits pour obtenir des probabilités
        probs = torch.sigmoid(output_for_this_text)
        predictions = torch.argmax(probs, dim=1).tolist()  # Renvoie seulement l'indice de la classe la plus probable pour chaque entrée
        return [list_label_level_0[idx] for idx in predictions]
    
    
    def full_predict(self, input):
        # Nettoyer et préparer le texte
        input_cleaned = get_text_from_html_doc(input) if '<' in input and '>' in input else input
        input_cleaned = input_cleaned[:510]  # Truncate to the first 510 characters

        # Tokenisation et préparation de l'input pour le modèle
        input_for_this_text = self.tokenizer(input_cleaned, padding=True, truncation=True, max_length=512, return_tensors='pt')
        output_for_this_text = torch.tensor([])

        # Calcul des logits par le modèle sur les données tokenisées
        with torch.no_grad():
            for i in range(0, len(input_for_this_text['input_ids']), self.batch_size):
                batch = {
                    'input_ids': input_for_this_text['input_ids'][i:i+self.batch_size],
                    'attention_mask': input_for_this_text['attention_mask'][i:i+self.batch_size]
                }
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_output = self.model(**batch)
                output_for_this_text = torch.cat((output_for_this_text, batch_output.logits.cpu()))

        # Application de la fonction sigmoïde sur les logits pour obtenir des probabilités
        probs = torch.sigmoid(output_for_this_text)
        predictions = torch.argmax(probs, dim=1).tolist()

        # Utiliser SequenceClassificationExplainer pour obtenir des attributions        
        explainer = SequenceClassificationExplainer(self.model, self.tokenizer)
        word_attributions = explainer(input_cleaned)

        return input_cleaned, [list_label_level_0[idx] for idx in predictions], word_attributions




if __name__ == '__main__':
    predictor = PatentPredicterAI(level = 0)
    text = """< ! - -   E P O   < D
  P   n = " 1 4 " >   - - > < c l a i
  m   i d = " c - e n - 0 0 0 1 "   n u m =
  " 0 0 0 1 " > < c l a i m - t e x t > A   t u r
  b i n e   e n g i n e   a s s e m b l y   ( 1 0
  )   c o m p r i s i n g : < c l a i m - t e x t >
  a   t u r b o f a n   e n g i n e   ( 1 8 ) ;
  < / c l a i m - t e x t > < c l a i m - t e x t > a n  
  e n g i n e   c o w l i n g   ( 2 2 )   a r r a
  n g e d   e x t e r i o r l y   t o   t h
  e   t u r b o f a n   e n g i n e   ( 1 8 ) ;
  < / c l a i m - t e x t > < c l a i m - t e x t > a   s
  e t   o f   a t   l e a s t   t w
  o   g e n e r a t o r s   ( 1 4 )   f i x e
  d   r e l a t i v e   t o   t h e   t
  u r b o f a n   e n g i n e   ( 1 8 ) ,   w i t
  h i n   t h e   e n g i n e   c o w l i n
  g   ( 2 2 ) ,   1   w h e r e i n   t
  h e   s e t   o f   g e n e r a t o
  r   a e r o l i n e   d i m e n s i o n s   ( 3
  6 )   a r e   d e f i n e d   i n  
  a   r a d i a l   d i m e n s i o n   e x t e n
  d i n g   b e t w e e n   o u t e r m o s t   r
  a d i a l l y   d i s t a l   p o r t i o n s  
  o f   a   r e s p e c t i v e   g e n e r a t o
  r   h o u s i n g . < / c l a i m - t e x t > < / c>"""
    text_sans_html = data_processing.get_text_from_html_doc(text)
    predictions, attributions = predictor.predict(input=text_sans_html, method='3*avg')
    for pred in predictions:
        print(list_label_level_0[pred])
    for word, score in attributions:
        print(f"{word}: {score}")