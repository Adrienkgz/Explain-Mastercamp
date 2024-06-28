from data_processing import get_all_datas
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

class PatentPredicterAI:
    def __init__(self, level, filepath_model = None, use_gpu = True, batch_size = 8, verbose = True):
        # Messages d'erreur
        if level not in [0, 1, 2, 3]: raise ValueError('level must be in [0, 1, 2, 3]')
        
        # Initialisation des paramètres 
        self.level = level # Le niveau de prédiction
        self.device = torch.device('cuda') if (use_gpu and torch.cuda.is_available()) else torch.device('cpu') # Paramètrage du device et si le modèle est chargé sur le processeur ou la carte graphique
        self.batch_size = batch_size # Taille des batchs de prédiction ( par exemple si batch_size = 8 alors lors de la prédiciton il prédira les éléments 8 par 8)
        self.filepath_model = f'saved_model_level_{level}' if filepath_model is None else filepath_model # Chemin du modèle à charger
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
        
        
    def predict(self, input, method, get_dictionnary_with_confidence = False):
        # Partie d'intialisation
        input_for_this_text = self.tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt') # On tokénise l'ensemble des sous blocs du texte
        output_for_this_text = torch.tensor([]) # On initialise un tuple vide pour stocker les prédictions
        
        # Partie où on demande au modèle de calculer les probabilités d'appartenance à chaque classe
        for i in range(0, len(input_for_this_text['input_ids']), self.batch_size):
            # Mise en forme du batch ( pas utile de comprendre )
            batch = {
                'input_ids': input_for_this_text['input_ids'][i:i+self.batch_size],
                'attention_mask': input_for_this_text['attention_mask'][i:i+self.batch_size]
            }
            
            with torch.no_grad(): # Méthode qui sert à améliorer la vitesse de calcul en disant à pytorch de ne pas garder en mémoire les gradients
                batch = {k: v.to(self.device) for k, v in batch.items()} # On envoie le batch de la mémoire au device ( soit la carte graphique soit le processeur)
                batch_output = self.model(**batch) # On fait passer les inputs dans le modèle pour en extraire les probabilités d'appartenance à chaque classe
                output_for_this_text = torch.cat((output_for_this_text, batch_output.logits.cpu())) # On concatène les prédictions et on repasse les prédictions sur le processeur pour libérer la mémoire de la carte graphique
            torch.cuda.empty_cache() # On libère la mémoire de la carte graphique pour éviter de surcharger la mémoire
        list_de_prediction_du_texte_pour_chaque_fenetre = torch.sigmoid(output_for_this_text) # On applique une fonction sigmoide pour avoir des probabilités entre 0 et 1
        
        # Partie où on transforme les probabilités en classes
        if 'avgfen' in method:
            liste_classes_prédites = self.transform_to_classes_probabilities_to_classes_avg_fen(probs = list_de_prediction_du_texte_pour_chaque_fenetre, coefficient_de_sureté = float(method.split('*')[0]), get_dictionnary_with_confidence = get_dictionnary_with_confidence)
        elif 'avg' in method:
            liste_classes_prédites = self.transform_to_classes_probabilities_to_classes_avg(probs = list_de_prediction_du_texte_pour_chaque_fenetre, coefficient_de_sureté = float(method.split('*')[0]), get_dictionnary_with_confidence = get_dictionnary_with_confidence)
        elif 'maxfen' in method:
            liste_classes_prédites = self.transform_to_classes_probabilities_to_classes_max_fenetre(probs = list_de_prediction_du_texte_pour_chaque_fenetre, coefficient_de_sureté = float(method.split('*')[0]), get_dictionnary_with_confidence = get_dictionnary_with_confidence)
        elif 'max' in method:
            liste_classes_prédites = self.transform_to_classes_probabilities_to_classes_max(probs = list_de_prediction_du_texte_pour_chaque_fenetre, coefficient_de_sureté = float(method.split('*')[0]), get_dictionnary_with_confidence = get_dictionnary_with_confidence)
        elif 'threshold' in method:
            liste_classes_prédites = self.transform_to_classes_probabilities_to_classes_over_threshold(probs = list_de_prediction_du_texte_pour_chaque_fenetre, threshold = float(method.split('*')[0]), get_dictionnary_with_confidence = get_dictionnary_with_confidence)
        elif 'thresholdfen' in method:
            liste_classes_prédites = self.transform_to_classes_probabilities_to_classes_over_threshold_fen(probs = list_de_prediction_du_texte_pour_chaque_fenetre, threshold = float(method.split('*')[0]), get_dictionnary_with_confidence = get_dictionnary_with_confidence)
        else:
            raise ValueError(f"'{method}' is not implemented")
        return liste_classes_prédites
        
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
            
        

if __name__ == '__main__':
    predictor = PatentPredicterAI(level = 0, filepath_model='saved_model')
    print('Test chunk 4')
    predictor.test(chunk=4, nbre_prediction=1000, use_fenetre_text=True, only_description=False, method='0.8*max', only_claim=True)
    





