from data_processing import get_all_datas
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch
from tqdm import tqdm

#### Parametres ####

BATCH_SIZE = 8
LEVEL = 1
FILEPATH_MODEL = '/results/checkpoint-4800'

tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')

inputs, outputs = get_all_datas('EFREI - LIPSTIP - 50k elements EPO.csv', 1,24, 1, use_fenetre_text=False, flatten = False, only_description=False)

# Load your model
loaded_model = BertForSequenceClassification.from_pretrained(f'folder_model_lvl_{LEVEL}/{FILEPATH_MODEL}')

# Set device to CPU
device = torch.device('cuda')
print(f"device: {device}")
loaded_model.to(device)

nbre_prediction = 100

# Predict new papers
list_pred_output = []
for text in tqdm(inputs[:nbre_prediction], desc='Prediction en cours'):
    new_inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    new_inputs = new_inputs
    new_outputs = torch.tensor([])
    for i in range(0, len(new_inputs['input_ids']), BATCH_SIZE):
        # On fait des prédictions par batch pour éviter de saturer la mémoire
        batch = {
            'input_ids': new_inputs['input_ids'][i:i+BATCH_SIZE],
            'attention_mask': new_inputs['attention_mask'][i:i+BATCH_SIZE]
        }
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_output = loaded_model(**batch)
            # On concatène les prédictions
            new_outputs = torch.cat((new_outputs, batch_output.logits.cpu()))
        torch.cuda.empty_cache()
    list_de_prediction_du_texte_pour_chaque_fenetre = torch.sigmoid(new_outputs)
    # On additionne tout les scores de prédictions de tout les éléments du texte 
    avg_pred = torch.mean(list_de_prediction_du_texte_pour_chaque_fenetre, dim=0)
    list_pred_output.append(avg_pred)

compteur, compteur_parfait, total_diff_len = 0, 0, 0
for i in range(nbre_prediction):
    true_output = outputs[i]
    new_prediction = list_pred_output[i]
    avg_pred = torch.mean(new_prediction)
    classe_predite, classe_vraies = [], []
    bon_point = 0
    for j, (true_value, new_value) in enumerate(zip(true_output, new_prediction)):
        if new_value > 3*avg_pred:
            classe_predite.append(j)
            print (f"j : {j}, true_value : {true_value}, value_pred : {new_value}")
            if true_value > 0.0:
                bon_point += 1
        elif true_value > 0.0:
            print(f"#### vrai : {j}, prédit : {new_value}")
        if true_value > 0.0:
            classe_vraies.append(j)
    diff_len = abs(len(classe_predite) - len(classe_vraies))
    print(f"avg_pred : {avg_pred}")
    print(f"longueur classe prédite : {len(classe_predite)}")
    print(f'diff long predit et long vrai : {diff_len}')
    print(f"classe prédite : {classe_predite}")
    print(f"classes vraies : {classe_vraies}")
    print(f'bon point : {bon_point} / {len(classe_vraies)}')
    if bon_point > 0:
        compteur += 1
    if bon_point == len(classe_vraies):
        compteur_parfait += 1
    total_diff_len += diff_len
    print(f"\n---------------n")
print(f"total_diff_len moyen: {total_diff_len/nbre_prediction}")
print(f"compteur : {compteur/nbre_prediction}")
print(f"compteur parfait : {compteur_parfait/nbre_prediction}")





