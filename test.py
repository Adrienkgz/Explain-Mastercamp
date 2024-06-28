import pandas as pd
from data_processing import get_outputs

chunks = pd.read_csv("EFREI - LIPSTIP - 50k elements EPO.csv", chunksize=1000, low_memory=True)
dictionnaire_compteur_de_classe = {}
for i, chunk in enumerate(chunks):
    util_data = chunk[['CPC', 'claim', 'description']].to_numpy()
    outputs = get_outputs(util_data, 0, False)
    for k, output in enumerate(outputs):
        for j, classe in enumerate(output):
            if classe == 1:
                if j not in dictionnaire_compteur_de_classe:
                    dictionnaire_compteur_de_classe[j] = []
                dictionnaire_compteur_de_classe[j].append((i, k))
            
            
# on regarde quel est la classe la moins présente dans le dataset
min_classe = 0
min_classe_size = len(dictionnaire_compteur_de_classe[0])
for i in range(9):
    if len(dictionnaire_compteur_de_classe[i]) < min_classe_size:
        min_classe = i
        min_classe_size = len(dictionnaire_compteur_de_classe[i])
print(min_classe, min_classe_size)

# On crée un dataset équilibré qui met un nombre égal d'éléments de chaque classe
list_compteur_classes = [0 for _ in range(9)]
chunks = pd.read_csv("EFREI - LIPSTIP - 50k elements EPO.csv", chunksize=1000, low_memory=True)
balanced_list = []

for chunk in chunks:
    util_data = chunk[['CPC', 'claim', 'description']].to_numpy()
    outputs = get_outputs(util_data, 0, False)
    for k, output in enumerate(outputs):
        for j, classe in enumerate(output):
            if classe == 1:
                if list_compteur_classes[j] < min_classe_size:
                    new_dict_to_add = chunk.iloc[k].to_dict()
                    balanced_list.append(new_dict_to_add)
                    list_compteur_classes[j] += 1
                    break
    if min(list_compteur_classes) == min_classe_size:
        break
    print(list_compteur_classes)

# On mélange les éléments du dataset
balanced_df = pd.DataFrame(balanced_list).sample(frac=1).reset_index(drop=True)

# On sauvegarde le dataset
balanced_df.to_csv("EFREI - LIPSTIP - 50k elements EPO balanced.csv", index=False)


