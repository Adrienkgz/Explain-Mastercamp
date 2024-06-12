from data_processing import get_all_datas
import torch
from listes_labels import list_label_level_0, list_label_level_1
print(len(list_label_level_0))
print(len(list_label_level_1))
torch.cuda.empty_cache()