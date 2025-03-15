
import torch
from torch.utils.data import Dataset
from model import *


class DataCollector(Dataset):
    def __init__(self, dataset, english_tokenizer, french_tokenizer, max_length=512):
        self.dataset = dataset
        self.english_tokenizer = english_tokenizer
        self.french_tokenizer = french_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        english_input_ids = torch.tensor(self.dataset[index]['src_ids'])
        french_input_ids = torch.tensor(self.dataset[index]['tgt_ids'])

        # Padder manuellement avec torch.nn.functional.pad ou en utilisant torch.cat
        src_pad_token = self.english_tokenizer.pad_token_id
        tgt_pad_token = self.french_tokenizer.pad_token_id
        
        # Pour l'anglais
        if len(english_input_ids) < self.max_length:
            pad_length = self.max_length - len(english_input_ids)
            english_input_ids = torch.cat([english_input_ids, torch.full((pad_length,), src_pad_token, dtype=english_input_ids.dtype)])
        else:
            english_input_ids = english_input_ids[:self.max_length]

        # Pour le français
        if len(french_input_ids) < self.max_length:
            pad_length = self.max_length - len(french_input_ids)
            french_input_ids = torch.cat([french_input_ids, torch.full((pad_length,), tgt_pad_token, dtype=french_input_ids.dtype)])
        else:
            french_input_ids = french_input_ids[:self.max_length]

        # Créer les masques de padding
        src_pad_mask = (english_input_ids != src_pad_token)
        tgt_pad_mask = (french_input_ids != tgt_pad_token)

        # Pour les tâches de traduction ou LM, on décale la cible
        input_tgt = french_input_ids[:-1].clone()
        label_tgt = french_input_ids[1:].clone()
        input_tgt_mask = (input_tgt != tgt_pad_token)
        label_tgt[label_tgt == tgt_pad_token] = -100

        return {
            "src_input_ids": english_input_ids,   # Taille fixe: (self.max_length,)
            "src_pad_mask": src_pad_mask,
            "tgt_input_ids": french_input_ids,      # Taille fixe: (self.max_length,)
            "tgt_pad_mask": torch.cat([input_tgt_mask, torch.full((1,), 0, dtype=french_input_ids.dtype)]),
            "tgt_labels": torch.cat([label_tgt, torch.full((1,), -100, dtype=french_input_ids.dtype)])
        }

    

        
    
    