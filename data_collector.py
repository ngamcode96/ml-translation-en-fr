
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tokenizer import CustomTokenizer
from model import *


max_src_ids = 0
max_tgt_ids = 0

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
            "src_input_ids": english_input_ids,   # Taille fixe : (self.max_length,)
            "src_pad_mask": src_pad_mask,
            "tgt_input_ids": french_input_ids,      # Taille fixe : (self.max_length,)
            "tgt_pad_mask": torch.cat([input_tgt_mask, torch.full((1,), 0, dtype=french_input_ids.dtype)]),
            "tgt_labels": torch.cat([label_tgt, torch.full((1,), -100, dtype=french_input_ids.dtype)])
        }

    

if __name__ == "__main__":
    
    path_to_tokenized_dataset = "data/tokenized_dataset" 
    path_to_english_tokenizer = "trained_tokenizers/vocab_en.json"
    path_to_french_tokenizer = "trained_tokenizers/vocab_fr.json"
    
    tokenized_dataset = load_from_disk(path_to_tokenized_dataset)
    
    english_tokenizer = CustomTokenizer(path_to_vocab=path_to_english_tokenizer)
    french_tokenizer = CustomTokenizer(path_to_vocab=path_to_french_tokenizer)
    
    train_dataset = DataCollector(dataset=tokenized_dataset["train"], english_tokenizer=english_tokenizer, french_tokenizer=french_tokenizer, max_length=64)
    test_dataset = DataCollector(dataset=tokenized_dataset["test"], english_tokenizer=english_tokenizer, french_tokenizer=french_tokenizer, max_length=64)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=4)
    
    from tqdm import tqdm
    config = TransformerConfig()
    src_embedding = SourceEmbedding(config=config)
    tgt_embedding = TargetEmbedding(config=config)
    position_encoding = PositionEncoding(config=config)
    attention = MultiheadAttention(config=config)
    feedforward = FeedForward(config=config)
    encoder = EncoderBlock(config=config)
    decoder = DecoderBlock(config=config)
    
    transformer = Transformer(config=config)
    
    print(transformer)
    
    for batch in tqdm(train_loader):
        src = batch["src_input_ids"]
        tgt = batch["tgt_input_ids"]
        x = src_embedding(src)
        y = tgt_embedding(tgt)
        
        x = position_encoding(x)
        y = position_encoding(y)
        encoded = encoder(x, attention_mask=batch["src_pad_mask"])
        decoded = decoder(x, y, batch["src_pad_mask"], batch["tgt_pad_mask"])
        
        output = transformer(src, tgt, batch["src_pad_mask"], batch["tgt_pad_mask"])
        
        print(output.shape)
    
        preds = transformer.inference(src_ids=src, tgt_start_id=1, tgt_end_id=2, max_seq_length=config.max_seq_length)
        translated_text = french_tokenizer.decode(preds)
        print(translated_text)

        
        
        
       
        # print(tgt_attention_mask.bool())
        break
            

        
    
    