from tokenizer import CustomTokenizer
from datasets import load_from_disk


def tokenize_dataset(path_to_dataset,
                    path_to_save, 
                    num_workers=24, 
                    truncate=False, 
                    max_length=512, 
                    min_length=3):
    
    english_tokenizer = CustomTokenizer(path_to_vocab="trained_tokenizers/vocab_en.json", truncate=truncate, max_length=max_length)
    french_tokenizer = CustomTokenizer(path_to_vocab="trained_tokenizers/vocab_fr.json", truncate=truncate, max_length=max_length)
    
    dataset = load_from_disk(path_to_dataset)
    
    def _tokenize_text(examples):
        
        english_text = examples["english_src"]
        french_text = examples["french_tgt"]
        
        src_ids = english_tokenizer.encode(english_text)
        tgt_ids = french_tokenizer.encode(french_text)
        
        batch = {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        }
        return batch
    
    tokenized_dataset = dataset.map(_tokenize_text, batched=True, num_proc=num_workers)
    tokenized_dataset = tokenized_dataset.remove_columns(["english_src", "french_tgt"])
    
    filter_func = lambda batch: [len(e) >= min_length for e in batch["tgt_ids"]]
    tokenized_dataset = tokenized_dataset.filter(filter_func, batched=True)
    
    print(tokenized_dataset)

    tokenized_dataset.save_to_disk(path_to_save)
    print("Tokenized dataset is successfully saved into the disk")
    

if __name__ == "__main__":
    path_to_dataset = "data/saved_data"
    path_to_save = "data/tokenized_dataset"
    tokenize_dataset(path_to_dataset=path_to_dataset, path_to_save=path_to_save)
    
    #push dataset into the hub: 
    tokenized_dataset = load_from_disk(dataset_path=path_to_save)
    tokenized_dataset.push_to_hub("ngia/tokenized-translation-en-fr")
    print("Tokenized dataset is successfully pushed into Hugging Face hub")
    
    
    
    

    