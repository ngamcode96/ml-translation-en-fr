from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch

import os


def create_dataset(root_data_path, save_data_path, cache_data_path, test_size=0.01):
    
    list_datasets = []
    
    for directory in os.listdir(root_data_path):
        
        path_to_dir = os.path.join(root_data_path, directory)
        
        if os.path.isdir(path_to_dir):
            
            print(f"Processing: {path_to_dir}")
            
            english_text = None
            french_text = None
            
            for file_dir in os.listdir(path_to_dir):
                
                if file_dir.endswith(".en"):
                    english_text = os.path.join(path_to_dir, file_dir)
                
                if file_dir.endswith(".fr"):
                    french_text = os.path.join(path_to_dir, file_dir)
                
            if english_text is not None and french_text is not None:
                english_dataset = load_dataset("text", data_files=english_text, cache_dir=cache_data_path)["train"]
                french_dataset = load_dataset("text", data_files=french_text, cache_dir=cache_data_path)["train"]
                
                english_dataset = english_dataset.rename_column("text", "english_src")
                dataset = english_dataset.add_column("french_tgt", french_dataset["text"])
                
                list_datasets.append(dataset)
    
    
    hf_dataset =  concatenate_datasets(list_datasets)
    hf_dataset = hf_dataset.train_test_split(test_size=test_size)
        
    hf_dataset.save_to_disk(save_data_path)
    print(f"Dataset successfully saved in: {save_data_path}")


def push_dataset_into_hf_hub(save_data_path):
    dataset = load_from_disk(dataset_path=save_data_path)
    dataset = dataset.shuffle()
    dataset.push_to_hub(repo_id="ngia/translation-en-fr")
    print("Successfully pushed on Hugging Face Hub")
        

if __name__ == "__main__":
    root_data_path = "data/raw_data/"
    save_data_path = "data/saved_data/"
    cache_data_path = "data/cached_data/"
    
    create_dataset(root_data_path=root_data_path, save_data_path=save_data_path, cache_data_path=cache_data_path)
    dataset = load_from_disk(dataset_path=save_data_path)
    print(dataset["train"][10])
    
    push_dataset_into_hf_hub(save_data_path=save_data_path)
    