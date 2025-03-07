from huggingface_hub import list_repo_files, hf_hub_download
import os
from datasets import load_dataset,load_from_disk
from tokenize_dataset import tokenize_dataset

def get_files_from_HF(repo_id, folder_name, local_dir):
    
    files = list_repo_files(repo_id)
    
    folder_files = [f for f in files if f.startswith(folder_name)]
    
    for file in folder_files:
        file_path = hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
        print(f"Downloaded: {file_path} to {local_dir}")
        


if __name__ == "__main__":
    # dataset = load_dataset("FrancophonIA/english_french", split="train")
    # dataset = dataset.rename_column("english", "english_src")
    # dataset = dataset.rename_column("french", "french_tgt")
    # dataset = dataset.train_test_split(test_size=0.1)
    
    # path_to_dataset = "/home/ngam/Documents/translator-en-fr/data/saved_data"
    # path_to_tokenized_dataset = "/home/ngam/Documents/translator-en-fr/data/tokenized_dataset"
    
    # dataset.save_to_disk(dataset_dict_path=path_to_dataset)
    # tokenize_dataset(path_to_dataset=path_to_dataset, path_to_save=path_to_tokenized_dataset)
    # tokenized_dataset = load_from_disk(dataset_path=path_to_tokenized_dataset)
    
    # tokenized_dataset.push_to_hub("ngia/tokenized-translation-en-fr-small")
    # print("Tokenized dataset is successfully pushed into Hugging Face hub")
    
    dataset = load_dataset("ngia/tokenized-translation-en-fr-small")
    print(dataset)
    print(dataset["train"][0])
    
    