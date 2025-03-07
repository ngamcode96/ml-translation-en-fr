from huggingface_hub import list_repo_files, hf_hub_download
import os

def get_files_from_HF(repo_id, folder_name, local_dir):
    
    files = list_repo_files(repo_id)
    
    folder_files = [f for f in files if f.startswith(folder_name)]
    
    for file in folder_files:
        file_path = hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
        print(f"Downloaded: {file_path} to {local_dir}")
        


if __name__ == "__main__":
    repo_id = "ngia/ml-translation-en-fr"
    folder_name = "checkpoint_80000"
    local_dir = "work_dir/Seq2Seq_Neural_Machine_Translation/checkpoint_80000"
    
    os.makedirs(name=local_dir, exist_ok=True)
    
    get_files_from_HF(repo_id=repo_id, folder_name=folder_name, local_dir=local_dir)