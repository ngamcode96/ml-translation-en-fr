from huggingface_hub import list_repo_files, hf_hub_download

def get_files_from_HF(repo_id, folder_name, local_dir):
    
    files = list_repo_files(repo_id)
    
    folder_files = [f for f in files if f.startswith(folder_name)]
    
    for file in folder_files:
        file_path = hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
        print(f"Downloaded: {file_path} to {local_dir}")
        


def get_file_FROM_HF(repo_id, file_path, local_dir):
    file_path = hf_hub_download(repo_id=repo_id, filename=file_path, local_dir=local_dir)
    return file_path

    

