import torch
from model import Transformer, TransformerConfig
from safetensors.torch import load_file
from tokenizer import CustomTokenizer
from datasets import load_dataset
path_to_model_safetensors = "checkpoints/model.safetensors"
path_to_src_tokenizer = "trained_tokenizers/vocab_en.json"
path_to_tgt_tokenizer = "trained_tokenizers/vocab_fr.json"

config = TransformerConfig(device='cpu', max_seq_length=512)
model = Transformer(config=config)

#load weights dict
model.load_weights_from_checkpoints(path_to_model_safetensors)
model.eval()

src_tokenizer = CustomTokenizer(path_to_vocab=path_to_src_tokenizer)
tgt_tokenizer = CustomTokenizer(path_to_vocab=path_to_tgt_tokenizer)


english_text = "I'm very sick and i want to see a doctor."

src_ids = torch.tensor(src_tokenizer.encode(english_text)).unsqueeze(0)

translated_ids = model.inference(src_ids=src_ids, tgt_start_id=tgt_tokenizer.eos_token_id, tgt_end_id=tgt_tokenizer.eos_token_id, max_seq_length=512)
translated_tokens = tgt_tokenizer.decode(translated_ids, skip_special_tokens=True)
print(f"English: {english_text} \nFrench: {translated_tokens}")


dataset = load_dataset("bilalfaye/english-wolof-french-translation", split="train")
samples = dataset.shuffle().select(range(50))

for i in range(50):
    sample = samples[i]
    src_ids = torch.tensor(src_tokenizer.encode(sample["en"])).unsqueeze(0)
    output_ids = model.inference(src_ids=src_ids, tgt_start_id=tgt_tokenizer.eos_token_id, tgt_end_id=tgt_tokenizer.eos_token_id, max_seq_length=512)
    predicted_tokens = tgt_tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"English: {sample["en"]}")
    print(f"French (labels): {sample["fr"]}")
    print(f"French (predicted): {predicted_tokens}")
    print("--------------------------------\n\n")
    
    