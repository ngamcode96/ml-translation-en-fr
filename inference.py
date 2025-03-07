import torch
from model import Transformer, TransformerConfig
from safetensors.torch import load_file
from tokenizer import CustomTokenizer

path_to_model_safetensors = "/home/ngam/Documents/translator-en-fr/checkpoints/model.safetensors"
path_to_src_tokenizer = "/home/ngam/Documents/translator-en-fr/trained_tokenizers/vocab_en.json"
path_to_tgt_tokenizer = "/home/ngam/Documents/translator-en-fr/trained_tokenizers/vocab_fr.json"

config = TransformerConfig(device='cpu', max_seq_length=512)
model = Transformer(config=config)

#load weights dict
weights_dict = load_file(filename=path_to_model_safetensors)
model.load_state_dict(weights_dict)
model.eval()

src_tokenizer = CustomTokenizer(path_to_vocab=path_to_src_tokenizer)
tgt_tokenizer = CustomTokenizer(path_to_vocab=path_to_tgt_tokenizer)


english_text = "How dare you speak like that?"

src_ids = torch.tensor(src_tokenizer.encode(english_text)).unsqueeze(0)

translated_ids = model.inference(src_ids=src_ids, tgt_start_id=tgt_tokenizer.eos_token_id, tgt_end_id=tgt_tokenizer.eos_token_id, max_seq_length=512)

translated_tokens = tgt_tokenizer.decode(translated_ids, skip_special_tokens=True)
print(translated_tokens)