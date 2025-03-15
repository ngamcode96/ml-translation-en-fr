import torch
from tokenizer import CustomTokenizer
from model import Transformer, TransformerConfig
import gradio as gr


# load tokenizers
path_to_src_tokenizer = "trained_tokenizers/vocab_en.json"
path_to_tgt_tokenizer = "trained_tokenizers/vocab_fr.json"

src_tokenizer = CustomTokenizer(path_to_vocab=path_to_src_tokenizer)
tgt_tokenizer = CustomTokenizer(path_to_vocab=path_to_tgt_tokenizer)



#load model
config = TransformerConfig(max_seq_length=512)
model = Transformer(config=config)

path_to_checkpoints = "checkpoints/model.safetensors"
model.load_weights_from_checkpoints(path_to_checkpoints=path_to_checkpoints)
model.eval()


def translate(input_text, skip_special_tokens=True):
    src_ids = torch.tensor(src_tokenizer.encode(input_text)).unsqueeze(0)
    output_ids = model.inference(src_ids=src_ids, tgt_start_id=tgt_tokenizer.bos_token_id, tgt_end_id=tgt_tokenizer.eos_token_id, max_seq_length=512)
    output_tokens = tgt_tokenizer.decode(input=output_ids, skip_special_tokens=skip_special_tokens)
    return output_tokens



with gr.Blocks() as demo:
    gr.Markdown("## Traduction Anglais → Français")
    
    with gr.Row():
        texte_input = gr.Textbox(label="Texte en anglais", lines=4)
        texte_output = gr.Textbox(label="Texte traduit (Français)",lines=4, interactive=False)
    
    bouton = gr.Button("Traduire")
    bouton.click(translate, inputs=texte_input, outputs=texte_output)

demo.launch()

