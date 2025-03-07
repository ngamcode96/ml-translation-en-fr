import os
import numpy as np
import torch

from model import Transformer, TransformerConfig
from data_collector import DataCollector
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from transformers import get_scheduler
from tokenizer import CustomTokenizer
from tqdm import tqdm
from accelerate import Accelerator
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"



# MODEL CONFIG
src_vocab_size: int = 32000
tgt_vocab_size: int = 32000
max_seq_length: int = 512
d_model: int = 512
num_heads: int = 8
num_encoder_layers: int = 6
num_decoder_layers: int = 6
dropout_p: float = 0.1
dff: int = 2048

config = TransformerConfig(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    max_seq_length=max_seq_length,
    d_model=d_model,
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dropout_p=0.1,
    dff=dff
)


# TOKENIZER CONFIG
src_tokenizer_path = "trained_tokenizers/vocab_en.json"
tgt_tokenizer_path = "trained_tokenizers/vocab_fr.json"

src_tokenizer = CustomTokenizer(path_to_vocab=src_tokenizer_path, max_length=config.max_seq_length)
tgt_tokenizer = CustomTokenizer(path_to_vocab=tgt_tokenizer_path, max_length=config.max_seq_length)


# DATALOADER CONFIG
path_to_data = "data/tokenized_dataset"
batch_size = 64
gradient_accumulation_steps = 2
# num_workers = 4

# Training Config
learning_rate = 1e-4
training_steps = 150000 
warmup_steps = 2000
scheduler_type = "cosine"
evaluation_steps = 10000
bias_norm_weight_decay = False
weight_decay = 0.001
betas = (0.9, 0.98)
adam_eps = 1e-6


#Logging Config
working_directory = "work_dir"
experiment_name = "Seq2Seq_Neural_Machine_Translation"
logging_interval = 1

#Resume from checkpoint
resume_from_checkpoint = None



#Prepare Accelerator
path_to_experiment = os.path.join(working_directory, experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb")

accelerator.init_trackers(experiment_name)

#config model device
config.device = accelerator.device


# Prepare Dataloaders
# dataset = load_from_disk(dataset_path=path_to_data)
dataset = load_dataset("ngia/tokenized-translation-en-fr")
accelerator.print("Dataset:", dataset)
min_batch_size = batch_size // gradient_accumulation_steps
train_dataset = DataCollector(dataset=dataset["train"], english_tokenizer=src_tokenizer, french_tokenizer=tgt_tokenizer, max_length=config.max_seq_length)
test_dataset = DataCollector(dataset=dataset["test"], english_tokenizer=tgt_tokenizer, french_tokenizer=tgt_tokenizer, max_length=config.max_seq_length)
    
train_loader = DataLoader(dataset=train_dataset, batch_size=min_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=min_batch_size, shuffle=False)


# Prepare model
model = Transformer(config=config)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of trainable parameters:", params)


# Prepare Optimizer
optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=learning_rate,
                                  betas=betas,
                                  eps=adam_eps)


# Define scheduler
scheduler = get_scheduler(
    name=scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=training_steps
)

# Define Loss Function
loss_fn = torch.nn.CrossEntropyLoss()


### Define a Sample Sentence for Testing ###
src_ids = torch.tensor(src_tokenizer.encode("I want to learn how to training a machine translation")).unsqueeze(0)


model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler
)

# model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
accelerator.register_for_checkpointing(scheduler)

#RESUME FROM CHECKPOINT
if resume_from_checkpoint is not None:

    ### Grab path to checkpoint ###
    path_to_checkpoint = os.path.join(path_to_experiment, resume_from_checkpoint)
    
    ### Load checkpoint on main process first, recommended here: (https://huggingface.co/docs/accelerate/en/concept_guides/deferring_execution) ###
    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_steps = int(resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")
else:
    completed_steps = 0
    

train = True
progress_bar = tqdm(range(completed_steps, training_steps), disable= not accelerator.is_local_main_process)


while train:
    accumulate_steps = 0
    accumulate_loss = 0
    accuracy = 0
    
    for batch in trainloader:
        src_input_ids = batch["src_input_ids"].to(accelerator.device)
        src_pad_mask = batch["src_pad_mask"].to(accelerator.device)
        tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
        tgt_pad_mask = batch["tgt_pad_mask"].to(accelerator.device)
        tgt_labels = batch["tgt_labels"].to(accelerator.device)
        
        model_output = model(
            src_input_ids,
            tgt_input_ids,
            src_pad_mask,
            tgt_pad_mask
        )
        
        model_output = model_output.flatten(0,1)
        tgt_labels = tgt_labels.flatten()
        loss = loss_fn(model_output, tgt_labels)
        
        ### Scale Loss and Accumulate ###
        loss = loss / gradient_accumulation_steps
        accumulate_loss += loss
        
        ### Compute Gradients ###
        accelerator.backward(loss)
        
        ### Compute Accuracy (ignoring -100 padding labels) ###
        model_output = model_output.argmax(axis=-1)
        mask = (tgt_labels != -100)
        output = model_output[mask]
        tgt_outputs = tgt_labels[mask]
        acc = (output == tgt_outputs).sum() / len(output)  
        accuracy += acc / gradient_accumulation_steps 
        
        ### Iterate Accumulation ###
        accumulate_steps += 1
        
        if accumulate_steps % gradient_accumulation_steps == 0:

            ### Clip and Update Model ###
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
            ### Log Results ###
            if completed_steps % logging_interval == 0:

                accumulate_loss = accumulate_loss.detach()
                accuracy = accuracy.detach()

                if accelerator.num_processes > 1:
                        accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))
                        accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))

                log = {"train_loss": accumulate_loss,
                        "training_acc": accuracy,
                        "learning_rate": scheduler.get_last_lr()[0]}
                
                accelerator.log(log, step=completed_steps)
                logging_string = f"[{completed_steps}/{training_steps}] Training Loss: {accumulate_loss} | Training Acc: {accuracy}"
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                    
            
            if completed_steps % evaluation_steps == 0:
                model.eval() 
                print("Evaluating!")
                
                test_losses = []
                test_accs = []

                for batch in tqdm(testloader, disable=not accelerator.is_main_process):

                    src_input_ids = batch["src_input_ids"].to(accelerator.device)
                    src_pad_mask = batch["src_pad_mask"].to(accelerator.device)
                    tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                    tgt_pad_mask = batch["tgt_pad_mask"].to(accelerator.device)
                    tgt_labels = batch["tgt_labels"].to(accelerator.device)

                    with torch.inference_mode():
                        model_output = model(src_input_ids, 
                                    tgt_input_ids, 
                                    src_pad_mask, 
                                    tgt_pad_mask)
                    
                    ### Flatten for Loss ###
                    model_output = model_output.flatten(0,1)
                    tgt_labels = tgt_labels.flatten()
                    
                    ### Compute Loss ###
                    loss = loss_fn(model_output, tgt_labels)

                    ### Compute Accuracy (make sure to ignore -100 targets) ###
                    model_output = model_output.argmax(axis=-1)
                    mask = (tgt_labels != -100)
                    model_output = model_output[mask]
                    tgt_labels = tgt_labels[mask]
                    accuracy = (model_output == tgt_labels).sum() / len(model_output)   

                    ### Store Results ###
                    loss = loss.detach()
                    accuracy = accuracy.detach()

                    if accelerator.num_processes > 1:
                        loss = torch.mean(accelerator.gather_for_metrics(loss))
                        accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))
            
                    ### Store Metrics ###
                    test_losses.append(loss.item())
                    test_accs.append(accuracy.item())

                test_loss = np.mean(test_losses)
                test_acc = np.mean(test_accs)

                log = {"test_loss": test_loss,
                        "test_acc": test_acc}   
                
                logging_string = f"Testing Loss: {test_loss} | Testing Acc: {test_acc}"
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
                
                ### Log and Save Model ###
                accelerator.log(log, step=completed_steps)
                accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{completed_steps}"))
                
                ### Testing Sentence ###
                if accelerator.is_main_process:
                    src_ids = src_ids.to(accelerator.device)
                    unrwapped = accelerator.unwrap_model(model)
                    translated = unrwapped.inference(src_ids, 
                                                    tgt_start_id=tgt_tokenizer.bos_token_id,
                                                    tgt_end_id=tgt_tokenizer.eos_token_id, max_seq_length=config.max_seq_length)
                    
                    translated = tgt_tokenizer.decode(translated, skip_special_tokens=False)

                    if accelerator.is_main_process:
                        progress_bar.write(f"Translation: {translated}")

                model.train()

            if completed_steps >= training_steps:
                train = False
                accelerator.save_state(os.path.join(path_to_experiment, f"final_checkpoint"))
                break
            
            ### Iterate Completed Steps ###
            completed_steps += 1
            progress_bar.update(1)

            ### Reset Accumulated Variables ###
            accumulate_loss = 0
            accuracy = 0
        
        
        
        
accelerator.end_training()
    