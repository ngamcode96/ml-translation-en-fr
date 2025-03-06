from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import glob
import os
import argparse



def train_tokenizer(path_to_data, lang):
    
    special_token_dict = {
    "pad_token" : "[PAD]",
    "start_token": "[BOS]",
    "end_token": "[EOS]",
    "unknown_token": "[UNK]"
    }
    
    tokenizer = Tokenizer(WordPiece(unk_token=special_token_dict["unknown_token"]))
    tokenizer.normalizer = normalizers.Sequence([NFC(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()
    
    files = []
    
    if lang == "fr":
        print("---------Training French Tokenizer--------------")
        files = glob.glob(os.path.join(path_to_data, "**/*.fr"))
        
    elif lang == "en":
        print("---------Training English Tokenizer--------------")
        files = glob.glob(os.path.join(path_to_data, "**/*.en"))
        
    trainer = WordPieceTrainer(vocab_size=32000, special_tokens=list(special_token_dict.values()))
    tokenizer.train(files, trainer)
    tokenizer.save(f"trained_tokenizers/vocab_{lang}.json")
    print(f"Tokenizer is successfully saved into trained_tokenizers/vocab_{lang}.json")
    


class CustomTokenizer:
    
    def __init__(self, path_to_vocab, truncate=False, max_length=512):
        self.path_to_vocab = path_to_vocab
        self.truncate = truncate
        self.max_length = max_length
        self.tokenizer = self.config_tokenizer()
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        self.pad_token = "[PAD]"
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        
        self.bos_token = "[BOS]"
        self.bos_token_id = self.tokenizer.token_to_id("[BOS]")
        
        self.eos_token = "[EOS]"
        self.eos_token_id = self.tokenizer.token_to_id("[EOS]")
        
        self.unk_token = "[UNK]"
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")
        
        self.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id)
            ]
        )
        
        if self.truncate:
            self.max_length = max_length - self.post_processor.num_special_tokens_to_add(is_pair=False)
        
    
    def config_tokenizer(self):
        tokenizer = Tokenizer.from_file(self.path_to_vocab)
        tokenizer.decoder = decoders.WordPiece()
        return tokenizer
    
    
    def encode(self, input):
        
        def _parse_process_tokenized(tokenized):
            if self.truncate:
                tokenized.truncate(self.max_length, direction="right")
            tokenized = self.post_processor.process(tokenized)
            return tokenized.ids

        if isinstance(input, str):
            tokenized = self.tokenizer.encode(input)
            tokenized = _parse_process_tokenized(tokenized)
            
        if isinstance(input, (list, tuple)):
            tokenized = self.tokenizer.encode_batch(input)
            tokenized = [_parse_process_tokenized(t) for t in tokenized]
        
        return tokenized
    
    def decode(self, input, skip_special_tokens=True):
        if isinstance(input, list):
            if all(isinstance(item, list) for item in input):
                decoded = self.tokenizer.decode_batch(input, skip_special_tokens=skip_special_tokens)
            elif all(isinstance(item, int) for item in input):
                decoded = self.tokenizer.decode(input, skip_special_tokens=skip_special_tokens)
        
        return decoded
        
    
if __name__ == "__main__":
   
    path_to_data_root = "/home/ngam/Documents/translator-en-fr/data/raw_data"
    #replace False by True if you want to train a new tokenizer
    if False:
        train_tokenizer(path_to_data_root, lang='fr')
        train_tokenizer(path_to_data_root, lang='en')
