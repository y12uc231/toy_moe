import os
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


#Tokenizer Builder
def build_and_save_tokenizer(file_path, tokenizer_path):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(special_tokens=["<unk>", "<sos>", "<eos>"])
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    tokenizer.train_from_iterator(lines, trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer



# Define tokenizer 
def get_tokenizer(train_file, tokenizer_fname):
    if os.path.exists(tokenizer_fname):
        print("Tokenizer found!") 
        tokenizer = Tokenizer.from_file(tokenizer_fname)
        tokenizer.end_token = "<eos>"
        tokenizer.end_token_id = tokenizer.token_to_id(tokenizer.end_token)
        tokenizer.generation_start_token = "<sos>"
        tokenizer.generation_start_token_id = tokenizer.token_to_id(tokenizer.generation_start_token)
    else:
        tokenizer = build_and_save_tokenizer(train_file, tokenizer_fname)
        tokenizer.end_token = "<eos>"
        tokenizer.end_token_id = tokenizer.token_to_id(tokenizer.end_token)
        tokenizer.generation_start_token = "<sos>"
        tokenizer.generation_start_token_id = tokenizer.token_to_id(tokenizer.generation_start_token)
    return tokenizer

def encode_text(tokenizer, text, max_length):
    encoded = tokenizer.encode(text)
    length_seq = len(encoded.ids[:])
    encoded_ids = encoded.ids[:]
    remaining = [tokenizer.end_token_id]
    repeats_needed = max_length - length_seq
    repeated_slice = remaining*repeats_needed
    encoded_ids.extend(repeated_slice)
    # print(f"Encoded sentence : {encoded_ids} \t", len(encoded_ids))
    return torch.tensor(encoded_ids), length_seq
