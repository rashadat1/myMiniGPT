import torch
import tiktoken

def load_pretrained_weights(path):
    state_dict = torch.load('smaller_gpt_model_weights.pth', map_location=torch.device('mps'))
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    
    return new_state_dict

def bpeEncode(s: str) -> list:
    """Uses GPT2 BPE tokenizer to encode a string into a list of integers"""
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(s)
    return tokens

def bpeDecode(int_list: list) -> str:
    """Uses GPT2 BPE tokenizer to decode a list of integers into a string"""
    enc = tiktoken.get_encoding('gpt2')
    decoded = enc.decode(int_list)
    return decoded


