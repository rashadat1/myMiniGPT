import torch
import tiktoken
import numpy as np

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
    return enc

def bpeDecode(int_list: list) -> str:
    """Uses GPT2 BPE tokenizer to decode a list of integers into a string"""
    enc = tiktoken.get_encoding('gpt2')
    decoded = enc.decode(int_list)
    return decoded

def load_tokens(filename):
    arr = np.load(filename)
    arr_tensor = torch.tensor(arr, dtype=torch.long)
    return arr_tensor

def generate_streaming_batch(split,train_data,batch_size,context_length,process_rank,num_processes,max_batches_per_epoch=None):
    """
    Generates a batch of input-output pairs from the streamed data
    Args:
        split: train or val split
        batch_size: Number of sequences per batch
        context_length: Number of tokens per sequence
    Returns:
        Tuple[Tensor, Tensor]: A batch of input (x) and target (y) sequences
    """
    if split == 'train':
        dataset = train_data.shard(num_shards=num_processes, index = process_rank)
    
    batch = []
    current_batch_count = 0
    
    for example in dataset:
        # each process picks its data portion
        text = example['text']
        tokens = bpeEncode(text)
        batch.extend(tokens)
        if len(batch) >= batch_size * context_length + 1:
            x = torch.tensor(batch[:batch_size * context_length], dtype=torch.long).view(batch_size, context_length)
            y = torch.tensor(batch[1:batch_size * context_length + 1], dtype=torch.long).view(batch_size, context_length)
            # advance the position in the tensor 
            yield x, y
            # discard the used tokens and move forward
            batch = batch[batch_size * context_length + 1:]
        current_batch_count += 1
