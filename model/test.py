import sys,os
import torch
import tiktoken
import torch.nn as nn
import math
from torch.nn import functional as F
from datasets import load_dataset
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utility import bpeDecode, bpeEncode
from GPT import GPT

# model hyperparameters
# 524288 tokens per batch (that is per step in max_iters) so with max_iters = 7k
# then this model will see in total over 3B tokens
batch_size = 8
context_length = 1024 # length of input sequences
total_batch_size = 524288 # 2**19 close to .5M in number of tokens
accumulation_steps = total_batch_size // (batch_size * context_length)
learning_rate = 1e-6
max_iters = 10000
eval_interval = 500
eval_iters = 200
vocab_size = 50257 # 50257 with BPE but it turns out using 50304 - the nearest power of 64 is more efficient
embed_size = 768
# use dropout for regularization to fight overfitting
dropout = 0.1
num_layers = 12
num_heads = 12

# generate
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_processes = 1
process_rank = 1

model2 = GPT(vocab_size=vocab_size,embed_size=embed_size,context_length=context_length,num_heads=num_heads,num_layers=num_layers)

torch.no_grad()

state_dict = torch.load('smaller_gpt_model_weights.pth', map_location=torch.device('mps'))

# Create a new dictionary without the "module." and "_orig_mod." prefixes
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "").replace("_orig_mod.", "")
    new_state_dict[new_key] = v

torch.set_float32_matmul_precision('high')
print(device)
model2.load_state_dict(new_state_dict)
print('Model weights loaded successfully')

question = 'String of text to input'
model2 = model2.to(device)
encoded_question = torch.tensor(bpeEncode(question),dtype=torch.long,device=device).unsqueeze(0)
with torch.no_grad():
    print(bpeDecode(model2.generate(encoded_question,max_new_tokens=100)[0].tolist()))