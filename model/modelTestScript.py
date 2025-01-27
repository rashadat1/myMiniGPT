import sys,os
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from datasets import load_dataset
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utility import bpeDecode, bpeEncode
from GPT import GPT
from config.GPTconfig import config

# generate
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_processes = 1
process_rank = 1

model2 = GPT(vocab_size=config['vocab_size'],embed_size=config['embed_size'],context_length=config['context_length'],num_heads=config['num_heads'],num_layers=config['num_layers'])

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