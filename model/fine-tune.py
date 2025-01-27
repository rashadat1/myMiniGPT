import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPT import GPT
from utils import bpeDecode, bpeEncode
import torch
from datasets import load_dataset
from utils.utility import load_pretrained_weights
from config.GPTconfig import config

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = GPT(vocab_size=config['batch_size'],embed_size=config['embed_size'],context_length=config['context_length'],num_heads=config['num_heads'],num_layers=config['num_layers'])
model = model.to(device)

weights = load_pretrained_weights('/Users/tarikrashada/Projects/myMiniGPT/smaller_gpt_model_weights.pth')
torch.set_float32_matmul_precision('high')

model.load_state_dict(weights)
print('Model weights loaded successfully')

num_processes = 1
process_rank = 1

dataset = load_dataset('openwebtext',trust_remote_code=True)
with open('/Users/tarikrashada/Projects/myMiniGPT/data/input.txt', 'r') as file:
    text = file.read()
    
tokens = bpeEncode(text)
print(len(tokens))

def generate_batch(split,batch_size,context_length):
    """
    Generates a batch of input-output pairs from the streamed data
    Args:
        split: train or val split
        batch_size: Number of sequences per batch
        context_length: Number of tokens per sequence
    Returns:
        Tuple[Tensor, Tensor]: A batch of input (x) and target (y) sequences
    """
    batch = []
    if split == 'train':
        dataset = train_data
    else:
        dataset = val_data
    
    for example in dataset:
        #if current_batch_count % num_processes == process_rank: 
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
