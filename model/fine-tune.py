import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPT import GPT
from utils import bpeDecode, bpeEncode
import torch
from datasets import load_dataset
from utils.utility import load_pretrained_weights

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

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = GPT(vocab_size=vocab_size,embed_size=embed_size,context_length=context_length,num_heads=num_heads,num_layers=num_layers)
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
