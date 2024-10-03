import os,sys
import torch
import tiktoken
import torch.nn as nn
import math
from torch.nn import functional as F
from datasets import load_dataset
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utility import bpeDecode,bpeEncode
from GPT import GPT

np.int = np.int32
np.float = np.float64
np.bool = np.bool_
# Set the environment variable for PyTorch CUDA memory allocation configuration

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


print('loading training and validation datasets in streaming mode')

train_data = load_dataset('allenai/c4','en',split='train',streaming=True,trust_remote_code=True)
val_data = load_dataset('allenai/c4','en',split='validation',streaming=True,trust_remote_code=True)


torch.no_grad()
def estimate_loss():
    # sum up individual token-level losses for all predictions in the batch
    # store this as total batch loss and then average over multiple batches
    out = {}
    # sets model to eval mode (disables features like dropout)
    model.eval()
    for split,data in zip(['train','val'],[train_data,val_data]):
        # for each split, initialize tensor to store loss values across eval_iters iterations
        losses = torch.zeros(eval_iters)
        for k, (X,Y) in enumerate(generate_streaming_batch(split,batch_size,context_length,rank,world_size)):
            if k >= eval_iters:
                break
            # for each split generate eval_iters batches and calculate loss on each
            X,Y = X.to(device), Y.to(device)
            loss, output = model(X,Y)
            losses[k] = loss.item()
        # average loss across eval_iters iterations for each batch
        out[split] = losses.mean()
    # puts model back on training mode
    model.train()
    return out

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


ddp = int(os.environ.get('RANK',-1)) != -1
print(ddp)
if ddp:
    print('ddp was chosen')
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    print(f'set device to {device}')
    master_process = rank == 0 # this process will do logging and checkpointing
else:
    print('Not ddp')
    rank = 0
    local_rank = 0
    world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")


def generate_streaming_batch(split,batch_size,context_length,process_rank,num_processes,max_batches_per_epoch=None):
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
        
    current_batch_count = 0
    
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
        current_batch_count += 1


model = GPT(vocab_size=vocab_size,embed_size=embed_size,context_length=context_length,num_heads=num_heads,num_layers=num_layers)
# removes python interpreter and runs a pytorch compiler to optimize tensor operations


if ddp:
    model = model.to(rank)
    model = torch.compile(model)
    model = DDP(model,device_ids=[local_rank])
else:
    model.to(device)

raw_model = model.module if ddp else model

torch.set_float32_matmul_precision('high')
# create a Pytorch optimizer
optimizer = raw_model.configure_optimizer_weight_decay(weight_decay=0.1,device=device)

accumulation_steps = accumulation_steps // world_size
if master_process:
    print(f"Total desired batch_size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {accumulation_steps}")

print("I am GPU ", rank)

import math
max_lr = 6e-4
min_lr = max_lr * 0.01
warmup_steps = 715 # warmup schedule that gpt3 used 
max_steps = max_iters

def learning_rate_schedule(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (max_lr - min_lr)

import time
training_dict = {'step':[],'loss':[],'lr':[],'dt':[],'tokens_processed':[]}
for step in range(max_iters):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    num_tokens_processed = 0
    for micro_step in range(accumulation_steps):
        xb, yb = next(generate_streaming_batch('train',batch_size,context_length,process_rank=rank,num_processes=world_size))
        xb, yb = xb.to(device), yb.to(device)
        B,T = xb.size()

        num_tokens_processed += B * T
        with torch.autocast(device_type='mps',dtype=torch.bfloat16):
            loss,output = model(xb,yb)
        loss = loss / accumulation_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == accumulation_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    # determine and set the learning rate for the iteration
    lr = learning_rate_schedule(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    training_dict['step'].append(step)
    training_dict['loss'].append(loss)
    training_dict['lr'].append(lr)
    training_dict['dt'].append(dt)
    training_dict['tokens_processed'].append(num_tokens_processed)
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt}s | num_tokens_processed: {num_tokens_processed:1f}")

if ddp:
    dist.destroy_process_group()

torch.save(model.state_dict(),'smaller_gpt_model_weights2.pth')
print('Model weights saved successfully')
