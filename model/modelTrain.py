import os,sys
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from datasets import load_dataset
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utility import bpeDecode,bpeEncode,load_tokens
from GPT import GPT
from config.GPTconfig import config
from config.datasetconfig import config as dataset_config

np.int = np.int32
np.float = np.float64
np.bool = np.bool_
# Set the environment variable for PyTorch CUDA memory allocation configuration
# number of accumulation steps to be used in gradient accumulation idea
accumulation_steps = config['total_batch_size'] // (config['batch_size'] * config['context_length'])

print('loading training and validation datasets in streaming mode')

# train_data = load_dataset('allenai/c4','en',split='train',streaming=True,trust_remote_code=True)
# val_data = load_dataset('allenai/c4','en',split='validation',streaming=True,trust_remote_code=True)



torch.no_grad()
class DataLoader:
    def __init__(self, batch_size, context_length, num_processes, process_rank, split):
        # get the list of all shards in the shard directory
        shard_path = os.path.join(dataset_config['data_dir'],dataset_config['shard_dir'])
        shards = os.listdir(shard_path)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        
        shards = [os.path.join(shard_path, s) for s in shards]
        
        self.shards = shards 
        self.batch_size = batch_size
        self.context_length = context_length
        self.num_processes = num_processes
        self.process_rank = rank
        self.split = split
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        
        # initialize at shard 0
        self.curr_shard = 0
        self.tokens = load_tokens(self.shards[self.curr_shard])
        self.current_position = self.batch_size * self.context_length * self.process_rank
    
    def next_batch(self):
        buffer = self.tokens[self.current_position : self.current_position + self.batch_size * self.context_length + 1]
        # get the inputs and targets for each batch
        x = (buffer[:-1]).view(self.batch_size, self.context_length)
        y = (buffer[1:]).view(self.batch_size, self.context_length)
        self.current_position += self.batch_size * self.context_length * self.num_processes # move forward in the tokens
        if self.current_position + (self.batch_size * self.context_length * self.num_processes + 1) > len(self.tokens):
            self.curr_shard = (self.curr_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.curr_shard])
            self.current_position = self.batch_size * self.context_length * self.process_rank
        return x,y 

'''
def estimate_loss():
    # sum up individual token-level losses for all predictions in the batch
    # store this as total batch loss and then average over multiple batches
    out = {}
    # sets model to eval mode (disables features like dropout)
    model.eval()
    for split,data in zip(['train','val'],[train_data,val_data]):
        # for each split, initialize tensor to store loss values across eval_iters iterations
        losses = torch.zeros(config['eval_iters'])
        for k, (X,Y) in enumerate(generate_streaming_batch(split,config['batch_size'],config['context_length'],rank,world_size)):
            if k >= config['eval_iters']:
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
'''
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
    
model = GPT(vocab_size=config['vocab_size'],embed_size=config['embed_size'],context_length=config['context_length'],num_heads=config['num_heads'],num_layers=config['num_layers'])
# removes python interpreter and runs a pytorch compiler to optimize tensor operations
train_loader = DataLoader(batch_size=config['batch_size'],context_length=config['context_length'],num_processes=world_size,process_rank=rank,split="train")

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
    print(f"Total desired batch_size: {config['total_batch_size']}")
    print(f"=> calculated gradient accumulation steps: {accumulation_steps}")

print("I am GPU ", rank)

import math
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # warmup schedule that gpt3 used 
max_steps = config['max_iters']

# cosine decay learning rate scheduler
def learning_rate_schedule(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (max_lr - min_lr)




# training loop
import time
training_dict = {'step':[],'loss':[],'lr':[],'dt':[],'tokens_processed':[]}
for step in range(config['max_iters']):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    num_tokens_processed = 0
    for micro_step in range(accumulation_steps):
        xb, yb = train_loader.next_batch()
        xb, yb = xb.to(device), yb.to(device)
        B,T = xb.size()

        num_tokens_processed += B * T
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            loss,output = model(xb,yb)
        loss = loss / accumulation_steps # scale the loss to account for our grad accumulation
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
    torch.cuda.synchronize() # wait for GPU to finish working
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    training_dict['step'].append(step)
    training_dict['loss'].append(loss)
    training_dict['lr'].append(lr)
    training_dict['dt'].append(dt)
    training_dict['tokens_processed'].append(num_tokens_processed)
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt}s | num_tokens_processed: {num_tokens_processed:1f}")

torch.save(model.state_dict(),'smaller_gpt_model_weights2.pth')
print('Model weights saved successfully')

if ddp:
    
    dist.destroy_process_group()
    
import sys; sys.exit(0)