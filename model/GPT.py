import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.embed_size % config.num_heads == 0
        # key, query value transformation for all heads in a batch
        self.c_attn = nn.Linear(config.embed_size,3 * config.embed_size)
        # projection of attention output
        self.c_proj = nn.Linear(config.embed_size,config.embed_size)
        self.num_heads = config.num_heads
        self.embed_size = config.embed_size
        
        self.register_buffer('tril',torch.tril(torch.ones(config.context_length,config.context_length)).view(1,1,config.context_length,config.context_length))
        
    def forward(self,x):
        B,T,C = x.size() # batch_size, context length, embedding dimensionality
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.embed_size,dim=2)
        # each has shape (batch_size, context_length, num_heads, embed dimension / num_heads)
        q = q.view(B,T,self.num_heads,C // self.num_heads).transpose(1,2)
        k = k.view(B,T,self.num_heads,C // self.num_heads).transpose(1,2)
        v = v.view(B,T,self.num_heads,C // self.num_heads).transpose(1,2)
        
        attnScore = q @ k.transpose(-2,-1)
        attnScore = attnScore / math.sqrt(k.size(-1))
        attnScore = attnScore.masked_fill(self.trill[:,:,:T,:T] == 0, float('-inf'))
        attnWeight = F.softmax(attnScore,dim=-1)
        
        attnOutput = attnWeight @ v
        attnOutput = attnOutput.transpose(1,2).contiguous().view(B,T,C) # re-assemble head outputs side by side
        # output of projection
        y = self.c_proj(attnOutput)
        return y        
                
class FeedForwardNetwork(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.gelu = nn.GELU(approximate='tanh')

class TransformerBlock(nn.Module)


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    embed_size: int = 768
    context_length: int = 1024
    

class GPT2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config