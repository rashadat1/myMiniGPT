import os
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import numpy as np

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

class AttentionHead(nn.Module):
    def __init__(self,embed_size,head_size,context_length):
        """Initialize the attention head with key, query, value vectors and create causal masking"""
        super().__init__()
        self.embed_size = embed_size
        self.head_size = head_size
        self.context_length = context_length
        self.key = nn.Linear(self.embed_size,self.head_size, bias=False)
        self.query = nn.Linear(self.embed_size,self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size,self.head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(self.context_length, self.context_length)))

    def forward(self, input):
        B,T,C = input.shape
        k = self.key(input)
        q = self.query(input)
        v = self.value(input)

        attnScores = k @ q.transpose(-2,-1) # Shape (batch_size, context_length, head_size) x (batch_size, head_size, context_length)
        attnScores = attnScores / math.sqrt(self.head_size)
        # add causal masking so future doesn't influence the past to make this a decoder block
        attnScores = attnScores.masked_fill_(self.tril[:T, :T] == 0, float('-inf'))
        # (batch_size, context_length, context_length)
        attnWeight = F.softmax(attnScores, dim=-1)
        attnWeight = self.dropout(attnWeight)
        attnOutput = attnWeight @ v # shape (batch_size, context_length, head_size)
        return attnOutput
    
    
class MultiHeadedAttention(nn.Module):
    def __init__(self,embed_size,num_heads,head_size,context_length):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.context_length = context_length
        self.heads = nn.ModuleList([AttentionHead(self.embed_size,self.head_size,self.context_length) for _ in range(self.num_heads)])
        # Linear projection of the output of the self attention layer
        self.proj = nn.Linear(self.embed_size,self.embed_size)
        self.proj.GPT_SCAL_INIT = 1
        self.dropout = nn.Dropout(dropout)

    def forward(self,input):
        out = torch.cat([h(input) for h in self.heads],dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
    
class FeedForward(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.ffd = nn.Sequential(
            nn.Linear(self.embed_size,self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size,self.embed_size),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.ffd(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,embed_size,num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        head_size = self.embed_size // self.num_heads
        self.ffwd = FeedForward(self.embed_size)
        self.attnMulti = MultiHeadedAttention(self.embed_size,self.num_heads,head_size,context_length)
        self.LN1 = nn.LayerNorm(self.embed_size)
        self.LN2 = nn.LayerNorm(self.embed_size)
    # residual connection implemented in the transformer block
    def forward(self,x):
        x = x + self.attnMulti(self.LN1(x))
        x = x + self.ffwd(self.LN2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self,vocab_size,embed_size,context_length,num_heads,num_layers):
        """Initialize the language model with token and positional embeddings as well as a linear layer"""
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_size) # embedding layer for token embeddings
        self.pos_embedding = nn.Embedding(self.context_length,self.embed_size) # positional embedding
        # self.fc = nn.Linear(self.embed_size,self.vocab_size) # linear layer to map to vocab size
        self.LN = nn.LayerNorm(self.embed_size)
        self.transformerBlocks = nn.Sequential(*[TransformerBlock(self.embed_size,self.num_heads) for _ in range(self.num_layers)])
        # weight tying scheme for parameter reduction and generalization
        # if a word's embedding is tied to its likelihood in output layer
        # this encourages the model to learn better representations
        #self.token_embedding.weight = self.fc.weight

        # initialize parameters based on initialization used in GPT2 paper
        self.apply(self._init_weights)

    def _init_weights(self,module):
        """Weight initialization using std deviation 0.02"""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module,'GPT_SCALE_INIT'):
                std *= (2 * self.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def configure_optimizer_weight_decay(self,weight_decay,device):
        decay = set()
        no_decay = set()
        self.weight_decay = weight_decay
        weight_decay_modules = (torch.nn.Linear,)
        non_weight_decay_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn,p in m.named_parameters():
                fpn = '%s.%s' % (mn,pn) if mn else pn # get full parameter name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, weight_decay_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, non_weight_decay_modules):
                    no_decay.add(fpn)
                param_dict = {pn: p for pn, p in self.named_parameters()}
        # make sure we considered every parameter and no parameter was double added
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        # create optim groups - any parameters that are 2d will be weight decayed
        # this will include weight tensors but not biases and layer norm weights
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},

        ]
        optimizer = torch.optim.AdamW(optim_groups,lr=1e-5,betas=(0.9,0.95),eps=1e-8)
        return optimizer

    def forward(self,input,targets=None):
        """
        Forward pass for the language model.

        Args:
            input (Tensor): Input tensor of token indices.
            targets (Tensor, optional): Target tokens for calculating loss. Defaults to None.

        Returns:
            Tuple[Tensor or None, Tensor]: The loss (if targets are provided) and the model output.
        """
        # T is the context length
        B,T = input.shape
        # create token and position embeddings
        tok_emb = self.token_embedding(input) # <- shape is (batch_size,context_length,embed_size) where embed_size is the length of the embedding vectors
        pos_emb = self.pos_embedding(torch.arange(T, device=device) % self.context_length) # integers from 0 to context_length-1, each is embedded to get context_length x embed_dim tensor
        # positional embeddings get broadcasted across batches
        x = tok_emb + pos_emb # (batch_size,context_length,embed_size) dimensional tensors
        x = self.transformerBlocks(x)
        x = self.LN(x)
        output = torch.matmul(x,self.token_embedding.weight.T) # <- (batch_size,context_length,vocab_size) or (B,T,vocab_size)

        if targets == None:
            loss = None

        else:
            B,T,C = output.shape
            output = output.view(B*T,C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(output,targets)

        return loss,output

    def generate(self,input,max_new_tokens):
        i = 0
        while i < max_new_tokens:
            # crop input to last context_length of tokens
            inp_cond = input[:,-self.context_length:]
            # call forward method
            _, output = self(inp_cond)
            # extract the embedding of the last token
            last_hidden_state = output[:,-1,:]
            # convert this embedding vector into a probability distribution over the vocabulary
            distribution = F.softmax(last_hidden_state, dim=-1)
            # sample from distribution to get new token index
            newToken = torch.multinomial(distribution,num_samples=1)
            input = torch.cat([input,newToken],dim=1)
            i += 1
        return input