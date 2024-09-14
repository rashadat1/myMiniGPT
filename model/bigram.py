import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

with open('/Users/tarikrashada/Projects/myMiniGPT/data/input.txt','r') as file:
    text = file.read()

chars = sorted(list(set(text)))

# model hyperparameters
batch_size = 32
context_length = 16 # length of input sequences
learning_rate = 1e-2
max_iters = 3000
eval_interval = 300
eval_iters = 200
vocab_size = len(chars)

# create character-to-integer and integer-to-character mappings for tokenization
char_to_int_map = {char : i for i,char in enumerate(chars)}
int_to_char_map = {i : char for i,char in enumerate(chars)}

def encode(s: str) -> list:
    """Encodes a string into a list of integers"""
    return [char_to_int_map[s[i]] for i in range(len(s))]

def decode(int_list: list) -> str:
    """Decodes a list of integerss back into a string"""
    return ''.join([int_to_char_map[i] for i in int_list])

# tokenized input data
data = torch.tensor(encode(text), dtype=torch.long)

# split the data into train and validation sets
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

def generate_batch(split):
    """
    Generates a batch of input-output pairs from the data
    Args:
        split (str): Either 'train' or 'val'
    Returns:
        Tuple[Tensor,Tensor]: A batch of input and target sequences
    """
    if split == 'train':
        data = train_data
    else:
        data = val_data
    # xi are the starting points for context windows chosen randomly
    # create input sequences and corresponding target sequences (shifted by 1 to predict next token)
    xi = torch.randint(0, len(data) - context_length, (batch_size,))
    x = torch.stack([data[start:start + context_length] for start in xi])
    y = torch.stack([data[start + 1:start + 1 + context_length] for start in xi])
    return x,y

torch.no_grad()
def estimate_loss():
    out = {}
    # sets model to eval mode (disables features like dropout)
    model.eval()
    for split in ['train','val']:
        # for each split, initialize tensor to store loss values across eval_iters iterations 
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # for each split generate eval_iters batches and calculate loss on each
            X,Y = generate_batch(split)
            loss, output = model(X,Y)
            losses[k] = loss.item()
        # average loss across eval_iters iterations for each batch
        out[split] = losses.mean()
    # puts model back on training mode
    model.train()
    return out
            

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        """Initialize the language model with the size of the vocabulary"""
        super().__init__()
        self.vocab_size = vocab_size
    
    def embedding_lookup(self,input):
        """
        Perform embedding lookup to map tokens to embedding vectors
        Args:
            input (Tensor): Input tensor of token indices. Has shape (batch_size, context_length)
        Returns:
            Tensor: Embedding vectors for the input tokens (batch_size, context_length, vocab_size)
        """
        return nn.Embedding(vocab_size,vocab_size)(input)
    
    def forward(self,input,targets=None):
        """
        Forward pass for the language model.

        Args:
            input (Tensor): Input tensor of token indices.
            targets (Tensor, optional): Target tokens for calculating loss. Defaults to None.

        Returns:
            Tuple[Tensor or None, Tensor]: The loss (if targets are provided) and the model output.
        """
        # context aware repr. of input token in terms of other tokens. Each token becomes a vector of length vocab_size
        output = self.embedding_lookup(input) # <- shape is (batch_size,context_length,vocab_size)
        if not targets:
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
            # call forward method
            _, output = self(input)
            # extract the embedding of the last token
            last_hidden_state = output[:,-1,:]
            # convert this embedding vector into a probability distribution over the vocabulary
            distribution = F.softmax(last_hidden_state)
            # sample from distribution to get new token index 
            newToken = torch.multinomial(distribution,num_samples=1)
            input = torch.cat([input,newToken],dim=1)
            i += 1
        return input
    

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = generate_batch('train')
    
    loss, output = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    # calculate gradients of loss for every model parameter
    loss.backward()
    # update model weights using computer gradients and learning rate
    optimizer.step()
    
# generate output from model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))