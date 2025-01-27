# model hyperparameters
# 524288 tokens per batch so with max_iters of 7k the model will see in total over 3B tokens
config = {
    'batch_size' : 8,
    'context_length' : 1024, # length of input sequences
    'total_batch_size' : 524288, # 2 ** 19 is close to 0.5M in number of tokens
    'learning_rate' : 1e-6,
    'max_iters' : 7000,
    'eval_interval' : 500,
    'eval_iters' : 200,
    'vocab_size' : 50257, # 50257 with BPE but it turns out the next power of two (50304) is likely more efficient
    'embed_size' : 768,
    'dropout' : 0.1,
    'num_layers' : 12,
    'num_heads' : 12
}