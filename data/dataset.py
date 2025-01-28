import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from config.datasetconfig import config as dataset_config
cores_to_use = os.cpu_count() // 2

# here we download the dataset, tokenize all of the documents and save them to a shard in the data_cache_dir
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__),dataset_config.data_dir,dataset_config.shard_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fineWeb = load_dataset(dataset_config.dataset_name, name=dataset_config.sample_name, split="train")

# initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
# document delimiter
eot = enc._special_tokens['<|endoftext|>']
def tokenize(document):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(document["text"]))
    
    tokens_np = np.array(tokens)
    assert (0 < tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large"
    tokens_np_uint16 = tokens_np.astype(np.uint16) # save space with integer encoding
    return tokens_np_uint16

num_processes = max(1, cores_to_use)
shard_indices = [0]
shard_size = dataset_config.shard_size
# creates a pool of processes to process the dataset in parallel
with mp.Pool(num_processes) as pool:
    shard_index = 0
    all_tokens = np.empty((shard_size,), dtype=np.uint16) # initialize buffer to hold the current shard
    token_count = 0 # tracks num tokens in current shard until the shard is full
    progress_bar = None
    for tokens in pool.imap(tokenize, fineWeb, chunksize=16):
        # applies the tokenize function to each element of the iterable fineWeb dataset (each worker processes a batch of <chunksize> documents)
        if token_count + len(tokens) < shard_size: 
            # if there is enough space - adds all of the tokens from the current document into this shard
            all_tokens[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}:")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fine_web_{split}_{shard_index:04d}.npy")
            shard_indices.append(shard_index)
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens[token_count:token_count + remainder] = tokens[:remainder]
            np.save(filename, all_tokens)
            shard_index += 1
            progress_bar = None
            all_tokens[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
    # write remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fine_web_{split}_{max(shard_indices):04d}.npy")
        np.save(filename, all_tokens[:token_count])