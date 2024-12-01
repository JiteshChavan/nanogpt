"""
FineWeb-Edu dataset (for pretraining on 10B tokens from gpt2 tokenizer)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B"
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ----------------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int (1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesnt exist yet
DATA_CACHE_DIR = os.path.join (os.path.dirname(__file__), local_dir)
os.makedirs (DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset ("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding ("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize (doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend (enc.encode_ordinary(doc["text"]))
    tokens_np = np.array (tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile (filename, tokens_np):
    np.save (filename, tokens_np)