import os
import numpy as np
import tiktoken
import torch

class DataLoaderGPT():
    def __init__(self, B, T, process_rank, num_processes, data_root, split='train', is_text=True, tokenizer=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.is_text = is_text #If we have text as input or if it is already tokenized
        self.tokenizer = tiktoken.get_encoding('gpt2') if tokenizer is None else tokenizer
        self.data_root = data_root
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        #if master_process:
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    
    def load_tokens(self, path):
        
        if self.is_text:
            # We have a text file
            with open(path, 'r') as f:
                text = f.read()
            tokens = self.tokenizer.encode(text)
            tokens = torch.tensor(tokens, dtype=torch.long)

        else:
            # We have a tokenized file
            tokens = np.load(path).astype(np.int32)
            tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens
