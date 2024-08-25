from GPT import GPT, GPTConfig
from DataLoaderGPT import DataLoaderGPT
import torch
import torch.nn as nn
import numpy as np
import time
import contextlib
import sys
print(sys.version)

# Seet seed
seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


#Set torch precision
torch.set_float32_matmul_precision('high')


# Find the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device is set to:', device)
print(f'Properties: {torch.cuda.get_device_properties(device=device)}')
#bf16supported = torch.cuda.is_bf16_supported()
bf16supported = False
print('bf16 supported:', bf16supported)

print(f'Python version: {sys.version}')
print(f'FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}')
print(f'torch version: {torch.__version__}')

# Load the GPT model
config = GPTConfig()
model = GPT(config)
model = model.to(device)

#Compile the model
print('Compiling the model')
model = torch.compile(model)
print('Model compiled...')

# Load the data
#path = '../data/simple/'
path = 'data/simple/'

#Some hyper params
epochs = 50
#total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
total_batch_size = 1024*16
B = 16 #micro batch size
T = 128 # sequence length
process_rank = 0
num_processes = 1
ddp_world_size = 1
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

split = 'train'
train_loader = DataLoaderGPT(B=B, T=T, process_rank=process_rank,num_processes=num_processes, split=split, data_root=path, is_text=True)


print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# Set bf16 if it is supported (speed things up)
precision_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if bf16supported else contextlib.nullcontext()
lr = 3e-4
# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for step in range(epochs):
    t0 = time.time()

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        with precision_context:
            logits, loss, = model(x, y)
        
        #import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    # Clip the gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    #if master_process:
    print(f"step {step:5d} | loss: {loss_accum:.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


print('Program terminated')
