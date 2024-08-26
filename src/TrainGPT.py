# -----------------------------------------------------------------------------
# simple launch:
# python TrainGPT.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 TrainGPT.py


from GPT import GPT, GPTConfig
from DataLoaderGPT import DataLoaderGPT
from LRScheduler import LRScheduler
import torch
import torch.nn as nn
import numpy as np
import time
import contextlib
import sys
import os

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import builtins


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    # init_process_group(backend='gloo')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    # Find the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Decorate print so that we only print from the master process
def print(*args, **kwargs):
    if master_process:
        builtins.print(*args, **kwargs)

        
# Seet seed
seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


#Set torch precision
torch.set_float32_matmul_precision('high')



#See if we are on linux, if so we can compile the model
compileSupported = sys.platform == 'linux'
print(f'Platform: {sys.platform} Compile supported: {compileSupported}')
print(f'Python version: {sys.version}')
print(f'torch version: {torch.__version__}')
print(f'FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}')



#bf16supported = torch.cuda.is_bf16_supported()
bf16supported = False
print('bf16 supported:', bf16supported)




print('Device is set to:', device)
print(f'Properties: {torch.cuda.get_device_properties(device=device)}')

# Load the GPT model
config = GPTConfig()

#Print the config:
print(f'GPTConfig: {config}')

model = GPT(config)
model = model.to(device)

#Compile the model
if compileSupported:
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
# process_rank = 0
# num_processes = 1
# ddp_world_size = 1
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

split = 'train'
train_loader = DataLoaderGPT(B=B, T=T, process_rank=ddp_rank,num_processes=ddp_world_size, split=split, data_root=path, is_text=True)


print(f"total desired batch size: {total_batch_size}")
print(f"calculated gradient accumulation steps: {grad_accum_steps}")

# Set bf16 if it is supported (speed things up)
precision_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if bf16supported else contextlib.nullcontext()


lr_scheduler = LRScheduler(max_lr=6e-4, min_lr=6e-5, warmup_steps=10, max_steps=50)
lr = lr_scheduler.get_lr(0)

# Create the optimizer
weight_decay = 0.01
param_dict = {pn: p for pn, p in model.named_parameters()}
param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}

# Weight decay applies to all parameters except biases and layernorms
# Embeddings and matmul stuff is decayed
decay_params = [p for n,p in param_dict.items() if p.dim() >=2]
no_decay_params = [p for n,p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': no_decay_params, 'weight_decay': 0.0}
]
num_decay_params = sum(p.numel() for p in decay_params)
num_no_decay_params = sum(p.numel() for p in no_decay_params)
print(f"number of decay params: {num_decay_params}")
print(f"number of no decay params: {num_no_decay_params}")


#optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=lr)
optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95), eps=1e-8, lr=lr, fused=True)
for step in range(epochs):
    t0 = time.time()

    # do one step of the optimization
    model.train()

    # Zero the gradients
    optimizer.zero_grad()
    loss_accum = 0.0

    #Run a number of micro steps to accumulate the gradients, normalize by the number of steps, since we want the mean gradient (not the pure sum)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)        

        #Use bf16 if supported
        with precision_context:
            #Forward pass
            logits, loss, = model(x, y)
        
        #import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    # Clip the gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Get the learning rate
    lr = lr_scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    #if master_process:
    print(f"step {step:5d} | loss: {loss_accum:.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


print('Program terminated')
