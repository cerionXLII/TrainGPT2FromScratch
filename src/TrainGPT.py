# -----------------------------------------------------------------------------
# simple launch:
# python TrainGPT.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 TrainGPT.py


from GPT import GPT, GPTConfig, Generator
from DataLoaderGPT import DataLoaderGPT
from LRScheduler import LRScheduler
import tiktoken
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

if ddp:
    print('Wrapping model in DDP')
    model = DDP(model, device_ids=[ddp_local_rank], output_device=device)

raw_model = model.module if ddp else model # get the underlying model to be able to sen them to the optimizer
# Load the data
#path = '../data/simple/'
path = 'data/simple/'

#Some hyper params
max_steps = 5000 # number of training steps rather than epochs

# create the log directory we will write checkpoints to and log to
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

#total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
total_batch_size = 1024*16
B = 16 #micro batch size
T = 256 # sequence length
# process_rank = 0
# num_processes = 1
# ddp_world_size = 1
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

train_loader = DataLoaderGPT(B=B, T=T, process_rank=ddp_rank,num_processes=ddp_world_size, split='train', data_root=path, is_text=True)
val_loader = DataLoaderGPT(B=B, T=T, process_rank=ddp_rank,num_processes=ddp_world_size, split='val', data_root=path, is_text=True)


print(f"total desired batch size: {total_batch_size}")
print(f"calculated gradient accumulation steps: {grad_accum_steps}")


# Set the tokenizer for sampling
tokenizer = tiktoken.get_encoding('gpt2')

# Set bf16 if it is supported (speed things up)
precision_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if bf16supported else contextlib.nullcontext()


lr_scheduler = LRScheduler(max_lr=6e-4, min_lr=6e-5, warmup_steps=10, max_steps=50)
lr = lr_scheduler.get_lr(0)

# Create the optimizer
weight_decay = 0.01
param_dict = {pn: p for pn, p in raw_model.named_parameters()}
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
for step in range(max_steps):
    t0 = time.time()

    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with precision_context:
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)


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
        
        if ddp:
            #Only sync the gradients between GPUs if we are on the last micro step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        
        loss.backward()

    # Get the mean loss between GPUs
    if ddp:        
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
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

    if master_process:
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

        if ((step % 250 == 0) or last_step):
            #Generate some sample output
            generator = Generator(raw_model, tokenizer)
            outputs = generator.generate('The meaning of life is: ', max_len=100, top_k=50, num_return_sequences=5)
            for text in outputs:
                print(text)

# Clean up
if ddp:
    print('Cleaning up... Destroying process group')
    destroy_process_group()

print('Program terminated')
