"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=3 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import argparse
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import matplotlib
# use non-interactive backend for headless logging
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import GPTConfig, GPT, get_lora_model

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
ffn_mult = 4.0 # default FFN expansion factor; can be reduced in configs

# TODO: MOE config defaults (can be overridden by config files)
n_expert = 0
n_routed_expert = 1
load_balancing_lambda = 0.01

# How often to update the routing diagram (in iterations). This only affects
# logging of the multi-layer diagram and does NOT change evaluation/checkpointing.
# Can be overridden from config files; default 1 (every iter) as requested.
plot_interval = 1

# LoRA params
lora_rank = 0
lora_alpha = 0.0 # set alpha to the first rank which is tried, then keep it fixed, and don't further tune it (see the paper for more info)
lora_dropout = 0.0
compute_grad_memory = False # compute the memory usage of the gradients

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system

# TODO: added for load balancing
load_balancing_lambda = 0.01

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# Add optional explicit resume checkpoint path via CLI arg that can coexist with
# the existing configurator-based overrides.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Explicit checkpoint path to resume from when init_from=resume')
parser.add_argument('--reset_iter', action='store_true', help='When resuming, reset iteration counter and best_val_loss (start fresh stage)')
known, unknown = parser.parse_known_args()
# Define global so configurator.py recognizes the key when passed as --resume_ckpt_path=...
resume_ckpt_path = known.resume_ckpt_path
reset_iter = known.reset_iter

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Resolve final output directory so every run lands under the base 'out' folder.
# Rules:
# - Absolute paths in out_dir are respected as-is.
# - Relative paths that already start with 'out/' are respected.
# - Other relative paths are nested under 'out/<out_dir>'.
# - If the result is exactly 'out', append an auto run name based on config file or timestamp.
_orig_out_dir = out_dir
try:
    import sys
    _cfg_file = globals().get('config_file', None)
    _cfg_name = None
    if _cfg_file:
        _cfg_name = os.path.splitext(os.path.basename(_cfg_file))[0]
    _stamp = time.strftime('%Y%m%d_%H%M%S')

    if not os.path.isabs(out_dir):
        norm = out_dir.replace('\\', '/').lstrip('./')
        if not (norm == 'out' or norm.startswith('out/')):
            out_dir = os.path.join('out', out_dir)
        # If still exactly 'out', create a run subfolder name with timestamp
        if norm == '' or norm == 'out':
            base_name = None
            if _cfg_name:
                base_name = _cfg_name
            elif isinstance(wandb_run_name, str) and len(wandb_run_name) > 0:
                base_name = wandb_run_name
            else:
                base_name = 'run'
            run_name = f"{base_name}-{_stamp}"
            out_dir = os.path.join('out', run_name)
    # else: absolute path -> leave unchanged
except Exception:
    # Best-effort; if anything goes wrong, fall back to original
    out_dir = _orig_out_dir

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8 # simulate 8 gpus
print("total number of tokens per iteration:", batch_size * block_size * gradient_accumulation_steps)

# TODO: debug
try:
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output directory: {out_dir}")
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    # Allow overriding train/val bin paths via config keys `train_bin` and `val_bin`.
    # Fallback to default layout: data/<dataset>/{train,val}.bin
    data_dir = os.path.join('data', dataset) if isinstance(dataset, str) else 'data'
    train_bin_path = globals().get('train_bin', os.path.join(data_dir, 'train.bin'))
    val_bin_path = globals().get('val_bin', os.path.join(data_dir, 'val.bin'))
    print(f"loading train bin: {train_bin_path}")
    print(f"loading val   bin: {val_bin_path}")
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    # Note: we only want to do LoRA fine-tuning when we resume or start with a pretrained model and NOT when we start from scratch
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
        ffn_mult=ffn_mult,
        n_expert=n_expert,
        n_routed_expert=n_routed_expert,
    ) # start with model_args (configurable)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint. Prefer explicit resume_ckpt_path if provided.
        ckpt_path = resume_ckpt_path if resume_ckpt_path else os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Resume requested (init_from=resume) but checkpoint not found: '{ckpt_path}'. "
                "Ensure you ran the merge (post_train.py) or copied the seed ckpt into this directory."
            )
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'lora_rank', 'lora_alpha', 'ffn_mult', 'n_expert', 'n_routed_expert']:
            model_args[k] = checkpoint_model_args.get(k, 0)
        # create the model

        # LoRA fine-tuning?    
        if lora_rank > 0:
            model_args['lora_rank'] = lora_rank
            model_args['lora_alpha'] = lora_alpha
            model_args['lora_dropout'] = lora_dropout

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint.get('iter_num', checkpoint.get('iter_global', 0))
        best_val_loss = checkpoint.get('best_val_loss', 1e9)

        if reset_iter:
            print(f"[resume] --reset_iter specified: resetting iter_num from {iter_num} -> 0 and best_val_loss -> 1e9; reinitializing optimizer.")
            iter_num = 0
            best_val_loss = 1e9
            # Rebuild optimizer fresh (discarding loaded state)
            optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

        # CLEANUP: remove any forced expert routing attributes so router can learn normally
        removed = 0
        for blk in model.transformer.h:
            mlp = getattr(blk, 'mlp', None)
            if mlp is not None and hasattr(mlp, 'force_expert_idx'):
                try:
                    delattr(mlp, 'force_expert_idx')
                    removed += 1
                except Exception:
                    pass
        if removed > 0:
            print(f"[resume] cleared force_expert_idx from {removed} MoE blocks")

        if lora_rank > 0:
            # Only make LoRA weights tunable
            print("Marking model as LoRA fine-tunable...")
            model = get_lora_model(model)
            print("Done.")

    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(
            dropout=dropout,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'lora_rank', 'lora_alpha', 'ffn_mult', 'n_expert', 'n_routed_expert']:
            model_args[k] = getattr(model.config, k)
        
        if lora_rank > 0:
            # Only make LoRA weights tunable
            print("Marking model as LoRA fine-tunable...")
            model = get_lora_model(model)
            print("Done.")

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        opt_state = checkpoint.get('optimizer', {})
        try:
            # Only load if state looks valid
            if isinstance(opt_state, dict) and 'param_groups' in opt_state:
                optimizer.load_state_dict(opt_state)
            else:
                print("[resume] optimizer state unavailable or invalid; using fresh optimizer")
        except Exception as e:
            print(f"[resume] failed to load optimizer state: {e}; proceeding with fresh optimizer")
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    # --- OLD CODE: _, loss = model(X, Y) ---
                    # --- UPDATED CODE: Unpack all 3 values, use the 2nd one (main_loss) ---
                    _, main_loss, _ = model(X, Y) 
                
                # Use the main loss for evaluation reporting
                losses[k] = main_loss.item() 
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        # Prepare a history buffer to accumulate routing frequencies per layer/expert
        # Prepare a history buffer to accumulate routing frequencies per layer/expert
        # Attach it to the underlying model object (handles DDP wrapper if present)
        try:
            target_model = model.module if isinstance(model, DDP) else model
            n_layer = int(getattr(target_model.config, 'n_layer', 12))
            n_expert = int(getattr(target_model.config, 'n_expert', 0))
        except Exception:
            # fallback defaults
            n_layer = 12
            n_expert = 0

        # routing history: { layer_idx: { 'iters': [...], 'freqs': { expert_idx: [...] } } }
        routing_history = {
            li: {'iters': [], 'freqs': {ei: [] for ei in range(max(1, n_expert))}}
            for li in range(n_layer)
        }
        # attach so later code (which uses raw_model) can access it via the underlying module
        try:
            target_model._wandb_routing_history = routing_history
        except Exception:
            # best-effort: if we can't attach, keep in global
            _wandb_routing_history = routing_history

except Exception as e:
    # Print the specific error and its full traceback
    rank = int(os.environ.get('RANK', -1)) if 'RANK' in os.environ else -1
    print(f"FATAL ERROR on rank {rank}: {e}")
    import traceback
    traceback.print_exc()
    # Ensure the process exits to signal failure
    exit(1)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            #TODO: updated , now also return aux loss : logits, loss = model(X, Y)
            # 1. Model returns 3 values: logits, main_loss (L_LM), aux_loss (L_aux)
            logits, main_loss, aux_loss = model(X, Y)

            # 2. Combine losses: L_total = L_LM + lambda * L_aux
            # Assuming config['load_balancing_lambda'] is 0.01
            total_loss = main_loss + config['load_balancing_lambda'] * aux_loss
            
            # 3. Scale the total_loss to account for gradient accumulation
            loss_for_backward = total_loss / gradient_accumulation_steps
        
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss_for_backward).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    if compute_grad_memory:
        # compute the gradient memory usage
        grad_memory = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_memory += p.grad.numel() * p.grad.element_size()
        grad_memory = grad_memory / 1024**2
        print(f"grad memory usage: {grad_memory:.2f} MB")

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # We fetch the individual losses and the total loss, which were calculated
        # using the unscaled main_loss and aux_loss prior to the division by gradient_accumulation_steps.
        
        # Scaling is already handled in the previous step, here we just fetch the values from GPU.
        main_lossf = main_loss.item() 
        aux_lossf = aux_loss.item()
        total_lossf = total_loss.item()

        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # Print a more informative log line, showing the three components of the loss
        print(f"iter {iter_num}: LM_loss {main_lossf:.4f}, AUX_loss {aux_lossf:.4f}, TOTAL_loss {total_lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        if wandb_log:
            # Collect per-layer routing frequencies and append to our history buffer
            try:
                # access routing history attached to the model (set up at init time)
                target_hist = getattr(raw_model, '_wandb_routing_history', None)
                # prefer the model-exposed latest_routing if present
                latest = getattr(raw_model, 'latest_routing', None)
                handled = False
                if latest is not None and 'layer_expert_freq' in latest and target_hist is not None:
                    layer_freqs = latest['layer_expert_freq']
                    for li, freqs in enumerate(layer_freqs):
                        if freqs is None:
                            continue
                        fcpu = freqs.detach().to('cpu')
                        # append iteration and per-expert values
                        target_hist[li]['iters'].append(iter_num)
                        for ei, val in enumerate(fcpu.tolist()):
                            target_hist[li]['freqs'].setdefault(ei, []).append(float(val))
                    handled = True

                # fallback: build from last_selected_experts inside blocks
                if not handled and hasattr(raw_model, 'transformer') and target_hist is not None:
                    n_exp = int(getattr(raw_model.config, 'n_expert', 0))
                    if n_exp and n_exp > 0:
                        for li, block in enumerate(raw_model.transformer.h):
                            mlp = getattr(block, 'mlp', None)
                            sel = getattr(mlp, 'last_selected_experts', None)
                            if sel is None:
                                continue
                            counts = torch.bincount(sel.reshape(-1), minlength=n_exp)
                            total = sel.numel() if sel.numel() > 0 else 1
                            freqs = counts.to(torch.float32) / float(total)
                            fcpu = freqs.detach().to('cpu')
                            target_hist[li]['iters'].append(iter_num)
                            for ei, val in enumerate(fcpu.tolist()):
                                target_hist[li]['freqs'].setdefault(ei, []).append(float(val))
            except Exception:
                # best-effort: if anything fails, skip routing accumulation for this step
                target_hist = None

            # base log: core training scalars + per-layer expert activation (scalar-only)
            base_log = {
                "iter": iter_num,
                "train/LM_loss": main_lossf,
                "train/AUX_loss": aux_lossf,
                "train/TOTAL_loss": total_lossf,
                "lr": lr,
                "mfu": running_mfu*100,
            }

            # If routing history exists, append latest per-layer/expert frequencies
            try:
                hist = getattr(raw_model, '_wandb_routing_history', None)
                if hist is not None:
                    # assume 12 layers and up to 4 experts as common defaults
                    for li in range(len(hist)):
                        freqs = hist[li]['freqs']
                        for ei, series in freqs.items():
                            if len(series) > 0:
                                base_log[f"layer{li}/expert{ei}"] = series[-1]
            except Exception:
                pass

            wandb.log(base_log)

            # Remove matplotlib image logging; keep scalar-only logging
    iter_num += 1
    local_iter_num += 1

    # TODO: added forced checkpoint at max_iters
    if iter_num > max_iters:
        
        # --- START ADDED CODE BLOCK ---
        if master_process:
            print(f"Max iterations ({max_iters}) reached. Forcing final checkpoint save.")
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            # Saving as ckpt.pt to ensure the default sample.py can find it
            print(f"saving final checkpoint to {os.path.join(out_dir, 'ckpt.pt')}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        # --- END ADDED CODE BLOCK ---
        
        break

if ddp:
    destroy_process_group()
