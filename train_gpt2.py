from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import inspect
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------

class CausalSelfAttention (nn.Module):

    def __init__ (self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear (config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear (config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias, more of a mask, although following the OpenAI/HF naming
        self.register_buffer ("bias", torch.tril (torch.ones (config.block_size, config.block_size))
                                .view (1, 1, config.block_size, config.block_size))
    def forward (self, x):
        B, T, C = x.size () # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, value for all heads in a batch and move head forward to be the batch dimension
        # nh is "number of heads", hs is "head size" and C (number of channels) = nh * hs
        # eg. in GPT-2 (124M), n_head = 12, hs = 64, so nh*hs = C = 768 channels in the transformer    
        qkv = self.c_attn (x)
        q, k, v = qkv.split (self.n_embd, dim = 2)
        k = k.view (B, T, self.n_head, C // self.n_head).transpose (1,2) # (B, nh, T, hs)
        q = q.view (B, T, self.n_head, C // self.n_head).transpose (1,2) # (B, nh, T, hs)
        v = v.view (B, T, self.n_head, C // self.n_head).transpose (1,2) # (B, nh, T, hs)

        # attention materializes the large (T,T) matrix for all the queries and keys dotproduct
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt (k.size(-1))) # (B, nh, T, T)
        #att = att.masked_fill (self.bias[:,:,:T,:T] == 0, float('-inf'))
        #att = F.softmax (att, dim=-1)
        #y = att @ v # (B, nh, T, T) @ (B, nH, T, Hs) -> (B, nH, T, Hs)
        
        # flash attention does not materialize the large (T,T) matrix for all the queries and keys
        # evaluates softmax in a streaming manner (online normalized calc for softmax by nvidia) and fuses matmul ops in the same kernel
        y = F.scaled_dot_product_attention (q, k, v, is_causal=True)

        y= y.transpose (1, 2).contiguous().view(B,T,C) # reassemble all head outputs side by side
        
        
        # output projection
        y = self.c_proj(y)

        return y

class MLP (nn.Module):

    def __init__ (self, config):
        super().__init__()
        self.c_fc = nn.Linear (config.n_embd, 4 * config.n_embd)
        # Approximation form tanh : History! When the non linearity was developed it was slow in tensorflow so this approximation was developed.
        # this approximation ended up being picked up by BERT and by GPT2 etc
        # No real reason to use the approximate version today, there shouldn't be any big difference anymore.
        # Historical Quirk! Trying to replicate GPT2 exactly, which uses tanh approximation of GELU.
        self.gelu = nn.GELU (approximate='tanh')                    # Gaussian Error Linear Units (Similar to ReLU except there's no exactly flat tail here at exactly 0; slightly smoother relu)
        self.c_proj = nn.Linear (4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward (self, x):
        x = self.c_fc (x)
        x = self.gelu (x)
        x = self.c_proj (x)
        return x


class Block (nn.Module):

    def __init__ (self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm (config.n_embd)
        self.attn = CausalSelfAttention (config)
        self.ln_2 = nn.LayerNorm (config.n_embd)
        self.mlp = MLP (config)
        
    def forward (self, x):
        x = x + self.attn (self.ln_1 (x))   # pre normalization
        x = x + self.mlp (self.ln_2 (x))        # it is preferable to have a clean residual stream all the way from supervision to the inputs
        return x                            # unlike attention papaer where layer norms are inside the residual stream
    # Attention is a aggregation function, pooling function, weighted sum function, reduce operation. This is where tokens communicate and exchange information
    # Whereas MLP happens at every single token individually, there's no info being collected or exchanged between the tokens.
    # so the attention is the reduce and the MLP is the map, so what you end up with is the transformer ends up just being repeated application
    # of map-reduce if you wanna think about it that way. 
    # MLP is where the network thinks on the information pooled from other tokens individually
    # And every one of these blocks iteratively refines, representation inside the residual stream.



@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # numbre of tokens: 50,000 BPE Merges + 256 bytes tokens + 1 <|endofthext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT (nn.Module):

    def __init__(self, config):
        super().__init__ ()
        self.config = config

        self.transformer = nn.ModuleDict (dict(
            wte = nn.Embedding (config.vocab_size, config.n_embd),
            wpe = nn.Embedding (config.block_size, config.n_embd),
            h = nn.ModuleList ([Block (config) for _ in range (config.n_layer)]),
            ln_f = nn.LayerNorm (config.n_embd), # Final linear norm
        ))
        self.lm_head = nn.Linear (config.n_embd, config.vocab_size, bias=False) # n_embd to vocab_size without bias

        # Weight tying shceme between embedding and pre softmax linear layer
        # Makes training efficient, otherwise we'd have 30% (50257*768 = 40M; 40/120 ~ 30% of parameters), because we don't have to train as many parameters
        # and it improves results by putting in the iductive bias that both these embeddings should share similarities between tokens.
        # SAVED A TON OF PARAMETERS!
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    
    # hard coding bad practice, wont scale with increasing fanin like Xavier or Kaming init
    # but we will keep this because that is the GPT2 initialization per their source code
    # 0.02 is reasonably similar to 1/root(768 or 1024 or 1600 or 1280) for the gpt2 models
    def _init_weights (self, module):
        if isinstance (module, nn.Linear):
            std = 0.02
            if hasattr (module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance (module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward (self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size ()
        assert T <= self.config.block_size, f"Can not forward a senquence of length {T}, block_size is {self.config.block_size}"
        pos = torch.arange (0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe (pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte (idx) # token embeddings of shape (B, T, n_embd)
        x = pos_emb + tok_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block (x)
        # forward the final layer norm and the classifier
        x = self.transformer.ln_f (x)
        logits = self.lm_head (x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # the reduction in F.cross_entropy is mean (be careful while accumulating gradients to match larger batch sizes)
            loss = F.cross_entropy (logits.view(B*T, self.config.vocab_size), targets.view(B*T)) # stretch out logits and targets for cross entropy loss
        return logits, loss


    @classmethod
    def from_pretrained (cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print ("Loading weights from pretrained gpt: %s" % model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':     dict (n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium':     dict (n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':     dict (n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':     dict (n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT (config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained (model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith ('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any (k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape # sd_hf[k].shape[::-1] reverses order of dimensions
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    # ---------------------------------------------------------------------------------------------------
    # weight decay:
    # At initialization the weights are initialized so that all outcomes are equally likely,
    # now we want the weights to be optimized (pushed or pulled) so that likely outcomes are assigned higher probabilities
    # but at the same time, the we subject the weights to a force, regularizing force that kind of works against optimizing force
    # we decay the weights while stepping in the direction of gradients
    def configure_optimizers (self, weight_decay, learning_rate, device):
        # start with all the candidate parameters (that require grad)
        param_dict = {pn : p for pn, p in self.named_parameters()}
        param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all tensors in matmul + embeddings decay, all biases and layer norms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : nodecay_params, 'weight_decay' : 0.0}
        ]
        num_decay_params = sum (p.numel () for p in decay_params)
        num_nodecay_params = sum (p.numel () for p in nodecay_params)
        print (f"num decayed parameter tensors : {len(decay_params)} with {num_decay_params:,} parameters")
        print (f"num non-decayed parameter tensors : {len(nodecay_params)} with  {num_nodecay_params} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print (f"using fused AdamW: {use_fused}")
        # kernel fusion for AdamW update instead of iterating over all the tensors to step which is a lot slower.
        optimizer = torch.optim.AdamW (optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ------------------------------------------------------------------------------------------------------------------------------------------
import tiktoken
import numpy as np


def load_tokens (filename):
    npt = np.load (filename)
    npt = npt.astype(np.int32) # Introspective change
    ptt = torch.tensor (npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir (data_root)
        shards = [s for s in shards if split in s]
        shards = sorted (shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len (shards) > 0, f"no shards found for split {split}"
        if master_process:
            print (f"found {len(shards)} shards for split {split}")
        # state init at shard zero
        self.reset ()
    
    def reset (self):
        self.current_shard = 0
        self.tokens = load_tokens (self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch (self):
        B = self.B
        T = self.T

        buf = self.tokens[self.current_position: self.current_position + B*T +1]
        x = buf[:-1].view(B,T) # inputs
        y = buf[1:].view(B,T) # targets
        # advance the position in the tensor
        self.current_position += B*T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens (self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
# -----------------------------------------------------------------------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
    # torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# setup DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int (os.environ.get ('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands cuda, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int (os.environ['RANK'])
    ddp_local_rank = int (os.environ['LOCAL_RANK'])
    ddp_world_size = int (os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device (device)
    master_process = ddp_rank == 0 # this process will do the logging checkpoining etc
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print (f"using device: {device}")
    

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding('gpt2')

# Simulate total batch size in GPT3 paper by gradient accumulation till batch of specified size is processed and then stepping.
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens.
# The setting of B is purely optimization performance kind of setting, so in any case you should be getting the same answers
# upto like a floating point error, because the gradient accumulation kicks in and can handle everything serially! (as necessary)
# Because the real batch size is 2**19 either way HAHAHA!
B = 4 # micro batch size
T = 1024 # sequence length

# INTROSPECTION! that multi GPU results match single GPU:
# tweak total batch size so that you have the same number of grad accum steps in both settings
# to remove the boundary effect of the data loader which gets us slightly different numbers
# so that we get the same batches and the same loss and gradients

assert total_batch_size % (B*T * ddp_world_size) == 0, "make sure total batch size is divisible by (B*T * ddp_world_size)"
grad_accum_steps = total_batch_size // (B*T * ddp_world_size) # each process will do B*T and theres ddp_world_size processes
if master_process: # then guard this
    print (f"total desired batch size: {total_batch_size}")
    print (f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite (B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite (B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision ('high')

# let's now decrease amount of stuff we are gonna be moving around by dropping down to bfloat16 (only maintain 16 bits per float)

# Creat model
# 8 exact same GPT models are created on 8 processes, because the seeds are fixed
model = GPT (GPTConfig(vocab_size=50304))
model.to(device)
use_compile = False # Torch Compile intereferes with HellaSwag eval and  Generataion. TODO: fix
if use_compile:
    model = torch.compile(model)
if ddp:
    # forward is unchanged, backward is mostly unchanged except there is overlap between computation and communication of gradients
    # while the backward pass is still going on, to average the gradients from all processes
    # we're tacking on this average as we will see in a bit
    model = DDP (model, device_ids=[ddp_local_rank]) 
raw_model = model.module if ddp else model      # always contains the "raw" unwrapped model

# PyTorch has it's own cosine decay learning rate scheduler
# but it's just 5 lines of code, I know what's exactly going on
# and I don't like to use abstractions where they are kind of unscrutable and idk what they are doing
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # to match GPT-3 warmup schedule
max_steps = 19073
def get_lr (it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min_learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos (math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



# optimize!
# first try to crush a batch and overfit
# optimizer = torch.optim.AdamW (model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers (weight_decay = 0.1, learning_rate = 6e-4, device = device)


# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs (log_dir, exist_ok=True)
log_file = os.path.join (log_dir, f"log.txt")
with open (log_file, "w") as f: # open for writing to clear the file
    pass

for step in range (max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval ()
        val_loader.reset ()
        with torch.no_grad ():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range (val_loss_steps):
                x, y = val_loader.next_batch ()
                x, y = x.to(device), y.to(device)
                with torch.autocast (device_type=device, dtype=torch.bfloat16): # Only possible to do in Ampere. torch.fp16 requires use of gradient scalers as it can not represent the entire range of fp32
                    logits, loss = model (x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce (val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            # we have roughly infinity data, so we are expecting train and val loss to be the same.
            print (f"Validation loss : {val_loss_accum.item():.4f}")
            with open (log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 4 == 0 or last_step):
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
    
    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate (iterate_examples ("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example (example)
            tokens = tokens.to(device)
            mask = mask.to (device)

            # get the logits
            with torch.no_grad ():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model (tokens)
                pred_norm = get_most_likely_row (tokens, mask, logits)
                num_total += 1
                num_correct_norm += int (pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor (num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor (num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce (num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce (num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print (f"HellaSwag Accuracy : {num_correct_norm}/{num_total}= {acc_norm:.4f}")
            with open (log_file, "a") as f:
                f.write (f"{step} hella {acc_norm:.4f}\n")
    
    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 50

        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor (tokens, dtype=torch.long)
        
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        # Separate generator object for sampling, because I don't want to impact the rng state of the random number generator that is the global one
        # used for training. I want this to be completely outside the training loop.
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed (42 + ddp_rank)

        while xgen.size(1) < max_length:
            
            # forward the model and get the logits
            with torch.no_grad ():
                logits, loss = model (xgen) # (B, T, Vocab_size)
                # take the logits at the last position
                logits = logits [:,-1,:] # (B, Vocab_size)
                # get the probabilities
                probs = F.softmax (logits, dim=-1) # (B, Vocab_size)
                # do top-k sampling of 50
                # top-k probs here becomes (4, 50) top-k indices is becomes (4,50)
                topk_probs, topk_indices = torch.topk (probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: Multinomial does not demand the input to sum to 1
                ix = torch.multinomial (topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather (topk_indices, -1, ix)
                # append to the sequence
                xgen= torch.cat ((xgen,xcol), dim=1)
        # print the generated text
        for i in range (num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print (f"rank {ddp_rank} sample {i} : {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad ()
    loss_accum = 0.0
    for micro_step in range (grad_accum_steps):
        x, y = train_loader.next_batch()
        # ship active batches to gpu during training to be memory efficient    
        x, y = x.to(device), y.to(device)
        with torch.autocast (device_type=device, dtype=torch.bfloat16):
            logits, loss = model (x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss so it comes out right.
        loss = loss / grad_accum_steps
        # return a new tensor detached from the computational graph, but it still shares the same underlying storage.
        # stops gradient tracking, lets use the tensor in computations without it contributing to gradients.
        loss_accum += loss.detach()

        # only sync gradients across the processes, at the end of the large batch 0.5M tokens
        # hacky way to disable gradient sync for every single micro step
        if ddp:
            # very last backward will have the grad_sync flag as true
            # for now this hack works, but not a good practice if pytorch takes the flag away.
            # averages gradients
            model.require_backward_grad_sync = (micro_step == (grad_accum_steps - 1))
        loss.backward () # remember that loss.backward() always deposits gradients (grad += new_grad)
    # when we come out of the micro steps inner loop
    # every rank will suddenly, magically have the average of all the gradients on all the ranks

    if ddp:
        # calculates average of loss_accum on all the ranks, and it deposits that average on all the ranks
        # all the ranks will contain loss_accum averaged up
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # global norm of parameter vector is the length of it basically.
    # clip global norm of the parameter vector, basically make sure that the "length" of the parameters vector and clip it to 1.0
    # You can get unlucky during optimization, maybe it's a bad data batch, unlucky batches yield very high loss, which could lead to very high gradients,
    # this could basically shock your model and shock your optimization.
    # so gradient norm clipping prevents model from getting too big of shocks in terms of gradient magnitudes, and it's upperbounded in this way.
    # fairly hacky solution, patch on top of deeper issues, people still do it fairly frequently.

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # also returns norm, useful for visualisation
    # determine and set the learning rate for this iteration
    lr = get_lr (step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in mili seconds
    tokens_per_sec = (train_loader.B * train_loader.T * ddp_world_size * grad_accum_steps) / (t1-t0)
    if master_process:
        print (f"step {step} | loss : {loss_accum.item():.6f} | lr {lr:.4e} | norm : {norm:.4f} | dt : {dt:.2f}ms, tok/sec : {tokens_per_sec}")
        with open (log_file, "a") as f:
            f.write (f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group ()

