
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Overwriting the methods of nn.Linear:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )

        # LoRA stuff
        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)

            self.lora_scaling = lora_alpha / lora_rank
            self.lora_A = nn.Parameter(torch.empty((lora_rank, self.in_features), device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, lora_rank), device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Same as nn.Linear
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = nn.Linear.forward(self, input)
        if not self.has_weights_merged and self.is_lora():
            # h = Wx + BAx * scaling
            x += self.lora_scaling * F.linear(
                F.linear(
                    self.lora_dropout(input),
                    self.lora_A
                ),
                self.lora_B
            )
        return x

    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

    def train(self, mode: bool = True) -> "LoRALinear":
        nn.Linear.train(self, mode)
        if self.has_weights_merged and self.is_lora():
            # de-merge weights, i.e., remove BA from W = W + BA
            self.weight.data -= self.lora_scaling * self.lora_B @ self.lora_A
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        nn.Linear.eval(self)
        if not self.has_weights_merged and self.is_lora():
            # merge weights, i.e., add BA to W
            self.weight.data += self.lora_scaling * self.lora_B @ self.lora_A
            self.has_weights_merged = True
        return self

def get_lora_model(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = LoRALinear(
            in_features=config.n_embd,
            out_features=3 * config.n_embd,
            bias=config.bias,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        # output projection
        self.c_proj = LoRALinear(
            in_features=config.n_embd,
            out_features=config.n_embd,
            bias=config.bias,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# TODO: added SequenceMoE class here
class SequenceMoE(nn.Module):
    """
    A Sequence-level Mixture of Experts layer. The router processes the average
    of the sequence to decide which experts will process the entire sequence.
    """
    def __init__(self, config):
        super().__init__()
        self.n_expert = config.n_expert
        self.top_k = config.n_routed_expert
        self.n_embd = config.n_embd

        # 1. Experts: A list of standard MLP/FFN layers (your original MLP serves as the expert)
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.n_expert)])

        # 2. Router: A single Linear layer to route the sequence representation
        # It takes the average embedding size (n_embd) and outputs logits for n_expert choices
        # The router does NOT use LoRA, similar to how the base MLP doesn't in your file.
        self.router = nn.Linear(config.n_embd, self.n_expert, bias=False)

        # (Optional) Implement load balancing loss logic here if needed
        # self.load_balancing_loss = ... 

        # Initialize the router weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.02)


    def forward(self, x):
            B, T, C = x.size() # batch size (B), sequence length (T), embedding dim (C)

            # 1. Sequence-Level Routing: Compute a single, global representation of the sequence
            # Average the tokens across the sequence dimension (T)
            sequence_rep = x.mean(dim=1) # shape: (B, C)

            # 2. Router: Compute expert scores (logits) for the batch
            router_logits = self.router(sequence_rep) # shape: (B, n_expert)
            #tODO: this is the papers first softmax
            # Calculate Full Probabilities (P_i) and ensure float32 for stable softmax
            router_probs_full = F.softmax(router_logits, dim=-1, dtype=torch.float32) # (B, n_expert)
            
            # 3. Top-K Selection and Gating
            # Convert back to the model's dtype (e.g., bfloat16/float16) for the forward pass
            router_probs = router_probs_full.to(x.dtype)
            
            # Get the top-k experts and their scores (weights)
            weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1) # (B, top_k)
            # TODO: this is the papers second softmax, the normailisation is identical
            # Normalize the top-k weights (crucial step from the paper's formula)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            
            # 4. Expert Processing and Weighted Sum
            final_output = torch.zeros_like(x)

            # Loop through the batch (B) to route each sequence
            for i in range(B):
                # The full sequence x[i] is processed by the selected experts
                current_token_sequence = x[i] # shape: (T, C)
                output_i = torch.zeros_like(current_token_sequence) 

                # Loop through the Top-K selected experts for this sequence
                for k in range(self.top_k):
                    expert_idx = selected_experts[i, k].item()
                    expert_weight = weights[i, k]

                    # Run the sequence through the k-th expert
                    expert_output = self.experts[expert_idx](current_token_sequence) # (T, C)

                    # Weight the expert's output by the router's probability/weight
                    output_i += expert_weight * expert_output

                final_output[i] = output_i

            # 5. Load Balancing Loss Calculation (L_aux)
            # We use the full probability tensor (router_probs_full) for L_aux, which is numerically stable (float32).

            # Expert Importance (P_i): Expected probability mass sent to each expert (across the batch)
            # Sum of P_i across the batch: (B, n_expert) -> (n_expert,) 
            expert_importance = router_probs_full.sum(dim=0) / B

            device = x.device
            
            is_assigned = torch.zeros_like(router_probs_full, dtype=torch.float32, device=device).scatter_(
                1, selected_experts.long(), 1.0 # The selected_experts is already on the GPU
            )
            expert_load = is_assigned.sum(dim=0) / B

            # L_aux = N * sum(Importance * Load)
            # The goal is to minimize this product, which encourages Importance (P_i) and Load (f_i) to be uniform (1/N)
            load_balancing_loss = self.n_expert * torch.sum(expert_importance * expert_load) 

            # Return final output and the calculated auxiliary loss
            return final_output, load_balancing_loss

# TODO: before self.mlp = MPL(config), now we change this to the MoE version if nr experts > 0
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        # Conditional replacement of MLP with MoE
        if config.n_expert > 0:
            print(f"Block initialized with SequenceMoE (Experts: {config.n_expert}, Top-K: {config.n_routed_expert})")
            self.mlp = SequenceMoE(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        
        # Determine if we need to unpack one or two values from self.mlp(self.ln_2(x))
        if isinstance(self.mlp, SequenceMoE):
            y, aux_loss = self.mlp(self.ln_2(x))
            x = x + y
            return x, aux_loss # Return two values from Block
        else:
            x = x + self.mlp(self.ln_2(x))
            return x # Standard Block returns one value

# TODO: this is our defaults, but we override them in config/bp_train_4gb_test.py
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # LoRA parameters
    lora_rank: int = 0
    lora_alpha: float = 0.0
    lora_dropout: float = 0.0
    # TODO : added two row below, nr N=4 experts and top k=2 routing
    n_expert: int = 4       # Number of experts (0 = disabled, use standard MLP)
    n_routed_expert: int = 2 # Top-K experts to route to (typically 2)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Initialize total auxiliary loss (will remain zero if n_expert=0)
        total_aux_loss = torch.zeros(1, device=device)
        
        for block in self.transformer.h:
            # Conditional check for MoE: if the block is an MoE, it returns (output, aux_loss)
            if isinstance(block.mlp, SequenceMoE):
                x, aux_loss = block(x) 
                total_aux_loss += aux_loss
            else:
                # If it's a standard block, it returns only (output)
                x = block(x)
                
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # RETURN THREE VALUES: logits, main_loss, and the accumulated auxiliary loss
            return logits, main_loss, total_aux_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            
            # RETURN THREE VALUES: logits, None for loss, and zero auxiliary loss
            return logits, None, torch.zeros(1, device=device)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Loading pretrained GPT-2 models from HuggingFace hub.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' or k.startswith("lora") for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        if 'lora_rank' in override_args:
            print(f"overriding lora_rank and lora_alpha to {override_args['lora_rank']}")
            config_args['lora_rank'] = override_args['lora_rank']
            config_args['lora_alpha'] = override_args['lora_alpha']
            if 'lora_dropout' in override_args:
                print(f"overriding lora_dropout to {override_args['lora_dropout']}")
                config_args['lora_dropout'] = override_args['lora_dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        if config.lora_rank == 0:
            assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        else:
            # Subtract the LoRA parameters from len(sd_keys)
            assert len(sd_keys_hf) == len(sd_keys) - 4 * config.n_layer
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        if self.config.lora_rank > 0:
            # No special treatment for LoRA
            return torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
