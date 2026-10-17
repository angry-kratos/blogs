import torch 
import torch.nn as nn
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time, json
import matplotlib.pyplot as plt

@dataclass(frozen = True)
class GPTconfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768





class MLP(nn.Module):
    def __init__(self, cfg: GPTconfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, cfg.n_embd*4 )
        self.activation = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(cfg.n_embd*4, cfg.n_embd)

    def forward(self, x):
        x_fc = self.c_fc(x)
        x_a = self.activation(x_fc)
        out = self.c_proj(x_a)
        return out




class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd// config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(1,1,config.block_size, config.block_size)))

    def forward(self, x, past_kv = None):
        B, T, C = x.size()
        c_x = self.c_attn(x)
        q,k,v = c_x.split(self.n_embd, dim = 2)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim = 2)
            v = torch.cat([past_v, v], dim = 2)

        present = (k, v)

        T_total = k.size(2)
        T_past = T_total - T

        score = q @ k.transpose(-1, -2)/ (self.head_dim)**0.5

        masked_score = score.masked_fill(self.mask[:,:,T_past:T_total, :T_total] == 0, float('-inf'))
        smax_score = torch.softmax(masked_score, dim = -1)
        final_score = smax_score @ v
        tr_score = final_score.transpose(1,2).contiguous()
        out_x = tr_score.view(B,T,self.n_embd)


        out = self.c_proj(out_x)
        return out, present

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_kv = None):
        a, present = self.attn(self.ln_1(x), past_kv)
        x = x + a
        x = x + self.mlp(self.ln_2(x))

        return x, present


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
                                wte = nn.Embedding(config.vocab_size, config.n_embd),
                                wpe = nn.Embedding(config.block_size, config.n_embd),
                                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                                ln_f = nn.LayerNorm(config.n_embd) ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx, past_kvs = None):
        B, T= idx.size()
        T_past = 0 if past_kvs is None else past_kvs[0][0].size(2)

        assert T + T_past <= self.block_size

        pos = torch.arange(T_past,T_past + T, device = idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        new_kvs = []
        for i, block in enumerate(self.transformer.h):
            past = None if past_kvs is None else past_kvs[i]
            x, present = block(x, past)
            new_kvs.append(present)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_kvs







def load_hf_weights(model):
    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = hf.state_dict()
    sd = model.state_dict()
    kept_hf = [k for k in sd_hf if not k.endswith((".attn.bias", ".attn.masked_bias"))]
    kept =  [k for k in sd if not k.endswith(".attn.mask")]
    assert len(kept_hf) == len(kept), f"Mismatched Keys"

    transposed = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight"
    )

    with torch.no_grad():
        for k in kept_hf:
            if k.endswith(transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, "transpose is not the same shape"
                sd[k].copy_(sd_hf[k].t()) 
            else:
                assert sd_hf[k].shape == sd[k].shape, "direct is not the same shape"
                sd[k].copy_(sd_hf[k]) 


    return model
    

def generate(model, idx, max_new_tokens):

    model.eval()
    
    for tn in range(max_new_tokens):
                
        with torch.no_grad():
            next_token, _ = model(idx)
            processed = next_token[:, -1, :].argmax(dim = 1, keepdim = True)
           
        idx = torch.cat((idx, processed), dim = -1)

    return idx


@torch.no_grad()
def timed_generate(model, idx, max_new_tokens, block_size = 1024, warmup = 10):
    is_cuda = idx.device.type == "cuda"

    for _ in range(warmup):
        model(idx)
    if is_cuda:
        torch.cuda.synchronize()

    times = []                                    # (seq_len, ms)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]           # never feed more than 1024 tokens

        if is_cuda: torch.cuda.synchronize()      # GPU idle before we start the clock
        t0 = time.perf_counter()

        logits, _ = model(idx_cond)                                    # (B, T, V)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

        if is_cuda: torch.cuda.synchronize()      # wait for the GPU to actually finish
        times.append((idx_cond.size(1), (time.perf_counter() - t0) * 1000))

        idx = torch.cat((idx, next_token), dim=1)  # outside the timed region
    return idx, times




@torch.no_grad()
def generate_cached(model, idx, max_new_tokens):
    logits, kvs = model(idx)                                   # prefill: whole prompt, builds the cache
    for _ in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
        logits, kvs = model(next_token, kvs)                   # decode: ONE token + cache
    return idx

@torch.no_grad()
def timed_generate_cached(model, idx, max_new_tokens, warmup=10):
    for _ in range(warmup):
        model(idx)
    torch.cuda.synchronize()
    logits, kvs = model(idx)                                   # prefill (not timed)
    times = []
    for _ in range(max_new_tokens):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        logits, kvs = model(next_token, kvs)
        torch.cuda.synchronize()
        times.append((kvs[0][0].size(2), (time.perf_counter() - t0) * 1000))
        idx = torch.cat((idx, next_token), dim=1)
    return idx, times