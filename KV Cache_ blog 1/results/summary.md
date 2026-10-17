# KV cache: key numbers
 
Setup: GPT-2 small (124M, from scratch), NVIDIA A40, fp32, TF32 off, batch 1,
greedy decoding, prompt "Hello, I'm a language model,", median of 5 runs per position.
 
## Correctness
- Max |logit diff| vs Hugging Face GPT-2: **6.10e-05**
- Cached generation == no-cache generation (200 tokens): **True**
 
## Speed (ms per generated token)
 
| context t | no cache | KV cache | measured speedup | FLOP reduction | no-cache TFLOPS (% peak) | cached TFLOPS (% peak) |
|---|---|---|---|---|---|---|
| 10 | 4.89 | 4.94 | 1.0× | 10× | 0.51 (1.35%) | 0.050 (0.13%) |
| 100 | 5.20 | 4.95 | 1.1× | 100× | 4.82 (12.90%) | 0.051 (0.14%) |
| 500 | 12.70 | 4.95 | 2.6× | 500× | 10.46 (27.96%) | 0.054 (0.14%) |
| 1000 | 29.69 | 4.95 | 6.0× | 1000× | 9.56 (25.57%) | 0.057 (0.15%) |
 
Total time to generate to 1024 tokens: **14.66 s** without cache vs **5.03 s** with it.
 
## Memory
- KV cache per token (formula 2 × layers × d_model × bytes): **72 KiB**
- Formula at 1024 tokens: **72.0 MiB**; measured: **72.0 MiB**
- Weights: **475 MiB**, so the cache outgrows the weights at
  batch × context ≈ **6,751 tokens** (≈ batch 6.6 at full context)
- Bonus: the 12 causal-mask buffers cost **48 MiB** (one 1024×1024 fp32 mask per layer)
 
## Wrong positions (arange(0, T) during decode): no crash, silently wrong
Correct:  "Hello, I'm a language model, not a programming language. I'm a language model. I'm a language model. I'm a language model. I'm a language model. I'm a language model. I'm a language model. I'm a language model. I'm a language model. I'm a language model. I"
 
Wrong:    "Hello, I'm a language model, not,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"
