"""
A guided tour of a single forward pass through a transformer, using TransformerLens.

The big idea of mechanistic interpretability with TransformerLens:
  `model.run_with_cache(tokens)` runs the model AND records every intermediate
  activation along the way into a dictionary-like `ActivationCache`. That cache is
  your microscope. This script loads the smallest practical model, runs one prompt
  through it, and prints/explains every step of the computation.

Run it:   .venv/bin/python explore.py
Or pick a different model:   .venv/bin/python explore.py gpt2
"""

import sys
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)  # we're only inspecting, never training — saves memory

# ----------------------------------------------------------------------------
# 1. LOAD A MODEL
# ----------------------------------------------------------------------------
# "gpt2" is the classic starting point (12 layers, 124M params) and downloads
# reliably. If you want the *smallest* possible thing to really trace by hand,
# try one of Neel Nanda's toy models, e.g.:
#     attn-only-1l   -> 1 layer, attention only, NO MLPs  (easiest to reason about)
#     attn-only-2l   -> 2 layers, attention only          (where "induction" appears)
#     gelu-1l        -> 1 layer with an MLP
# Pass any of these as a command-line arg.
model_name = sys.argv[1] if len(sys.argv) > 1 else "attn-only-2l"

print(f"Loading '{model_name}' ...")
model = HookedTransformer.from_pretrained(model_name)
model.eval()

cfg = model.cfg
print(f"\n{'='*70}\nMODEL SHAPE\n{'='*70}")
print(f"  layers (n_layers):        {cfg.n_layers}")
print(f"  residual width (d_model): {cfg.d_model}")
print(f"  attention heads (n_heads):{cfg.n_heads}")
print(f"  head dim (d_head):        {cfg.d_head}")
print(f"  MLP hidden (d_mlp):       {cfg.d_mlp}")
print(f"  vocab size (d_vocab):     {cfg.d_vocab}")
print(f"  has MLPs:                 {cfg.attn_only is False}")

# ----------------------------------------------------------------------------
# 2. TOKENIZE A PROMPT
# ----------------------------------------------------------------------------
prompt = "The capital of France is"
tokens = model.to_tokens(prompt)  # shape [batch=1, seq_len]
str_tokens = model.to_str_tokens(prompt)

print(f"\n{'='*70}\nTOKENS\n{'='*70}")
print(f"  prompt:  {prompt!r}")
print(f"  tokens:  {tokens.tolist()[0]}")
print(f"  as text: {str_tokens}")
print("  (note the leading <|endoftext|> / BOS token TransformerLens prepends)")

# ----------------------------------------------------------------------------
# 3. RUN WITH CACHE — this is the microscope
# ----------------------------------------------------------------------------
logits, cache = model.run_with_cache(tokens)

# ----------------------------------------------------------------------------
# 4. EVERY ACTIVATION IN THE FORWARD PASS
# ----------------------------------------------------------------------------
# Each entry in the cache is a tensor produced at a named "hook point". The names
# follow a strict convention so you always know *where* in the computation you are:
#
#   hook_embed                      token embeddings
#   hook_pos_embed                  positional embeddings
#   blocks.{i}.hook_resid_pre       residual stream entering layer i
#   blocks.{i}.ln1.hook_normalized  input to attention, after layer-norm
#   blocks.{i}.attn.hook_q/k/v      per-head query/key/value vectors
#   blocks.{i}.attn.hook_pattern    attention pattern (softmax'd scores) <- the famous one
#   blocks.{i}.attn.hook_z          per-head weighted-sum of values
#   blocks.{i}.hook_attn_out        attention output written back to residual stream
#   blocks.{i}.hook_resid_mid       residual stream after attention, before MLP
#   blocks.{i}.mlp.hook_pre/post    MLP hidden activations (pre/post nonlinearity)
#   blocks.{i}.hook_mlp_out         MLP output written back to residual stream
#   blocks.{i}.hook_resid_post      residual stream leaving layer i
#   ln_final.hook_normalized        final residual stream, normalized
#
# Let's print every key with its shape:
print(f"\n{'='*70}\nALL CACHED ACTIVATIONS  (shape = [batch, seq, ...])\n{'='*70}")
for name, tensor in cache.items():
    print(f"  {name:42s} {tuple(tensor.shape)}")

# ----------------------------------------------------------------------------
# 5. THE RESIDUAL STREAM STORY
# ----------------------------------------------------------------------------
# A transformer is best understood as a "residual stream": a running sum vector
# (per token) that each layer READS from and WRITES to. The model's whole
# computation is: start with embeddings, then every attention head and MLP adds
# its contribution. Let's watch that sum grow, layer by layer, for the LAST token
# (the position whose prediction becomes the next word).
last = -1  # last token position
print(f"\n{'='*70}\nRESIDUAL STREAM at the final token, layer by layer\n{'='*70}")
print("  (showing the L2 norm of the residual vector as it accumulates)\n")

resid = cache["hook_embed"][0, last] + cache["hook_pos_embed"][0, last]
print(f"  embeddings (token + position)      |resid| = {resid.norm():.2f}")
for i in range(cfg.n_layers):
    attn_out = cache[f"blocks.{i}.hook_attn_out"][0, last]
    # In attn-only models there's no MLP, so the stream goes straight from
    # resid_pre -> (add attention) -> resid_post, with no resid_mid in between.
    if cfg.attn_only:
        after_attn = cache[f"blocks.{i}.hook_resid_post"][0, last]
        print(f"  + layer {i} attention               |resid| = {after_attn.norm():.2f}"
              f"   (attn wrote a vector of norm {attn_out.norm():.2f})")
    else:
        after_attn = cache[f"blocks.{i}.hook_resid_mid"][0, last]
        mlp_out = cache[f"blocks.{i}.hook_mlp_out"][0, last]
        after_mlp = cache[f"blocks.{i}.hook_resid_post"][0, last]
        print(f"  + layer {i} attention               |resid| = {after_attn.norm():.2f}"
              f"   (attn wrote a vector of norm {attn_out.norm():.2f})")
        print(f"  + layer {i} MLP                     |resid| = {after_mlp.norm():.2f}"
              f"   (mlp wrote a vector of norm {mlp_out.norm():.2f})")

# ----------------------------------------------------------------------------
# 6. WHAT EACH ATTENTION HEAD IS LOOKING AT
# ----------------------------------------------------------------------------
# The attention pattern tells you, for each head, which previous tokens the
# current token is "reading from". For the last token, let's see where layer-0
# head-0 attends.
print(f"\n{'='*70}\nATTENTION PATTERN  (layer 0, head 0, from the final token)\n{'='*70}")
pattern = cache["blocks.0.attn.hook_pattern"][0, 0, last]  # [n_heads, seq, seq] -> [seq]
for tok, weight in zip(str_tokens, pattern.tolist()):
    bar = "#" * int(weight * 40)
    print(f"  {tok!r:20s} {weight:5.2f} {bar}")

# ----------------------------------------------------------------------------
# 7. THE PREDICTION
# ----------------------------------------------------------------------------
# Logits -> probabilities over the vocabulary. The last position predicts the
# next token.
print(f"\n{'='*70}\nPREDICTION (top 10 next-token candidates)\n{'='*70}")
probs = logits[0, last].softmax(dim=-1)
top = probs.topk(10)
for prob, tok_id in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {prob:6.2%}  {model.to_string([tok_id])!r}")

print(f"\nDone. Open this file and read it top-to-bottom — every section is a place "
      f"you can poke at.\nTry changing `prompt`, or run with a different model: "
      f".venv/bin/python explore.py gpt2")
