"""
Run the models and export everything the website needs to visualize a forward pass.
Writes a single self-contained JSON blob to data.json.

Captured per model:
  - config (layers, heads, d_model, attn_only)
  - the prompt + its string tokens
  - attention patterns  [layer][head][query_pos][key_pos]
  - residual-stream checkpoints with their L2 norm (how the running sum grows)
  - logit lens: at each checkpoint, what the model WOULD predict for the next token
    if you decoded the residual stream right there (top-5)
  - final top-10 next-token prediction
"""

import json
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

PROMPT = "The Eiffel Tower is in the city of"
MODELS = ["attn-only-2l", "gpt2"]
TOPK_LENS = 5
TOPK_FINAL = 10


def logit_lens(model, resid_vec):
    """Decode a single residual-stream vector into a next-token distribution."""
    normed = model.ln_final(resid_vec.unsqueeze(0))          # [1, d_model]
    logits = normed @ model.W_U + model.b_U                  # [1, d_vocab]
    probs = logits[0].softmax(dim=-1)
    top = probs.topk(TOPK_LENS)
    return [
        {"token": model.to_string([tid]), "prob": float(p)}
        for p, tid in zip(top.values.tolist(), top.indices.tolist())
    ]


def export_model(name):
    print(f"  loading {name} ...")
    model = HookedTransformer.from_pretrained(name)
    model.eval()
    cfg = model.cfg

    # prepend_bos=False: gpt2 predicts more naturally without the leading BOS token
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    str_tokens = model.to_str_tokens(PROMPT, prepend_bos=False)
    last = tokens.shape[1] - 1

    logits, cache = model.run_with_cache(tokens)

    # --- attention patterns: [layer][head][q][k] ---
    attention = []
    for i in range(cfg.n_layers):
        pat = cache[f"blocks.{i}.attn.hook_pattern"][0]      # [head, q, k]
        attention.append(pat.tolist())

    # --- residual checkpoints (norms) + logit lens at each ---
    checkpoints = []

    def add_checkpoint(label, kind, vec):
        checkpoints.append({
            "label": label,
            "kind": kind,                  # "embed" | "attn" | "mlp"
            "norm": float(vec.norm()),
            "lens": logit_lens(model, vec),
        })

    embed = cache["hook_embed"][0, last] + cache["hook_pos_embed"][0, last]
    add_checkpoint("embeddings", "embed", embed)
    for i in range(cfg.n_layers):
        if cfg.attn_only:
            add_checkpoint(f"L{i} attention", "attn",
                           cache[f"blocks.{i}.hook_resid_post"][0, last])
        else:
            add_checkpoint(f"L{i} attention", "attn",
                           cache[f"blocks.{i}.hook_resid_mid"][0, last])
            add_checkpoint(f"L{i} MLP", "mlp",
                           cache[f"blocks.{i}.hook_resid_post"][0, last])

    # --- final prediction ---
    probs = logits[0, last].softmax(dim=-1)
    top = probs.topk(TOPK_FINAL)
    prediction = [
        {"token": model.to_string([tid]), "prob": float(p)}
        for p, tid in zip(top.values.tolist(), top.indices.tolist())
    ]

    return {
        "name": name,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_model": cfg.d_model,
        "attn_only": bool(cfg.attn_only),
        "prompt": PROMPT,
        "str_tokens": str_tokens,
        "attention": attention,
        "checkpoints": checkpoints,
        "prediction": prediction,
    }


def main():
    print("Exporting visualization data ...")
    data = {"models": [export_model(m) for m in MODELS]}
    with open("data.json", "w") as f:
        json.dump(data, f)
    print("Wrote data.json")
    for m in data["models"]:
        print(f"  {m['name']:14s} top prediction: {m['prediction'][0]['token']!r}")


if __name__ == "__main__":
    main()
