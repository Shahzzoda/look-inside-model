"""
Export the FULL computation graph of a forward pass: every intermediate
activation tensor, so the website can render it as an interactive DAG you can
click through and trace backwards.

We use attn-only-2l (2 layers, no MLPs) so the graph stays small enough to see
the whole thing. Writes graph_data.json.
"""

import json
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

PROMPT = "The cat sat on the"   # short + clean single-token words
MODEL = "attn-only-2l"


def r(t):
    """Round a tensor to 3 decimals and return nested python lists (compact JSON)."""
    return t.round(decimals=3).tolist()


def main():
    print(f"loading {MODEL} ...")
    model = HookedTransformer.from_pretrained(MODEL)
    model.eval()
    cfg = model.cfg

    tokens = model.to_tokens(PROMPT, prepend_bos=True)
    str_tokens = model.to_str_tokens(PROMPT, prepend_bos=True)
    last = tokens.shape[1] - 1

    logits, cache = model.run_with_cache(tokens)

    # Collect every activation the graph references. Drop the batch dim (=1).
    acts = {}

    def grab(name):
        acts[name] = r(cache[name][0])

    grab("hook_embed")
    grab("hook_pos_embed")
    for L in range(cfg.n_layers):
        grab(f"blocks.{L}.hook_resid_pre")
        grab(f"blocks.{L}.ln1.hook_normalized")
        grab(f"blocks.{L}.attn.hook_q")            # [pos, head, d_head]
        grab(f"blocks.{L}.attn.hook_k")
        grab(f"blocks.{L}.attn.hook_v")
        # scores/pattern are [head, q, k] (no batch dim now)
        acts[f"blocks.{L}.attn.hook_attn_scores"] = r(cache[f"blocks.{L}.attn.hook_attn_scores"][0])
        acts[f"blocks.{L}.attn.hook_pattern"] = r(cache[f"blocks.{L}.attn.hook_pattern"][0])
        grab(f"blocks.{L}.attn.hook_z")            # [pos, head, d_head]
        grab(f"blocks.{L}.hook_attn_out")
        grab(f"blocks.{L}.hook_resid_post")
    grab("ln_final.hook_normalized")

    # final next-token prediction (top 10) at the last position
    probs = logits[0, last].softmax(dim=-1)
    top = probs.topk(10)
    prediction = [
        {"token": model.to_string([tid]), "prob": round(float(p), 5)}
        for p, tid in zip(top.values.tolist(), top.indices.tolist())
    ]

    out = {
        "model": MODEL,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_head": cfg.d_head,
        "d_model": cfg.d_model,
        "prompt": PROMPT,
        "str_tokens": str_tokens,
        "token_ids": tokens[0].tolist(),
        "acts": acts,
        "prediction": prediction,
    }

    with open("graph_data.json", "w") as f:
        json.dump(out, f)
    size = len(json.dumps(out)) // 1024
    print(f"wrote graph_data.json ({size} KB)")
    print(f"  tokens: {str_tokens}")
    print(f"  prediction: {prediction[0]['token']!r} ({prediction[0]['prob']:.1%})")
    print(f"  activations captured: {len(acts)}")


if __name__ == "__main__":
    main()
