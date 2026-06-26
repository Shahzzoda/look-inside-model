"""
Local web app: serves the forward-pass visualizer AND an embedded chat panel
that is "Claude with the context of this page."

Why a local server instead of the published Artifact? Artifacts run in a locked
sandbox that blocks all network requests, so a page hosted there cannot call the
Claude API. This Flask app serves the same visualization from your machine and
adds a /chat endpoint that proxies to Claude — your API key stays server-side.

Run it:
    export ANTHROPIC_API_KEY=sk-ant-...      # your key (kept server-side)
    .venv/bin/python app.py
    # then open http://127.0.0.1:5050

Uses the Anthropic Python SDK (claude-opus-4-8, streaming).
"""

import json
import os

import anthropic
from flask import Flask, Response, request


def _load_dotenv(path=".env"):
    """Minimal .env loader (no dependency). Does NOT override an already-set env
    var, and skips blank/placeholder values so a stale template can't shadow a
    real exported key."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and val != "sk-ant-your-key-here" and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the activation data once at startup (same file the page embeds).
# ---------------------------------------------------------------------------
with open("graph_data.json") as f:
    DATA = json.load(f)

# Build the page: graph template + injected data + injected chat widget.
with open("graph_template.html") as f:
    TEMPLATE = f.read()
with open("chat_widget.html") as f:
    WIDGET = f.read()


def build_page():
    page = TEMPLATE.replace("__DATA__", json.dumps(DATA))
    # inject the chat widget just before the tooltip div near the end of the body
    marker = '<div class="tip" id="tip"></div>'
    return page.replace(marker, WIDGET + "\n" + marker, 1)


PAGE = build_page()

# ---------------------------------------------------------------------------
# Anthropic client (reads ANTHROPIC_API_KEY from the environment).
# ---------------------------------------------------------------------------
try:
    CLIENT = anthropic.Anthropic()
    CLIENT_ERR = None
except Exception as e:  # no key set, etc.
    CLIENT = None
    CLIENT_ERR = str(e)

# The SDK constructs even with no key (it validates at request time), so detect
# key presence directly for an honest startup message.
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or getattr(CLIENT, "api_key", None))

MODEL = "claude-opus-4-8"

# A glossary so Claude can map a hook name to what it means in this model.
HOOK_GLOSSARY = {
    "hook_embed": "token embeddings, W_E[token_id]",
    "hook_pos_embed": "learned positional embeddings, W_pos[position]",
    "hook_resid_pre": "residual stream entering the block",
    "ln1.hook_normalized": "LayerNorm of the residual stream: (x-mean)/sqrt(var+eps)",
    "attn.hook_q": "queries per head: LN(x) @ W_Q + b_Q",
    "attn.hook_k": "keys per head: LN(x) @ W_K + b_K",
    "attn.hook_v": "values per head: LN(x) @ W_V + b_V",
    "attn.hook_attn_scores": "Q·K^T / sqrt(d_head), causally masked (before softmax)",
    "attn.hook_pattern": "softmax(scores) over keys — the attention probabilities",
    "attn.hook_z": "pattern · V — attention-weighted sum of values, per head",
    "hook_attn_out": "z concatenated across heads and projected: z @ W_O + b_O",
    "hook_resid_post": "residual stream leaving the block (= resid_pre + attn_out)",
    "ln_final.hook_normalized": "final LayerNorm before unembedding to logits",
}


def stats_for(hook):
    """Compute min/max/mean/L2 over an activation array (any nesting), if present."""
    arr = DATA["acts"].get(hook)
    if arr is None:
        return None
    flat, stack = [], [arr]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        else:
            flat.append(x)
    if not flat:
        return None
    n = len(flat)
    mn, mx = min(flat), max(flat)
    mean = sum(flat) / n
    norm = sum(v * v for v in flat) ** 0.5
    return f"min={mn:.3f}, max={mx:.3f}, mean={mean:.3f}, L2norm={norm:.2f}, count={n}"


def base_system():
    toks = " ".join(DATA["str_tokens"])
    lines = [
        "You are a friendly mechanistic-interpretability tutor embedded inside an "
        "interactive visualizer. The user is a BEGINNER getting into mech interp, "
        "working toward understanding activations and doing feature labeling. Be "
        "concrete, encouraging, and concise (a few short paragraphs max). Prefer "
        "plain language; define jargon the first time you use it. When it helps, "
        "ground your answer in THIS model's real numbers and shapes (given below). "
        "Use backtick `code` for tensor names, shapes, and formulas.",
        "",
        f"THE MODEL: {DATA['model']} — {DATA['n_layers']} layers, "
        f"{DATA['n_heads']} attention heads/layer, d_head={DATA['d_head']}, "
        f"d_model={DATA['d_model']} (the residual-stream width: every token is a "
        f"vector of {DATA['d_model']} numbers). It is attention-only (no MLPs).",
        f"THE PROMPT: \"{DATA['prompt']}\" → tokens: {toks}",
        f"It predicts the next token: \"{DATA['prediction'][0]['token']}\" "
        f"({DATA['prediction'][0]['prob']*100:.0f}%).",
        "",
        "THE VISUALIZER has two tabs:",
        "  • 'Computation graph' — every activation is a node; arrows flow input→prediction; "
        "clicking a node lights up its inputs so you can trace backwards.",
        "  • 'Activation values' — a zoomable numeric grid; step through each stage and read the real values.",
        "",
        "HOOK-NAME GLOSSARY (TransformerLens naming for this model):",
    ]
    for k, v in HOOK_GLOSSARY.items():
        lines.append(f"  - ...{k}: {v}")
    return "\n".join(lines)


BASE_SYSTEM = base_system()


def system_for(context):
    """Augment the base system prompt with whatever the user is currently looking at."""
    extra = []
    tab = (context or {}).get("tab")
    hook = (context or {}).get("hook")
    title = (context or {}).get("title")
    if tab:
        extra.append(f"\nRIGHT NOW the user is on the '{tab}' tab.")
    if hook and title:
        extra.append(f"They have selected the node '{title}' (hook `{hook}`).")
        st = stats_for(hook)
        if st:
            extra.append(f"That activation's real values: {st}.")
        extra.append("If their question is vague ('what is this?', 'what am I "
                     "looking at?'), answer about THIS selected node.")
    return BASE_SYSTEM + "\n" + "\n".join(extra)


@app.route("/")
def index():
    return PAGE


@app.route("/chat", methods=["POST"])
def chat():
    if CLIENT is None:
        return Response(
            f"[Claude isn't configured. Set your API key and restart:  "
            f"export ANTHROPIC_API_KEY=sk-ant-...   ({CLIENT_ERR})]",
            mimetype="text/plain; charset=utf-8",
        )
    payload = request.get_json(force=True)
    messages = payload.get("messages", [])
    system = system_for(payload.get("context"))

    def generate():
        try:
            with CLIENT.messages.stream(
                model=MODEL,
                max_tokens=2048,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except anthropic.AuthenticationError:
            yield "[Authentication failed — check that ANTHROPIC_API_KEY is a valid key.]"
        except anthropic.RateLimitError:
            yield "[Rate limited — wait a moment and try again.]"
        except Exception as e:  # surface anything else to the chat panel
            yield f"[error: {e}]"

    return Response(generate(), mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Forward-pass visualizer + Claude chat")
    print("  → http://127.0.0.1:5050")
    if not HAS_KEY:
        print("  ⚠ ANTHROPIC_API_KEY not set — the chat panel will report an auth error.")
        print("    Set it and restart:  export ANTHROPIC_API_KEY=sk-ant-...")
    else:
        print("  ✓ Claude is configured (model: %s)" % MODEL)
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
