"""Capture reference activations and greedy generations from HF transformers."""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
NEW_TOKENS = 20
OUT = Path("reference/qwen3-0.6b.npz")


def first(x):
    # Some HF modules return tuples (e.g. attention returns (output, weights)).
    return x[0] if isinstance(x, tuple) else x


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    acts = {}
    handles = []

    def save(module, name):
        def hook(module, args, output):
            acts[name] = first(output).detach().float().numpy()

        handles.append(module.register_forward_hook(hook))

    m = model.model
    save(m.embed_tokens, "embed")
    for i, layer in enumerate(m.layers):
        save(layer, f"layer{i}")
    save(m.norm, "final_norm")
    l0 = m.layers[0]
    for name in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]:
        save(getattr(l0, name), f"layer0.{name}")
    for name in ["q_norm", "k_norm"]:
        save(getattr(l0.self_attn, name), f"layer0.self_attn.{name}")

    enc = tok(PROMPT, return_tensors="pt")
    ids = enc.input_ids
    with torch.no_grad():
        logits = model(ids).logits

    # Remove hooks so generation doesn't overwrite the captured activations.
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=False)
    generated = out[:, ids.shape[1] :]

    OUT.parent.mkdir(exist_ok=True)
    np.savez(
        OUT,
        input_ids=ids.numpy(),
        logits=logits.float().numpy(),
        generated=generated.numpy(),
        **acts,
    )

    print(f"prompt: {PROMPT!r}")
    print(f"token ids: {ids[0].tolist()}")
    print(f"saved {len(acts) + 3} arrays to {OUT}")
    print(f"greedy continuation: {tok.decode(generated[0])!r}")


if __name__ == "__main__":
    main()
