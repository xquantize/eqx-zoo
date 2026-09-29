"""Capture reference activations from HF transformers for parity tests."""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
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

    def save(name):
        def hook(module, args, output):
            acts[name] = first(output).detach().float().numpy()

        return hook

    m = model.model
    m.embed_tokens.register_forward_hook(save("embed"))
    for i, layer in enumerate(m.layers):
        layer.register_forward_hook(save(f"layer{i}"))
    m.norm.register_forward_hook(save("final_norm"))

    l0 = m.layers[0]
    for name in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]:
        getattr(l0, name).register_forward_hook(save(f"layer0.{name}"))
    for name in ["q_norm", "k_norm"]:
        getattr(l0.self_attn, name).register_forward_hook(save(f"layer0.self_attn.{name}"))

    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits

    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, input_ids=ids.numpy(), logits=logits.float().numpy(), **acts)

    print(f"prompt: {PROMPT!r}")
    print(f"token ids: {ids[0].tolist()}")
    print(f"saved {len(acts) + 2} arrays to {OUT}")
    for k in ["embed", "layer0.self_attn.q_norm", "layer0.self_attn.k_norm", "layer0", "final_norm"]:
        print(f"  {k:28s} {acts[k].shape}")
    print(f"  {'logits':28s} {tuple(logits.shape)}")

    top = torch.topk(logits[0, -1], 5)
    print("top-5 next tokens:")
    for p, i in zip(torch.softmax(logits[0, -1], -1)[top.indices], top.indices):
        print(f"  {tok.decode([int(i)])!r:14s} p={p:.3f}")


if __name__ == "__main__":
    main()
