"""Capture reference activations from Hugging Face transformers."""

import numpy as np

PROMPT = "The capital of France is"
NEW_TOKENS = 20


def capture(repo_id: str) -> dict[str, np.ndarray]:
    """Run the Hugging Face model in float32 and record activations and a greedy continuation.

    The batch axis is removed from every array.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    acts: dict[str, np.ndarray] = {}
    handles = []

    def record(module, name):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            acts[name] = out[0].detach().float().numpy()

        handles.append(module.register_forward_hook(hook))

    m = model.model
    record(m.embed_tokens, "embed")
    for i, layer in enumerate(m.layers):
        record(layer, f"layer{i}")
    record(m.norm, "final_norm")
    l0 = m.layers[0]
    for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
        record(getattr(l0, name), f"layer0.{name}")
    for name in ("q_norm", "k_norm"):
        if hasattr(l0.self_attn, name):
            record(getattr(l0.self_attn, name), f"layer0.self_attn.{name}")

    enc = tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc.input_ids).logits

    # Remove hooks so generation doesn't overwrite the captured activations.
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=False)

    return {
        "input_ids": enc.input_ids[0].numpy(),
        "logits": logits[0].float().numpy(),
        "generated": out[0, enc.input_ids.shape[1] :].numpy(),
        **acts,
    }
