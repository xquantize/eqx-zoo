"""Capture reference activations from Hugging Face transformers."""

import numpy as np

PROMPT = "The capital of France is"
NEW_TOKENS = 20


def capture(repo_id: str) -> dict[str, np.ndarray]:
    """Load a Hub checkpoint in float32 and record its reference activations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    # Cast after loading rather than passing a dtype: the keyword is `torch_dtype` in older
    # transformers and `dtype` in newer ones. Upcasting the stored weights is exact.
    model = AutoModelForCausalLM.from_pretrained(repo_id, attn_implementation="eager")
    model = model.float().eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    return {**capture_model(model, ids), "bf16_logits": bf16_logits(model, ids)}


def capture_model(model, input_ids) -> dict[str, np.ndarray]:
    """Record activations and a greedy continuation for `input_ids` of shape (1, seq).

    The batch axis is removed from every array.
    """
    import torch

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

    with torch.no_grad():
        logits = model(input_ids).logits

    # Remove hooks so generation doesn't overwrite the captured activations.
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
        )

    return {
        "input_ids": input_ids[0].numpy(),
        "logits": logits[0].float().numpy(),
        "generated": out[0, input_ids.shape[1] :].numpy(),
        **acts,
    }


def bf16_logits(model, input_ids) -> np.ndarray:
    """Logits after casting `model` to bfloat16 in place: the yardstick for bf16 accuracy."""
    import torch

    model = model.to(torch.bfloat16)
    with torch.no_grad():
        return model(input_ids).logits[0].float().numpy()


def capture_encoder(model, input_ids) -> dict[str, np.ndarray]:
    """Record an encoder's embeddings, every layer's output and its final hidden states.

    The batch axis is removed from every array.
    """
    import torch

    acts: dict[str, np.ndarray] = {}
    handles = []

    def record(module, name):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            acts[name] = out[0].detach().float().numpy()

        handles.append(module.register_forward_hook(hook))

    record(model.embeddings, "embed")
    for i, layer in enumerate(model.encoder.layer):
        record(layer, f"layer{i}")
    with torch.no_grad():
        hidden = model(input_ids).last_hidden_state
    for handle in handles:
        handle.remove()

    return {"input_ids": input_ids[0].numpy(), "hidden": hidden[0].float().numpy(), **acts}
