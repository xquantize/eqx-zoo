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


def bf16_hidden(model, input_ids) -> np.ndarray:
    """Final hidden states after casting `model` to bfloat16 in place: the bf16 yardstick."""
    import torch

    model = model.to(torch.bfloat16)
    with torch.no_grad():
        return model(input_ids).last_hidden_state[0].float().numpy()


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


SENTENCES = [
    "The capital of France is Paris.",
    "A quick brown fox jumps over the lazy dog, again and again and again.",
    "Hello!",
]


def capture_sentence_transformers(repo_id: str) -> dict[str, np.ndarray]:
    """Capture sentence-transformers' tokenization and embeddings, in float32 and bf16."""
    import torch
    from sentence_transformers import SentenceTransformer

    # Load in float32 explicitly: sentence-transformers otherwise uses the checkpoint's
    # stored dtype, which would make a bf16-stored model's reference bf16.
    st = SentenceTransformer(repo_id, device="cpu").to(torch.float32)
    # `tokenize` was renamed `preprocess`; older releases only have `tokenize`.
    tokenize = st.preprocess if hasattr(st, "preprocess") else st.tokenize
    features = tokenize(SENTENCES)
    types = features.get("token_type_ids", torch.zeros_like(features["input_ids"]))
    embeddings = st.encode(SENTENCES, convert_to_numpy=True)
    bf16 = st.to(torch.bfloat16).encode(SENTENCES, convert_to_tensor=True).float().numpy()
    return {
        "st_input_ids": features["input_ids"].numpy(),
        "st_attention_mask": features["attention_mask"].numpy(),
        "st_token_type_ids": types.numpy(),
        "st_embeddings": embeddings,
        "st_bf16_embeddings": bf16,
    }


def capture_encoder_checkpoint(repo_id: str) -> dict[str, np.ndarray]:
    """Capture an encoder's HF activations and its sentence-transformers embeddings."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id, attn_implementation="eager").float().eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    return {
        **capture_encoder(model, ids),
        "bf16_hidden": bf16_hidden(model, ids),
        **capture_sentence_transformers(repo_id),
    }


def capture_decoder(model, input_ids) -> dict[str, np.ndarray]:
    """Record a headless decoder's embeddings, every layer's output and final hidden states.

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

    record(model.embed_tokens, "embed")
    for i, layer in enumerate(model.layers):
        record(layer, f"layer{i}")
    with torch.no_grad():
        hidden = model(input_ids).last_hidden_state
    for handle in handles:
        handle.remove()

    return {"input_ids": input_ids[0].numpy(), "hidden": hidden[0].float().numpy(), **acts}


def capture_embedder_checkpoint(repo_id: str) -> dict[str, np.ndarray]:
    """Capture a decoder embedder's HF activations and its sentence-transformers embeddings."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id, attn_implementation="eager").float().eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    return {
        **capture_decoder(model, ids),
        "bf16_hidden": bf16_hidden(model, ids),
        **capture_sentence_transformers(repo_id),
    }


# An original passage used twice: repeated text gives many confident predictions in its second
# half, where position bugs are glaring. Positions beyond 256 matter because bf16 represents
# integers exactly only up to 256.
LONG_TEXT = (
    "The old lighthouse stood at the edge of the harbour for more than a century, and in that "
    "time it watched the town change around it. Fishing boats gave way to ferries, the wooden "
    "piers were rebuilt in concrete, and the narrow lanes behind the market filled with cafes "
    "and small shops. Every evening, the keeper climbed the spiral staircase, checked the lamp, "
    "and wrote a few lines in a logbook that had been kept since the first night the light was "
    "lit. The entries were short: the direction of the wind, the height of the waves, the ships "
    "that passed, and sometimes a note about a storm that kept the town awake. When the light "
    "was finally automated, the logbooks were moved to the town library, where students now "
    "read them to learn how the weather, the trade, and the people of the coast had changed "
    "over the years. Some of them noticed that the handwriting changed every few decades, as "
    "one keeper retired and another took over, but the habit of recording each night never "
    "stopped. In winter the storms were long and fierce, and the keepers wrote about waves "
    "breaking over the harbour wall, about boats that came home late, and about the lamp that "
    "had to be cleaned twice a night because of the salt. In summer the entries were calmer, "
    "describing clear evenings, visiting yachts, and children who came to watch the sunset from "
    "the rocks below the tower."
)
LONG_LENGTH = 512


def capture_long_decisions(repo_id: str) -> dict[str, np.ndarray]:
    """Float32 decisions and their margins on a long input, and Hugging Face's bf16 noise.

    Stores, for each position, float32's top-1 token and the gap between its top two logits,
    plus sigma: the RMS difference between Hugging Face's bf16 and float32 logits.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ids = AutoTokenizer.from_pretrained(repo_id)(LONG_TEXT + " " + LONG_TEXT).input_ids
    assert len(ids) >= LONG_LENGTH, f"long input has only {len(ids)} tokens"
    ids = ids[:LONG_LENGTH]
    model = AutoModelForCausalLM.from_pretrained(repo_id, attn_implementation="eager")
    model = model.float().eval()
    x = torch.tensor([ids])
    with torch.no_grad():
        fp32 = model(x).logits[0]
        bf16 = model.to(torch.bfloat16)(x).logits[0].float()
    top2 = fp32.topk(2).values
    return {
        "long_input_ids": np.array(ids),
        "long_top1": fp32.argmax(-1).numpy(),
        "long_margin": (top2[:, 0] - top2[:, 1]).numpy(),
        "long_sigma": np.array(float((bf16 - fp32).pow(2).mean().sqrt())),
    }


def capture_gradients(model, input_ids) -> dict[str, np.ndarray]:
    """Gradients of the mean next-token cross-entropy, as `grad.<parameter name>`.

    Uses the loss transformers computes with `labels=input_ids`. Call this before anything
    converts `model` to another dtype.
    """
    model.zero_grad()
    model(input_ids, labels=input_ids).loss.backward()
    grads = {
        f"grad.{name}": param.grad.detach().float().numpy()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    model.zero_grad()
    return grads


def capture_hidden_gradients(model, input_ids) -> dict[str, np.ndarray]:
    """Gradients of sum(hidden * probe), for models without a language-model head.

    The probe is a fixed random array the shape of the final hidden states, stored as
    `grad_probe` so the JAX side uses the same one. Call before anything converts `model`
    to another dtype.
    """
    import torch

    model.zero_grad()
    hidden = model(input_ids).last_hidden_state[0]
    probe = np.random.default_rng(0).normal(size=tuple(hidden.shape)).astype(np.float32)
    (hidden * torch.from_numpy(probe)).sum().backward()
    grads = {
        f"grad.{name}": param.grad.detach().float().numpy()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    model.zero_grad()
    return {"grad_probe": probe, **grads}
