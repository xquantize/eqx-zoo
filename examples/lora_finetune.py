"""Fine-tune a LoRA adapter on a language model in JAX, then use it from PEFT.

A minimal end-to-end example: next-token training of a LoRA adapter on a handful of short
texts with Optax, only the adapter's parameters being trained; then `save_adapter`, and a check
that PEFT loads the adapter and reproduces our logits. Six texts show the mechanics; real
fine-tuning needs a real dataset and a held-out evaluation. PEFT and Transformers are
development dependencies of eqx-zoo, so `uv run` has them.

Run with:
    uv run --with optax python examples/lora_finetune.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from peft import PeftModel
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from eqx_zoo import CausalLM, lora

REPO = "HuggingFaceTB/SmolLM2-135M"
OUTPUT = "lora-smollm2"
TEXTS = [
    "The lighthouse keeper wrote the wind and the waves in the logbook every night.",
    "In winter, salt spray covered the lamp, so it was cleaned twice a night.",
    "The ferry to the island leaves at nine every morning, and returns at dusk.",
    "The old logbooks were moved to the town library, where anyone may read them.",
    "The wooden piers were rebuilt in concrete, and cafes filled the narrow lanes.",
    "After a century the lighthouse light was automated, and the keepers left.",
]
LENGTH = 32

tokenizer = Tokenizer.from_pretrained(REPO)
tokenizer.enable_padding(length=LENGTH)  # a fixed length, so the training step compiles once
tokenizer.enable_truncation(LENGTH)


def encode(texts):
    """Tokenize texts into padded token ids and attention masks (1 for real tokens)."""
    batch = tokenizer.encode_batch(texts)
    return jnp.array([e.ids for e in batch]), jnp.array([e.attention_mask for e in batch])


def loss_fn(trainable, frozen, ids, mask):
    """Mean next-token cross-entropy over the real tokens of a batch."""
    model = eqx.combine(trainable, frozen)
    logits = jax.vmap(lambda i, m: model(i, attention_mask=m)[0])(ids, mask)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:])
    weights = mask[:, 1:].astype(losses.dtype)  # padding is not a target
    return (losses * weights).sum() / weights.sum()


model = CausalLM.from_pretrained(REPO)
# Rank-8 LoRA on every linear layer but the output head (PEFT's "all-linear").
model = lora.add_adapter(model, r=8, alpha=16, target_modules="all-linear", key=jax.random.key(0))
# `trainable` marks the adapter's arrays; everything else stays frozen.
trainable, frozen = eqx.partition(model, lora.trainable(model))
print(f"training {sum(x.size for x in jax.tree.leaves(trainable)):,} adapter parameters")
ids, mask = encode(TEXTS)

optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(trainable)


@eqx.filter_jit
def train_step(trainable, frozen, opt_state, ids, mask):
    """One optimisation step: compute the loss and its gradients, then update the adapter."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(trainable, frozen, ids, mask)
    updates, opt_state = optimizer.update(grads, opt_state, trainable)
    return eqx.apply_updates(trainable, updates), opt_state, loss


for step in range(20):
    trainable, opt_state, loss = train_step(trainable, frozen, opt_state, ids, mask)
    if step % 5 == 0 or step == 19:
        print(f"step {step:2d}  loss {float(loss):.4f}")
model = eqx.combine(trainable, frozen)

# Save the adapter in PEFT's format. It holds only the adapter, not the base model.
lora.save_adapter(model, OUTPUT)

# PEFT loads the base model and our adapter, and reproduces our logits.
base = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
peft_model = PeftModel.from_pretrained(base, OUTPUT).eval()
prompt = ids[0][: int(mask[0].sum())]  # the first text, without padding
ours = np.asarray(model(prompt)[0])
with torch.no_grad():
    theirs = peft_model(torch.tensor(np.asarray(prompt))[None]).logits[0].numpy()
error, largest = np.abs(ours - theirs).max(), np.abs(ours).max()
print(f"PEFT vs eqx-zoo logits, max difference: {error:.1e} (logits up to {largest:.0f})")
