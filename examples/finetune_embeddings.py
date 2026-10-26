"""Fine-tune a sentence-embedding model in JAX, then use it from sentence-transformers.

A minimal end-to-end example: contrastive training with in-batch negatives (the loss
sentence-transformers calls MultipleNegativesRankingLoss) on a handful of query/passage
pairs, with Optax; then `save_pretrained`, and a check that sentence-transformers reproduces
the fine-tuned embeddings. Eight pairs show the mechanics; real fine-tuning needs a real
dataset and a held-out evaluation.

Run with:
    uv run --with optax python examples/finetune_embeddings.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tokenizers import Tokenizer

from eqx_zoo import Encoder

REPO = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT = "finetuned-minilm"
PAIRS = [
    ("When does the ferry leave?", "The ferry to the island departs at nine every morning."),
    ("Who kept the logbook?", "Each lighthouse keeper wrote a few lines in the logbook nightly."),
    ("Where are the old logbooks now?", "The logbooks were moved to the town library."),
    ("What changed on the piers?", "The wooden piers were rebuilt in concrete."),
    ("Why was the lamp cleaned twice a night?", "In winter, salt spray covered the lamp."),
    ("What filled the lanes behind the market?", "Cafes and small shops filled the narrow lanes."),
    ("What did keepers record?", "Wind direction, wave height and passing ships were logged."),
    ("When was the light automated?", "The lighthouse light was automated after a century."),
]

tokenizer = Tokenizer.from_pretrained(REPO)
tokenizer.enable_padding(length=32)  # a fixed length, so the training step compiles once
tokenizer.enable_truncation(32)


def encode(texts):
    """Tokenize texts into padded token ids and attention masks."""
    batch = tokenizer.encode_batch(texts)
    return jnp.array([e.ids for e in batch]), jnp.array([e.attention_mask for e in batch])


def loss_fn(model, queries, passages):
    """In-batch negatives: each query should match its own passage best."""
    q = jax.vmap(model.embed)(*queries)
    p = jax.vmap(model.embed)(*passages)
    scores = 20.0 * q @ p.T  # cosine similarities (embeddings are normalised), scaled
    return optax.softmax_cross_entropy_with_integer_labels(scores, jnp.arange(len(q))).mean()


model = Encoder.from_pretrained(REPO)
queries = encode([q for q, _ in PAIRS])
passages = encode([p for _, p in PAIRS])

optimizer = optax.adamw(2e-5)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, queries, passages):
    """One optimisation step: compute the loss and its gradients, then update the model."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, queries, passages)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), opt_state, loss


for step in range(30):
    model, opt_state, loss = train_step(model, opt_state, queries, passages)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  loss {float(loss):.4f}")

# Save in Hugging Face / sentence-transformers format; the tokenizer is saved separately.
model.save_pretrained(OUTPUT)
from transformers import AutoTokenizer  # noqa: E402

AutoTokenizer.from_pretrained(REPO).save_pretrained(OUTPUT)

# sentence-transformers loads the fine-tuned model and reproduces our embeddings.
from sentence_transformers import SentenceTransformer  # noqa: E402

ours = np.asarray(jax.vmap(model.embed)(*queries))
theirs = SentenceTransformer(OUTPUT, device="cpu").encode([q for q, _ in PAIRS])
print(f"sentence-transformers vs eqx-zoo, max difference: {np.abs(ours - theirs).max():.1e}")
