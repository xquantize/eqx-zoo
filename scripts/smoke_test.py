"""Download and run models with a minimal install: eqx-zoo and only its own dependencies.

CI runs this from the built wheel in an isolated environment, so dependencies resolve as for
a new user (for example huggingface-hub 2.x, which the development lock file excludes).
Correctness is verified by the test suite; this checks that installing and loading work.
"""

import importlib.metadata

import jax.numpy as jnp
import numpy as np

from eqx_zoo import CausalLM, Encoder

for package in ("eqx-zoo", "huggingface-hub", "jax", "equinox", "safetensors"):
    print(f"{package:16s} {importlib.metadata.version(package)}")

# "[CLS] hello world [SEP]" in BERT's uncased vocabulary.
embedding = Encoder.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").embed(
    jnp.array([101, 7592, 2088, 102])
)
assert embedding.shape == (384,), embedding.shape
assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5), "embedding is not normalised"

out = CausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")(jnp.array([1, 2, 3, 4]))
logits = out[0] if isinstance(out, tuple) else out
assert logits.shape == (4, 49152), logits.shape
assert np.isfinite(logits).all(), "logits are not finite"
print("smoke test passed")
