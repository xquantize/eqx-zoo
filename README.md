<p align="center">
  <img src="assets/banner.svg" alt="eqx-zoo: verified Equinox ports of pretrained models" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <a href="https://github.com/patrick-kidger/equinox"><img src="https://img.shields.io/badge/built%20with-Equinox-8b5cf6" alt="Built with Equinox"></a>
</p>

eqx-zoo provides pretrained models as plain [Equinox](https://github.com/patrick-kidger/equinox) modules, loaded straight from Hugging Face checkpoints and numerically verified against the reference implementation in 🤗 Transformers.

Every model is an ordinary pytree, so `jax.jit`, `jax.grad`, `jax.vmap` and the rest of the JAX ecosystem work on it directly.

## Installation

```bash
pip install "eqx-zoo @ git+https://github.com/xquantize/eqx-zoo"
```

Requires Python 3.12+. The example below also uses `pip install tokenizers`.

## Quick example

```python
import jax.numpy as jnp
from tokenizers import Tokenizer

from eqx_zoo import Qwen3ForCausalLM, generate

tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

prompt = jnp.array(tokenizer.encode("The capital of France is").ids)
tokens = generate(model, prompt, max_new_tokens=30)
print(tokenizer.decode(tokens.tolist()))
# Paris. The capital of Italy is Rome. The capital of Spain is Madrid. ...
```

## Models

| Model | Class |
|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `Qwen3ForCausalLM` |

Each model is tested against Hugging Face activations captured in float32: the output of every layer must match, and greedy generation must reproduce the reference output token for token. See [`tests/`](tests) and [`scripts/make_reference.py`](scripts/make_reference.py).

## Development

```bash
git clone https://github.com/xquantize/eqx-zoo && cd eqx-zoo
uv sync
uv run python scripts/make_reference.py   # capture reference activations
uv run pytest
```

Contributions are welcome. [`AGENTS.md`](AGENTS.md) describes the conventions every model follows.

## See also

- **Neural networks:** [Equinox](https://github.com/patrick-kidger/equinox).
- **Typing:** [jaxtyping](https://github.com/patrick-kidger/jaxtyping).
- **Other JAX model collections:** [Levanter](https://github.com/stanford-crfm/levanter) (LLM training), [Bonsai](https://github.com/jax-ml/bonsai) (Flax NNX), [Equimo](https://github.com/RhizomeResearch/Equimo) (vision).

## License

Apache 2.0. Pretrained weights are distributed by their original authors under their own licenses.
