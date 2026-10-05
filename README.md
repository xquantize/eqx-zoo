<p align="center">
  <img src="https://raw.githubusercontent.com/xquantize/eqx-zoo/main/assets/banner.png" alt="eqx-zoo: verified Equinox ports of pretrained models" width="100%">
</p>

<p align="center">
<a href="https://github.com/xquantize/eqx-zoo/actions/workflows/ci.yml"><img src="https://github.com/xquantize/eqx-zoo/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://pypi.org/project/eqx-zoo/"><img src="https://img.shields.io/pypi/v/eqx-zoo" alt="PyPI"></a>
<a href="https://github.com/xquantize/eqx-zoo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
<img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
<a href="https://github.com/patrick-kidger/equinox"><img src="https://img.shields.io/badge/built%20with-Equinox-8b5cf6" alt="Built with Equinox"></a>
</p>

eqx-zoo provides pretrained models as plain [Equinox](https://github.com/patrick-kidger/equinox) modules, loaded straight from Hugging Face checkpoints and numerically verified against their reference implementations: language models against 🤗 Transformers, and sentence-embedding models against [sentence-transformers](https://www.sbert.net). Since Transformers v5 removed its JAX models, eqx-zoo is a verified way to keep using them in JAX.

Every model is an ordinary pytree, so `jax.jit`, `jax.grad`, `jax.vmap` and the rest of the JAX ecosystem work on it directly.

## Installation

```bash
pip install eqx-zoo
```

Requires Python 3.12+. The example below also uses `pip install tokenizers`.

## Quick example

```python
import jax.numpy as jnp
from tokenizers import Tokenizer

from eqx_zoo import CausalLM, generate

tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = CausalLM.from_pretrained("Qwen/Qwen3-0.6B")

prompt = jnp.array(tokenizer.encode("The capital of France is").ids)
tokens = generate(model, prompt, max_new_tokens=30)
print(tokenizer.decode(tokens.tolist()))
# Paris. The capital of Italy is Rome. The capital of Spain is Madrid. ...
```

## Batched generation

Prompts of different lengths are left-padded and generated together. With greedy decoding, each row is identical to generating that prompt on its own.

```python
from eqx_zoo import generate_batch

tokenizer.enable_padding(direction="left")
prompts = ["The capital of France is", "The largest planet in the solar system is"]
batch = tokenizer.encode_batch(prompts)

ids = jnp.array([e.ids for e in batch])
mask = jnp.array([e.attention_mask for e in batch])
tokens = generate_batch(model, ids, mask, max_new_tokens=20)
```

## Sentence embeddings

`Encoder.embed` reads each checkpoint's sentence-transformers configuration and applies its own pooling and normalisation, so embeddings match sentence-transformers.

```python
import jax
import jax.numpy as jnp
from tokenizers import Tokenizer

from eqx_zoo import Encoder

tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer.enable_padding()
model = Encoder.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

batch = tokenizer.encode_batch(["The cat sits on the mat.", "A feline rests on a rug."])
ids = jnp.array([e.ids for e in batch])
mask = jnp.array([e.attention_mask for e in batch])
embeddings = jax.vmap(model.embed)(ids, mask)
print(embeddings @ embeddings.T)  # cosine similarities, as the embeddings are normalised
```

Decoder-based embedding models such as Qwen3-Embedding work the same way, through `DecoderEmbedder`. For search, queries take the instruction prompt from the model's sentence-transformers configuration; documents are embedded as they are.

```python
from eqx_zoo import DecoderEmbedder

model = DecoderEmbedder.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
tokenizer.enable_padding()

prompt = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
batch = tokenizer.encode_batch(
    [prompt + "What is the capital of France?", "Paris is the capital of France."]
)
ids = jnp.array([e.ids for e in batch])
mask = jnp.array([e.attention_mask for e in batch])
query, passage = jax.vmap(model.embed)(ids, mask)
print(query @ passage)  # cosine similarity
```

## Models

Any checkpoint with a supported architecture loads with `CausalLM.from_pretrained` or `Encoder.from_pretrained`, from the Hugging Face Hub or a local directory. These checkpoints are verified by the test suite, and collected on the Hugging Face Hub in [Verified in eqx-zoo](https://huggingface.co/collections/xquantize/verified-in-eqx-zoo-6ac2d731fd1ef3e109541bf7).

### Language models (`CausalLM`)

| Architecture | Verified checkpoints |
|---|---|
| `LlamaForCausalLM` | [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| `Qwen3ForCausalLM` | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| `Qwen2ForCausalLM` | [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| `Qwen3MoeForCausalLM` | [MiniMind-3 MoE](https://huggingface.co/jingyaogong/minimind-3-moe) |

Mixture-of-experts models, including [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), use the same `Qwen3MoeForCausalLM` architecture; experts are computed with grouped matrix multiplications. `generate_batch` currently processes their prompts one after another ([#23](https://github.com/xquantize/eqx-zoo/issues/23)).

### Encoders and embedding models (`Encoder`)

| Architecture | Verified checkpoints |
|---|---|
| `BertModel` | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) |
| `XLMRobertaModel` | [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) |

`RobertaModel` shares the XLM-RoBERTa implementation and is verified on tiny random models.

### Decoder embedding models (`DecoderEmbedder`)

| Architecture | Verified checkpoints |
|---|---|
| `Qwen3ForCausalLM` (saved without a head) | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |

### Verification

Language models are tested against Hugging Face activations in two tiers:

- **float32:** every layer's output must match, and greedy generation must reproduce the reference output token for token.
- **bfloat16:** logits must be about as accurate as Hugging Face's own bfloat16, measured against its float32 output. Exact greedy agreement isn't required in bf16, since it drifts even between Hugging Face's own bf16 and float32 runs.

Encoders and decoder embedders are tested layer by layer against Hugging Face, and their embeddings must match sentence-transformers for a padded batch of sentences, in float32. In bfloat16, their hidden states and embeddings must be about as accurate as the reference libraries' own bfloat16.

On GPUs and TPUs, JAX's default float32 matrix multiplications may use reduced precision internally (for example TF32 on recent NVIDIA GPUs), so float32 results can differ noticeably from a float32 reference. To request true float32, set `jax.config.update("jax_default_matmul_precision", "highest")`, or use `with jax.default_matmul_precision("highest"):` for a block of code. The test suite requests highest precision.

Every architecture is also tested on tiny randomly initialised models, which cover code paths that no single checkpoint exercises. See [`tests/`](https://github.com/xquantize/eqx-zoo/tree/main/tests).

## Development

```bash
git clone https://github.com/xquantize/eqx-zoo && cd eqx-zoo
uv sync
uv run pytest -m "not checkpoint"   # fast: tiny random models, no downloads
uv run pytest                       # full: also downloads and verifies checkpoints
uv run python benchmarks/generation.py Qwen/Qwen3-0.6B   # load, compile and throughput
```

Contributions are welcome. See [`CONTRIBUTING.md`](https://github.com/xquantize/eqx-zoo/blob/main/CONTRIBUTING.md) for setup, conventions and how to add a model.

## See also

- **Neural networks:** [Equinox](https://github.com/patrick-kidger/equinox).
- **Typing:** [jaxtyping](https://github.com/patrick-kidger/jaxtyping).
- **Other JAX model collections:** [Levanter](https://github.com/stanford-crfm/levanter) (LLM training), [Bonsai](https://github.com/jax-ml/bonsai) (Flax NNX), [Equimo](https://github.com/RhizomeResearch/Equimo) (vision).

## License

Apache 2.0. Pretrained weights are distributed by their original authors under their own licenses.
