"""Autoregressive generation for eqx-zoo language models."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from eqx_zoo.layers import SparseMoE


def _check_args(max_new_tokens: int, temperature: float, key: PRNGKeyArray | None) -> None:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature > 0 and key is None:
        raise ValueError("a PRNG key is required when temperature > 0")


def _contains_moe(model) -> bool:
    """Whether `model` contains a mixture-of-experts layer anywhere in its module tree."""
    is_moe = lambda node: isinstance(node, SparseMoE)  # noqa: E731
    return any(is_moe(node) for node in jax.tree.leaves(model, is_leaf=is_moe))


def _generate(model, prompt_ids, prompt_mask, max_new_tokens, temperature, key):
    """Generate for a single prompt; `prompt_mask` marks real tokens, or is `None`."""
    keys = jax.random.split(key, max_new_tokens)

    def pick(logits, k):
        if temperature == 0:
            return jnp.argmax(logits).astype(jnp.int32)
        return jax.random.categorical(k, logits / temperature).astype(jnp.int32)

    cache = model.init_cache(prompt_ids.shape[0] + max_new_tokens)
    logits, cache = model(prompt_ids, cache, attention_mask=prompt_mask)
    first = pick(logits[-1], keys[0])

    def step(carry, k):
        token, cache = carry
        logits, cache = model(token[None], cache)
        token = pick(logits[0], k)
        return (token, cache), token

    _, rest = jax.lax.scan(step, (first, cache), keys[1:])
    return jnp.concatenate([first[None], rest])


@eqx.filter_jit
def generate(
    model,
    prompt_ids: Int[Array, " prompt"],
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> Int[Array, " max_new_tokens"]:
    """Generate tokens autoregressively with a key/value cache.

    The prompt is processed in a single forward pass, then each new token is produced
    by one cached decoding step inside `jax.lax.scan`. Generation always runs for
    `max_new_tokens` steps; truncate at an end-of-sequence token afterwards if needed.

    Args:
        model: A zoo language model providing `__call__(ids, cache, attention_mask)` and
            `init_cache`.
        prompt_ids: Token ids of the prompt.
        max_new_tokens: Number of tokens to generate; must be positive.
        temperature: Sampling temperature. `0.0` selects the most likely token (greedy).
        key: PRNG key, required when `temperature > 0`.

    Returns:
        The generated token ids, excluding the prompt.

    Raises:
        ValueError: If `max_new_tokens` is not positive, or sampling without a `key`.
    """
    _check_args(max_new_tokens, temperature, key)
    key = jax.random.key(0) if key is None else key
    return _generate(model, prompt_ids, None, max_new_tokens, temperature, key)


@eqx.filter_jit
def generate_batch(
    model,
    prompt_ids: Int[Array, "batch prompt"],
    prompt_mask: Bool[Array, "batch prompt"] | Int[Array, "batch prompt"],
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> Int[Array, "batch max_new_tokens"]:
    """Generate tokens for a batch of left-padded prompts.

    Padding is ignored by attention and does not advance positions, so with greedy
    decoding each row is identical to calling `generate` on that prompt alone. Prompts
    must be left-padded, so that each prompt's last token is a real token.

    Models with mixture-of-experts layers generate their prompts one after another rather
    than in parallel; results are identical, but batching gives no speed-up for them yet.

    Args:
        model: A zoo language model providing `__call__(ids, cache, attention_mask)` and
            `init_cache`.
        prompt_ids: Left-padded token ids, one row per prompt.
        prompt_mask: `1` or `True` for real tokens and `0` or `False` for padding.
        max_new_tokens: Number of tokens to generate for each prompt; must be positive.
        temperature: Sampling temperature. `0.0` selects the most likely token (greedy).
        key: PRNG key, required when `temperature > 0`. It is split into one key per
            prompt, so sampled rows differ from separate `generate` calls.

    Returns:
        The generated token ids, one row per prompt, excluding the prompts.

    Raises:
        ValueError: If `prompt_ids` and `prompt_mask` are not both of shape
            `(batch, prompt)`, if `max_new_tokens` is not positive, or sampling without a
            `key`.
    """
    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise ValueError("prompt_ids and prompt_mask must both have shape (batch, prompt)")
    _check_args(max_new_tokens, temperature, key)
    keys = jax.random.split(jax.random.key(0) if key is None else key, prompt_ids.shape[0])

    def generate_one(ids, mask, k):
        return _generate(model, ids, mask, max_new_tokens, temperature, k)

    if _contains_moe(model):
        # `jax.lax.ragged_dot`, used by mixture-of-experts layers, cannot yet be vmapped,
        # so these prompts are generated one after another instead of together.
        return jax.lax.map(lambda args: generate_one(*args), (prompt_ids, prompt_mask, keys))
    return jax.vmap(generate_one)(prompt_ids, prompt_mask, keys)
