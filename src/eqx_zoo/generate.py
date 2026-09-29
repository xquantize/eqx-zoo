"""Autoregressive generation for eqx-zoo language models."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray


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
        model: A zoo language model providing `__call__(ids, cache)` and `init_cache`.
        prompt_ids: Token ids of the prompt.
        max_new_tokens: Number of tokens to generate; must be positive.
        temperature: Sampling temperature. `0.0` selects the most likely token (greedy).
        key: PRNG key, required when `temperature > 0`.

    Returns:
        The generated token ids, excluding the prompt.

    Raises:
        ValueError: If `max_new_tokens` is not positive, or sampling without a `key`.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature > 0 and key is None:
        raise ValueError("a PRNG key is required when temperature > 0")
    keys = jax.random.split(jax.random.key(0) if key is None else key, max_new_tokens)

    def pick(logits, k):
        if temperature == 0:
            return jnp.argmax(logits).astype(jnp.int32)
        return jax.random.categorical(k, logits / temperature).astype(jnp.int32)

    cache = model.init_cache(prompt_ids.shape[0] + max_new_tokens)
    logits, cache = model(prompt_ids, cache)
    first = pick(logits[-1], keys[0])

    def step(carry, k):
        token, cache = carry
        logits, cache = model(token[None], cache)
        token = pick(logits[0], k)
        return (token, cache), token

    _, rest = jax.lax.scan(step, (first, cache), keys[1:])
    return jnp.concatenate([first[None], rest])
