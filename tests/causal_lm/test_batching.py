"""Batched generation tests: each prompt must generate exactly as it would alone."""

import jax.numpy as jnp
import numpy as np
import pytest

from eqx_zoo import generate, generate_batch

NEW_TOKENS = 8


def left_pad_batch(prompts: list[list[int]]):
    """Left-pad prompts to a common length, returning ids and their attention mask."""
    width = max(len(p) for p in prompts)
    ids = np.zeros((len(prompts), width), dtype=np.int32)
    mask = np.zeros((len(prompts), width), dtype=bool)
    for row, prompt in enumerate(prompts):
        ids[row, width - len(prompt) :] = prompt
        mask[row, width - len(prompt) :] = True
    return jnp.asarray(ids), jnp.asarray(mask)


def test_batch_matches_individual(case):
    full = case.reference["input_ids"].tolist()
    prompts = [full, full[:2], full[1:4]]  # different lengths, so padding varies per row
    ids, mask = left_pad_batch(prompts)

    batch = generate_batch(case.model, ids, mask, NEW_TOKENS)
    for row, prompt in enumerate(prompts):
        alone = generate(case.model, jnp.asarray(prompt), NEW_TOKENS)
        np.testing.assert_array_equal(batch[row], alone, err_msg=f"{case.name}: prompt {row}")


def test_batch_of_one(case):
    prompt = jnp.asarray(case.reference["input_ids"])
    mask = jnp.ones((1, prompt.shape[0]), dtype=bool)
    batch = generate_batch(case.model, prompt[None], mask, NEW_TOKENS)
    np.testing.assert_array_equal(batch[0], generate(case.model, prompt, NEW_TOKENS))


def test_batch_validation(case):
    ids = jnp.zeros((2, 4), dtype=jnp.int32)
    with pytest.raises(ValueError, match="shape"):
        generate_batch(case.model, ids, jnp.ones((2, 3), dtype=bool), NEW_TOKENS)
    with pytest.raises(ValueError, match="PRNG key"):
        generate_batch(case.model, ids, jnp.ones((2, 4), dtype=bool), NEW_TOKENS, temperature=0.7)
