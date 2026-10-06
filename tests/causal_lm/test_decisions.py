"""Margin-aware decision tests on a long input.

bf16 legitimately flips near-tied decisions, so a decision has to match float32's only where
float32 is confident: where its top two logits differ by more than CONFIDENT times sigma,
Hugging Face's own bf16 logit noise on the same input. Measured on 512 tokens: correct code
flips margins up to 4.8 sigma (Hugging Face's own bf16, up to 3.4 sigma), while passing
positions through bf16, which is exact only up to 256, flips margins of 12-209 sigma, a bug
the short-prompt tests don't see. Teacher-forced cached decoding checks the prefill-then-decode
path without one flipped decision cascading into the rest.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from support.bf16 import MODES, run

CONFIDENT = 8.0  # in units of sigma; see the module docstring
PREFILL = 8


def logits_of(model, ids):
    out = model(ids)
    return out[0] if isinstance(out, tuple) else out


def cached_teacher_forced(model, ids):
    """Prefill a few tokens, then feed the rest through the cache one at a time."""
    cache = model.init_cache(ids.shape[0])
    first, cache = model(ids[:PREFILL], cache)

    def step(cache, token):
        logits, cache = model(token[None], cache)
        return cache, logits[0]

    _, rest = jax.lax.scan(step, cache, ids[PREFILL:])
    return jnp.concatenate([first, rest])


def long_input(case) -> jnp.ndarray:
    if "long_input_ids" not in case.reference:
        pytest.skip("no long-input reference for this model")
    return jnp.asarray(case.reference["long_input_ids"])


def check_decisions(case, logits: np.ndarray) -> None:
    ref = case.reference
    sigma = float(ref["long_sigma"])
    confident = ref["long_margin"] > CONFIDENT * sigma
    wrong = np.flatnonzero(confident & (logits.argmax(-1) != ref["long_top1"]))
    if len(wrong):
        first = wrong[0]
        raise AssertionError(
            f"{case.name}: {len(wrong)} confident decisions differ from float32; the first is at "
            f"position {first}, where float32's margin is {ref['long_margin'][first] / sigma:.1f} "
            f"sigma (limit {CONFIDENT} sigma)"
        )


@pytest.mark.parametrize("jit", MODES)
def test_bf16_long_decisions(case, bf16_model, jit):
    check_decisions(case, run(bf16_model, logits_of, long_input(case), jit=jit))


def test_bf16_long_cached_decisions(case, bf16_model):
    check_decisions(case, run(bf16_model, cached_teacher_forced, long_input(case), jit=True))
