"""Parity tests for shared layers against Qwen3-0.6B layer 0."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from eqx_zoo.layers import RMSNorm, SwiGLU

HIDDEN = 1024
INTERMEDIATE = 3072
EPS = 1e-6
LAYER0 = "model.layers.0"


def load_rmsnorm(hf_weight, name: str) -> RMSNorm:
    norm = RMSNorm(HIDDEN, eps=EPS)
    return eqx.tree_at(lambda m: m.weight, norm, hf_weight(f"{name}.weight"))


def test_input_layernorm(reference, hf_weight):
    norm = load_rmsnorm(hf_weight, f"{LAYER0}.input_layernorm")
    out = norm(jnp.asarray(reference["embed"]))
    np.testing.assert_allclose(out, reference["layer0.input_layernorm"], rtol=1e-5, atol=1e-5)


def test_post_attention_layernorm(reference, hf_weight):
    norm = load_rmsnorm(hf_weight, f"{LAYER0}.post_attention_layernorm")
    residual = reference["embed"] + reference["layer0.self_attn"]
    out = norm(jnp.asarray(residual))
    np.testing.assert_allclose(
        out, reference["layer0.post_attention_layernorm"], rtol=1e-5, atol=1e-5
    )


def test_mlp(reference, hf_weight):
    mlp = SwiGLU(HIDDEN, INTERMEDIATE, key=jax.random.key(0))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        mlp = eqx.tree_at(
            lambda m: getattr(m, proj).weight, mlp, hf_weight(f"{LAYER0}.mlp.{proj}.weight")
        )
    out = jax.vmap(mlp)(jnp.asarray(reference["layer0.post_attention_layernorm"]))
    np.testing.assert_allclose(out, reference["layer0.mlp"], rtol=1e-4, atol=1e-4)
