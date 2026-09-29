"""Parity tests for shared layers against Qwen3-0.6B layer 0."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from eqx_zoo.layers import Attention, RMSNorm, SwiGLU, apply_rope, rope_cos_sin

HIDDEN = 1024
INTERMEDIATE = 3072
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
EPS = 1e-6
LAYER0 = "model.layers.0"


def set_weights(module, hf_weight, prefix: str, paths: list[str]):
    """Copy HF tensors into ``module``.

    Each path is both the equinox attribute path and the suffix of the HF name,
    e.g. ``"q_proj.weight"`` under ``prefix="model.layers.0.self_attn"``.
    """
    for path in paths:

        def where(m, path=path):
            for attr in path.split("."):
                m = getattr(m, attr)
            return m

        module = eqx.tree_at(where, module, hf_weight(f"{prefix}.{path}"))
    return module


def assert_close(actual, expected, tol: float):
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=tol, atol=tol)


def make_attention(hf_weight) -> Attention:
    attn = Attention(
        HIDDEN,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=ROPE_THETA,
        eps=EPS,
        qk_norm=True,
        key=jax.random.key(0),
    )
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    return set_weights(attn, hf_weight, f"{LAYER0}.self_attn", [f"{n}.weight" for n in names])


def test_input_layernorm(reference, hf_weight):
    norm = set_weights(RMSNorm(HIDDEN, eps=EPS), hf_weight, f"{LAYER0}.input_layernorm", ["weight"])
    assert_close(norm(jnp.asarray(reference["embed"])), reference["layer0.input_layernorm"], 1e-5)


def test_post_attention_layernorm(reference, hf_weight):
    norm = RMSNorm(HIDDEN, eps=EPS)
    norm = set_weights(norm, hf_weight, f"{LAYER0}.post_attention_layernorm", ["weight"])
    residual = jnp.asarray(reference["embed"] + reference["layer0.self_attn"])
    assert_close(norm(residual), reference["layer0.post_attention_layernorm"], 1e-5)


def test_mlp(reference, hf_weight):
    mlp = SwiGLU(HIDDEN, INTERMEDIATE, key=jax.random.key(0))
    paths = [f"{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    mlp = set_weights(mlp, hf_weight, f"{LAYER0}.mlp", paths)
    out = jax.vmap(mlp)(jnp.asarray(reference["layer0.post_attention_layernorm"]))
    assert_close(out, reference["layer0.mlp"], 1e-4)


def test_rope_properties():
    x = jax.random.normal(jax.random.key(0), (4, 2, HEAD_DIM))
    cos, sin = rope_cos_sin(jnp.arange(4), HEAD_DIM, ROPE_THETA)
    y = apply_rope(x, cos, sin)
    assert_close(y[0], np.asarray(x[0]), 1e-6)  # position 0 is the identity
    assert_close(jnp.linalg.norm(y, axis=-1), np.asarray(jnp.linalg.norm(x, axis=-1)), 1e-5)


def test_qk_norm(reference, hf_weight):
    attn = make_attention(hf_weight)
    x = jnp.asarray(reference["layer0.input_layernorm"])
    q = jax.vmap(attn.q_proj)(x).reshape(-1, NUM_HEADS, HEAD_DIM)
    k = jax.vmap(attn.k_proj)(x).reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    assert_close(attn.q_norm(q), reference["layer0.self_attn.q_norm"], 1e-4)
    assert_close(attn.k_norm(k), reference["layer0.self_attn.k_norm"], 1e-4)


def test_self_attention(reference, hf_weight):
    attn = make_attention(hf_weight)
    x = jnp.asarray(reference["layer0.input_layernorm"])
    out, _ = attn(x, jnp.arange(x.shape[0]))
    assert_close(out, reference["layer0.self_attn"], 1e-4)
