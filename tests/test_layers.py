"""Parity tests for shared layers against Qwen3-0.6B layer 0."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from eqx_zoo.layers import (
    Attention,
    Llama3RopeScaling,
    RMSNorm,
    SwiGLU,
    apply_rope,
    causal_mask,
    rope_cos_sin,
    rope_inv_freq,
)

HIDDEN = 1024
INTERMEDIATE = 3072
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
EPS = 1e-6
LAYER0 = "model.layers.0"


def set_weights(module, qwen3_weight, prefix: str, paths: list[str]):
    """Copy HF tensors into ``module``.

    Each path is both the equinox attribute path and the suffix of the HF name,
    e.g. ``"q_proj.weight"`` under ``prefix="model.layers.0.self_attn"``.
    """
    for path in paths:

        def where(m, path=path):
            for attr in path.split("."):
                m = getattr(m, attr)
            return m

        module = eqx.tree_at(where, module, qwen3_weight(f"{prefix}.{path}"))
    return module


def assert_close(actual, expected, tol: float):
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=tol, atol=tol)


def make_attention(qwen3_weight) -> Attention:
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
    return set_weights(attn, qwen3_weight, f"{LAYER0}.self_attn", [f"{n}.weight" for n in names])


def test_input_layernorm(qwen3_reference, qwen3_weight):
    norm = set_weights(
        RMSNorm(HIDDEN, eps=EPS), qwen3_weight, f"{LAYER0}.input_layernorm", ["weight"]
    )
    assert_close(
        norm(jnp.asarray(qwen3_reference["embed"])), qwen3_reference["layer0.input_layernorm"], 1e-5
    )


def test_post_attention_layernorm(qwen3_reference, qwen3_weight):
    norm = RMSNorm(HIDDEN, eps=EPS)
    norm = set_weights(norm, qwen3_weight, f"{LAYER0}.post_attention_layernorm", ["weight"])
    residual = jnp.asarray(qwen3_reference["embed"] + qwen3_reference["layer0.self_attn"])
    assert_close(norm(residual), qwen3_reference["layer0.post_attention_layernorm"], 1e-5)


def test_mlp(qwen3_reference, qwen3_weight):
    mlp = SwiGLU(HIDDEN, INTERMEDIATE, key=jax.random.key(0))
    paths = [f"{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    mlp = set_weights(mlp, qwen3_weight, f"{LAYER0}.mlp", paths)
    out = jax.vmap(mlp)(jnp.asarray(qwen3_reference["layer0.post_attention_layernorm"]))
    assert_close(out, qwen3_reference["layer0.mlp"], 1e-4)


def test_rope_properties():
    x = jax.random.normal(jax.random.key(0), (4, 2, HEAD_DIM))
    cos, sin = rope_cos_sin(jnp.arange(4), HEAD_DIM, ROPE_THETA)
    y = apply_rope(x, cos, sin)
    assert_close(y[0], np.asarray(x[0]), 1e-6)  # position 0 is the identity
    assert_close(jnp.linalg.norm(y, axis=-1), np.asarray(jnp.linalg.norm(x, axis=-1)), 1e-5)


def test_qk_norm(qwen3_reference, qwen3_weight):
    attn = make_attention(qwen3_weight)
    x = jnp.asarray(qwen3_reference["layer0.input_layernorm"])
    q = jax.vmap(attn.q_proj)(x).reshape(-1, NUM_HEADS, HEAD_DIM)
    k = jax.vmap(attn.k_proj)(x).reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    assert_close(attn.q_norm(q), qwen3_reference["layer0.self_attn.q_norm"], 1e-4)
    assert_close(attn.k_norm(k), qwen3_reference["layer0.self_attn.k_norm"], 1e-4)


def test_self_attention(qwen3_reference, qwen3_weight):
    attn = make_attention(qwen3_weight)
    x = jnp.asarray(qwen3_reference["layer0.input_layernorm"])
    positions = jnp.arange(x.shape[0])
    out, _ = attn(x, positions, causal_mask(positions, positions))
    assert_close(out, qwen3_reference["layer0.self_attn"], 1e-4)


def test_llama3_rope_scaling_bands():
    scaling = Llama3RopeScaling(
        factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )
    base = rope_inv_freq(64, 500_000.0)
    scaled = rope_inv_freq(64, 500_000.0, scaling)
    wavelen = 2 * np.pi / np.asarray(base)
    high, low = wavelen < 8192 / 4.0, wavelen > 8192 / 1.0
    assert high.any() and low.any()
    assert_close(scaled[high], np.asarray(base[high]), 1e-6)  # high frequencies unchanged
    assert_close(scaled[low], np.asarray(base[low] / 32.0), 1e-6)  # low frequencies scaled
