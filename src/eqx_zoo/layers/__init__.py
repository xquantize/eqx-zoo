"""Building blocks shared across model families.

All modules operate on a single, unbatched sequence; use `jax.vmap` to batch them.
Attribute names mirror the corresponding Hugging Face checkpoint parameters so that
pretrained weights can be loaded by name.
"""

from eqx_zoo.layers.attention import Attention, Cache, KVCache, causal_mask
from eqx_zoo.layers.mlp import SwiGLU
from eqx_zoo.layers.moe import SparseMoE
from eqx_zoo.layers.norm import LayerNorm, RMSNorm
from eqx_zoo.layers.pooling import cls_pool, l2_normalize, mean_pool
from eqx_zoo.layers.rope import Llama3RopeScaling, apply_rope, rope_cos_sin, rope_inv_freq

__all__ = [
    "Attention",
    "Cache",
    "KVCache",
    "causal_mask",
    "Llama3RopeScaling",
    "LayerNorm",
    "RMSNorm",
    "cls_pool",
    "l2_normalize",
    "mean_pool",
    "SwiGLU",
    "apply_rope",
    "rope_cos_sin",
    "rope_inv_freq",
    "SparseMoE",
]
