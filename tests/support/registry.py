"""Models verified by the parity suite: pretrained checkpoints and tiny random models."""

# Pretrained checkpoints: test id -> Hugging Face repository.
CHECKPOINTS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B",
}

# Tiny randomly initialised models: test id -> (transformers config class, config overrides).
# Each exercises specific code paths; the shared base config is in `support/tiny.py`.
TINY_MODELS = {
    # Base Llama: no bias, no q/k norm, plain RoPE, tied output head.
    "tiny-llama": ("LlamaConfig", {"tie_word_embeddings": True}),
    # Llama 3 RoPE scaling. A short original context makes all three frequency bands active.
    "tiny-llama3": (
        "LlamaConfig",
        {
            "tie_word_embeddings": True,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 32,
            },
        },
    ),
    # q/k/v bias and an untied output head.
    "tiny-qwen2": ("Qwen2Config", {"tie_word_embeddings": False}),
    # Per-head q/k norm, with head_dim (32) != hidden_size / num_attention_heads (16).
    "tiny-qwen3": ("Qwen3Config", {"tie_word_embeddings": True, "head_dim": 32}),
}
