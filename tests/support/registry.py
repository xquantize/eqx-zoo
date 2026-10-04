"""Models verified by the parity suite: pretrained checkpoints and tiny random models."""

# Pretrained checkpoints: test id -> Hugging Face repository.
CHECKPOINTS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B",
    "minimind-3-moe": "jingyaogong/minimind-3-moe",
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
    # Routed experts (2 of 8, renormalised), a dense layer mixed in via `mlp_only_layers`,
    # and an untied output head.
    "tiny-qwen3-moe": (
        "Qwen3MoeConfig",
        {
            "tie_word_embeddings": False,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "norm_topk_prob": True,
            "mlp_only_layers": [1],
        },
    ),
}

# Pretrained encoders: test id -> Hugging Face repository.
ENCODER_CHECKPOINTS = {
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",  # mean pooling
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",  # [CLS] pooling
    "multilingual-e5-small": "intfloat/multilingual-e5-small",  # BERT, multilingual vocab
    "multilingual-e5-base": "intfloat/multilingual-e5-base",  # XLM-RoBERTa, mean pooling
}

# Tiny randomly initialised encoders: test id -> (transformers config class, config overrides).
TINY_ENCODERS = {
    # BERT: post-norm LayerNorm, learned positions, token types, biases everywhere.
    "tiny-bert": ("BertConfig", {}),
    # RoBERTa: positions count only non-padding tokens, from pad_token_id + 1; one token type.
    # PROMPT_IDS starts with id 1, the pad id here, so padding-aware positions are exercised.
    "tiny-roberta": ("RobertaConfig", {"type_vocab_size": 1, "pad_token_id": 1}),
    "tiny-xlm-roberta": ("XLMRobertaConfig", {"type_vocab_size": 1, "pad_token_id": 1}),
}
