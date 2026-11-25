"""Models verified by the parity suite: pretrained checkpoints and tiny random models."""

# Pretrained checkpoints: test id -> Hugging Face repository.
CHECKPOINTS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",  # hybrid Gated DeltaNet; language model only
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B",
    "minimind-3-moe": "jingyaogong/minimind-3-moe",
    "nano-mistral": "crumb/nano-mistral",  # Mistral; its sliding window exceeds its context
    "gpt2": "openai-community/gpt2",  # GPT-2 small (124M)
    "pythia-160m": "EleutherAI/pythia-160m",  # GPT-NeoX; parallel residual, partial rotary
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
    "tiny-qwen2": ("Qwen2Config", {"tie_word_embeddings": False, "pad_token_id": 1}),
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
    # Qwen3.5: a Gated DeltaNet layer then a gated, partially rotary attention layer, with
    # zero-centred norms; more DeltaNet value heads than key heads; a tied output head.
    "tiny-qwen35": (
        "Qwen3_5TextConfig",
        {
            "head_dim": 32,
            "tie_word_embeddings": True,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "attn_output_gate": True,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
            },
        },
    ),
    # Mistral: Llama-style attention without biases; untied output head; no sliding window.
    "tiny-mistral": ("MistralConfig", {"sliding_window": None, "tie_word_embeddings": False}),
    # Mistral with a 4-token sliding window, shorter than the prompt, so it takes effect.
    "tiny-mistral-window": ("MistralConfig", {"sliding_window": 4, "tie_word_embeddings": False}),
    # GPT-2: learned positions, LayerNorm, fused Conv1D projections, tied output head.
    "tiny-gpt2": ("GPT2Config", {}),
    # GPT-NeoX: fused per-head-interleaved qkv, LayerNorm, GELU MLP, untied output head; a
    # parallel residual and the default partial rotary (a quarter of each 16-dim head: 4 dims).
    "tiny-gpt-neox": ("GPTNeoXConfig", {}),
    # GPT-NeoX with a sequential residual and rotary over the whole head (rotary_pct 1.0).
    "tiny-gpt-neox-sequential": (
        "GPTNeoXConfig",
        {"use_parallel_residual": False, "rotary_pct": 1.0},
    ),
}

# Pretrained encoders: test id -> Hugging Face repository.
ENCODER_CHECKPOINTS = {
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",  # mean pooling
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",  # [CLS] pooling
    "multilingual-e5-small": "intfloat/multilingual-e5-small",  # BERT, multilingual vocab
    "multilingual-e5-base": "intfloat/multilingual-e5-base",  # XLM-RoBERTa, mean pooling
    # MedCPT: no sentence-transformers pipeline; [CLS] pooling, per the model card. The
    # article encoder takes (title, abstract) pairs, so token types 0 and 1 both occur.
    "medcpt-query-encoder": "ncbi/MedCPT-Query-Encoder",
    "medcpt-article-encoder": "ncbi/MedCPT-Article-Encoder",
}

# Pooling for checkpoints without a sentence-transformers pipeline, from their model cards.
# Their reference embeddings come from Transformers, since sentence-transformers would
# silently fall back to mean pooling for them.
ENCODER_POOLING = {
    "medcpt-query-encoder": "cls",
    "medcpt-article-encoder": "cls",
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

# Decoder-based embedding models: test id -> Hugging Face repository.
EMBEDDER_CHECKPOINTS = {
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",  # last-token pooling
}

# Tiny decoder embedders: test id -> (transformers config class, config overrides).
TINY_EMBEDDERS = {
    # A Qwen3 decoder saved without a language-model head, as embedding models are.
    "tiny-qwen3-embedder": ("Qwen3Config", {"tie_word_embeddings": True, "head_dim": 32}),
}

# Hub LoRA adapters: test id -> (base repository, adapter repository, adapter git revision).
# The revision is pinned, so that a later change to the adapter can't change what is verified.
# Each adapter's base has the weights it was trained against: the base repository's weight files
# are unchanged since before the adapter was created (checked from the Hub's file hashes).
ADAPTER_CHECKPOINTS = {
    # Qwen3, q and v only, with dropout; Apache-2.0.
    "qwen3-0.6b-tldr-lora": (
        "Qwen/Qwen3-0.6B",
        "phh/Qwen3-0.6B-TLDR-Lora",
        "0f39534202828516c2645f1aeb8c5e9e9a0f6801",
    ),
    # Every linear layer, on a base whose q/k/v have biases; Apache-2.0.
    "qwen2.5-0.5b-instruct-python-lora": (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "ali4566544/qwen2.5-0.5b-instruct-python-lora",
        "5ecafffc2e23deebae61af586617143e596743a1",
    ),
    # SmolLM2 (Llama architecture), q and v only; the adapter states no licence (the base is
    # Apache-2.0).
    "smollm2-135m-smoltalk-lora": (
        "HuggingFaceTB/SmolLM2-135M",
        "hfm8tr/smollm2-135m-smoltalk-lora",
        "fb39c7012c3b125f94d2bb1a093254994053f4ca",
    ),
    # Llama 3.2 1B, every linear layer, rank-stabilised (use_rslora); Llama 3.2 licence. Its
    # base is the repository registered as `llama-3.2-1b` above.
    "llama-3.2-1b-centaur-lora-r4": (
        "unsloth/Llama-3.2-1B",
        "socius/Llama-Centaur-1B-LoRA-r4",
        "186b1af6c6a30721317e402368cd9326cafd4968",
    ),
}

# Every registered checkpoint: used to validate --checkpoint and to create CI jobs.
ALL_CHECKPOINTS = {
    **CHECKPOINTS,
    **ENCODER_CHECKPOINTS,
    **EMBEDDER_CHECKPOINTS,
    **ADAPTER_CHECKPOINTS,
}
