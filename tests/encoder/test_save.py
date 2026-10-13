"""Saving encoders, verified by loading them with transformers and sentence-transformers."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from support.reference import SENTENCES
from support.registry import TINY_ENCODERS
from transformers import AutoModel

TOKENIZER_FILES = [
    "tokenizer*",
    "vocab*",
    "special_tokens_map.json",
    "sentencepiece*",
    "merges.txt",
]


def test_transformers_loads_saved_encoder(case, tmp_path):
    """Hugging Face loads what we save, and computes the reference hidden states from it."""
    if case.name not in TINY_ENCODERS:
        pytest.skip("checked with transformers on the tiny models")
    case.model.save_pretrained(tmp_path)

    hf = AutoModel.from_pretrained(tmp_path, attn_implementation="eager").eval()
    with torch.no_grad():
        hidden = hf(torch.from_numpy(case.reference["input_ids"])[None]).last_hidden_state
    np.testing.assert_allclose(hidden[0].numpy(), case.reference["hidden"], rtol=1e-5, atol=1e-5)

    original = json.loads((Path(case.source) / "config.json").read_text())
    saved = json.loads((tmp_path / "config.json").read_text())
    differing = {k for k in original.keys() | saved.keys() if original.get(k) != saved.get(k)}
    assert differing <= {"torch_dtype", "dtype"}, f"config fields changed: {sorted(differing)}"


def test_sentence_transformers_loads_saved_encoder(case, tmp_path):
    """sentence-transformers loads what we save, and computes the reference embeddings."""
    if "st_embeddings" not in case.reference:
        pytest.skip("no sentence-transformers reference for this model")
    from sentence_transformers import SentenceTransformer

    case.model.save_pretrained(tmp_path)
    tokenizer = Path(snapshot_download(case.source, allow_patterns=TOKENIZER_FILES))
    for file in tokenizer.iterdir():
        if file.is_file():
            shutil.copy(file, tmp_path / file.name)

    st = SentenceTransformer(str(tmp_path), device="cpu")
    np.testing.assert_allclose(
        st.encode(SENTENCES), case.reference["st_embeddings"], rtol=1e-4, atol=1e-5
    )
