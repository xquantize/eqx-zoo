"""Fixtures shared by the LoRA tests."""

import pytest
from support.tiny import build


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    """A function giving the directory of a tiny checkpoint, built once per model."""
    cache = {}

    def get(name):
        if name not in cache:
            directory = tmp_path_factory.mktemp(name)
            build(name, directory)
            cache[name] = directory
        return cache[name]

    return get
