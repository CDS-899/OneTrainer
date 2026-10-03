"""
Run: python -m tests.fork.test_prompt_files_and_cache_cpu
  - prompt pairs file
  - sample prompt embedding cache (hit skips encode(), disabled while the text encoder trains)
"""
import json
import os
import tempfile
from types import SimpleNamespace

from modules.modelSampler.SamplePromptCache import cached_prompt_encoding
from modules.trainer.extension.ContextDistillation import load_prompt_pairs

import torch


def test_prompt_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"short": "s", "dense": {"k": "v"}}) + "\n")
            f.write("\n")
            f.write(json.dumps({"short": "a cat", "dense": "a cat"}) + "\n")
            f.write(json.dumps({"prompt": "p", "dense_prompt": "d"}) + "\n")
            f.write(json.dumps({"short": "incomplete"}) + "\n")
        pairs = load_prompt_pairs(path)
    assert pairs == [("s", '{"k": "v"}'), ("a cat", "a cat"), ("p", "d")], pairs
    print("prompt file OK")


def test_sample_cache():
    calls = []

    def encode():
        calls.append(1)
        return torch.ones(2, 3), torch.zeros(2, dtype=torch.bool)

    frozen = SimpleNamespace(train_config=SimpleNamespace(train_text_encoder_or_embedding=lambda: False))
    a = cached_prompt_encoding(frozen, ("p", "", 2), torch.device("cpu"), encode)
    b = cached_prompt_encoding(frozen, ("p", "", 2), torch.device("cpu"), encode)
    assert len(calls) == 1 and torch.equal(a[0], b[0]) and isinstance(b, tuple)
    cached_prompt_encoding(frozen, ("other", "", 2), torch.device("cpu"), encode)
    assert len(calls) == 2

    training_te = SimpleNamespace(train_config=SimpleNamespace(train_text_encoder_or_embedding=lambda: True))
    cached_prompt_encoding(training_te, ("p", "", 2), torch.device("cpu"), encode)
    cached_prompt_encoding(training_te, ("p", "", 2), torch.device("cpu"), encode)
    assert len(calls) == 4

    standalone_sampling = SimpleNamespace()
    cached_prompt_encoding(standalone_sampling, ("p", "", 2), torch.device("cpu"), encode)
    assert len(calls) == 5
    print("sample cache OK")


if __name__ == "__main__":
    test_prompt_file()
    test_sample_cache()
    print("ALL OK")
