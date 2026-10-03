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
from modules.trainer.extension.ContextDistillation import PromptPair, load_prompt_pairs

import torch


def test_prompt_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"student": "s", "teacher": {"k": "v"}}) + "\n")
            f.write("\n")
            f.write(json.dumps({"student": "a cat", "teacher": "a cat", "ar": "3:4"}) + "\n")
        pairs = load_prompt_pairs(path)
        assert pairs == [PromptPair("s", '{"k": "v"}', None), PromptPair("a cat", "a cat", (3.0, 4.0))], pairs

        for bad in ({"short": "s", "dense": "d"}, {"student": "s"}, {"student": "s", "teacher": "t", "ar": "wide"}):
            with open(path, "w") as f:
                f.write(json.dumps(bad) + "\n")
            try:
                load_prompt_pairs(path)
                raise AssertionError(f"accepted {bad}")
            except ValueError:
                pass
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
