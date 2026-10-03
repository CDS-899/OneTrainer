"""
Run: python -m tests.fork.test_prompt_files_and_cache_cpu
  - prompt file entry types (pair / preserve / trigger with contexts, descriptions, generic)
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
        with open(os.path.join(tmp, "contexts.txt"), "w") as f:
            f.write("# comment\nportrait of {subject}\n\n{subject} in a garden\n")
        path = os.path.join(tmp, "pairs.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"short": "s", "dense": {"k": "v"}}) + "\n")
            f.write(json.dumps({"preserve": ["a cat", "a dog"]}) + "\n")
            f.write(json.dumps({"trigger": "T", "description": ["d1", "d2"], "contexts": ["{subject} at night"],
                                "contexts_file": "contexts.txt", "generic": "a man"}) + "\n")
        pairs = load_prompt_pairs(path)

    assert pairs[0] == ("s", '{"k": "v"}')
    assert pairs[1:3] == [("a cat", "a cat"), ("a dog", "a dog")]
    trigger_pairs = pairs[3:]
    # 3 contexts x (2 descriptions + 1 generic)
    assert len(trigger_pairs) == 9, trigger_pairs
    assert ("T at night", "d1 at night") in trigger_pairs
    assert ("portrait of T", "portrait of d2") in trigger_pairs
    assert ("a man in a garden", "a man in a garden") in trigger_pairs
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
