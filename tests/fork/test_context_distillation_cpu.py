"""
CPU test of context distillation with a tiny randomly initialized Anima transformer.

Run: python -m tests.fork.test_context_distillation_cpu
Checks the mechanics only (no real model is involved):
  - buckets: square default, "ar" snaps to OneTrainer's own buckets, fixed and multi resolutions
  - teacher latents are generated with the LoRA switched off and cached; the cache is reused, invalidated
    when a setting changes, and cleared by "clear cache before training"
  - an epoch trains every pair at most once, in single-bucket batches
  - only LoRA parameters receive gradients, identical prompts give zero loss at init, the student learns
"""
import hashlib
import json
import os
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace

from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.AnimaLoRASetup import AnimaLoRASetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.trainer.extension.ContextDistillation import (
    ContextDistillation,
    PromptPair,
    bucket_resolutions,
    resolve_bucket,
)
from modules.trainer.extension.FlowModelAdapter import AnimaFlowAdapter
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.AspectBucketing import AspectBucketing

import torch

from diffusers import CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler

TEXT_DIM = 32
TEXT_LEN = 8
ENCODE_CALLS = []


def fake_encode(self, prompts):
    # deterministic pseudo text embedding per prompt
    ENCODE_CALLS.append(len(prompts))
    out = []
    for p in prompts:
        seed = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        out.append(torch.randn(1, TEXT_LEN, TEXT_DIM, generator=g))
    return torch.cat(out)


def write_pairs(tmp, pairs):
    path = os.path.join(tmp, "pairs.jsonl")
    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    return path


def tiny_transformer():
    return CosmosTransformer3DModel(
        in_channels=16, out_channels=16, num_attention_heads=2, attention_head_dim=32, num_layers=2,
        text_embed_dim=TEXT_DIM, adaln_lora_dim=8, max_size=(4, 32, 32), extra_pos_embed_type=None,
        encoder_hidden_states_channels=TEXT_DIM, crossattn_proj_in_channels=TEXT_DIM,
    )


def build(tmp, pairs, batch_size=1, model=None, **cd_overrides):
    torch.manual_seed(0)
    config = TrainConfig.default_values()
    config.model_type = ModelType.ANIMA
    config.training_method = TrainingMethod.LORA
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.train_dtype = DataType.FLOAT_32
    config.lora_rank = 4
    config.lora_alpha = 4.0
    config.layer_filter = "attn1,attn2,ff"
    config.resolution = "64"
    config.batch_size = batch_size
    config.cache_dir = os.path.join(tmp, "cache")
    config.clear_cache_before_training = False
    cd = config.context_distillation
    cd.enabled = True
    cd.prompts_path = write_pairs(tmp, pairs)
    cd.teacher_steps = 3
    cd.teacher_cfg = 2.0
    for k, v in cd_overrides.items():
        setattr(cd, k, v)

    if model is None:
        model = AnimaModel(ModelType.ANIMA)
        model.transformer = tiny_transformer()
        model.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        model.train_dtype = DataType.FLOAT_32
        model.autocast_context = nullcontext()
        model.transformer_lora = LoRAModuleWrapper(model.transformer, "transformer", config, config.layer_filter.split(","))
        model.transformer_lora.hook_to_module()
        model.transformer.requires_grad_(False)
        for p in model.transformer_lora.parameters():
            p.requires_grad_(True)
        model.materialize_only = lambda *a, **k: None

    setup = AnimaLoRASetup(torch.device("cpu"), torch.device("cpu"), False)
    setup.setup_train_device = lambda m, c: m.transformer.train()

    trainer = SimpleNamespace(
        model=model, model_setup=setup, train_device=torch.device("cpu"),
        callbacks=SimpleNamespace(on_update_status=lambda s: None),
    )
    AnimaFlowAdapter.encode_prompts = fake_encode
    ext = ContextDistillation(config)
    return config, model, trainer, ext


def cache_files(config):
    d = os.path.join(config.cache_dir, "context_distillation")
    return sorted(f for f in os.listdir(d) if f.endswith(".pt")) if os.path.isdir(d) else []


def test_buckets():
    # identical to mgds AspectBucketing (the function OneTrainer uses for images)
    ab = AspectBucketing(64, "a", "b", "c", "d", "e", False, "f", "g", "h")
    reference, _ = ab._AspectBucketing__create_automatic_buckets([512, 1024])
    assert set(bucket_resolutions(512, 64)) == set(reference[512])
    assert set(bucket_resolutions(1024, 64)) == set(reference[1024])

    square = PromptPair("s", "t", None)
    tall = PromptPair("s", "t", (3.0, 4.0))
    wide = PromptPair("s", "t", (16.0, 9.0))
    assert resolve_bucket(square, "512", 64, 0) == (512, 512)
    h, w = resolve_bucket(tall, "512", 64, 0)
    assert h > w and (h, w) in bucket_resolutions(512, 64), (h, w)
    h, w = resolve_bucket(wide, "1024", 64, 0)
    assert w > h and (h, w) in bucket_resolutions(1024, 64), (h, w)
    assert resolve_bucket(tall, "768x512", 64, 0) == (512, 768)  # fixed size wins over ar
    multi = {resolve_bucket(PromptPair(f"s{i}", "t", None), "256,1024", 64, 0) for i in range(20)}
    assert multi == {(256, 256), (1024, 1024)}, multi
    assert resolve_bucket(square, "256,1024", 64, 0) == resolve_bucket(square, "256,1024", 64, 0)
    print("buckets OK")


def test_cache_and_epochs():
    with tempfile.TemporaryDirectory() as tmp:
        pairs = [{"student": f"s{i}", "teacher": f"teacher prompt {i}"} for i in range(5)] + \
                [{"student": f"w{i}", "teacher": f"wide teacher {i}", "ar": "2:1"} for i in range(3)]
        config, model, trainer, ext = build(tmp, pairs, batch_size=2, resolution="64")
        ENCODE_CALLS.clear()
        ext.on_train_start(trainer)
        assert len(cache_files(config)) == 8
        assert len(set(ext.buckets)) == 2
        first_calls = len(ENCODE_CALLS)
        assert first_calls > 0

        # epoch: 5 square pairs -> 2 batches (1 dropped), 3 wide pairs -> 1 batch (1 dropped)
        batches = ext._epoch_batches(0)
        assert len(batches) == 3, batches
        seen = [i for b in batches for i in b]
        assert len(seen) == len(set(seen)), "a pair was trained twice in one epoch"
        assert all(len({ext.buckets[i] for i in b}) == 1 for b in batches), "mixed buckets in a batch"
        assert ext._epoch_batches(0) == batches and ext._epoch_batches(1) != batches

        # a second run reuses everything: no text encoding, no generation
        _, _, trainer2, ext2 = build(tmp, pairs, batch_size=2, model=model, resolution="64")
        ENCODE_CALLS.clear()
        ext2.on_train_start(trainer2)
        assert ENCODE_CALLS == [], ENCODE_CALLS
        assert ext2.entry_paths == ext.entry_paths

        # changing a setting that affects the latents regenerates them
        _, _, trainer3, ext3 = build(tmp, pairs, batch_size=2, model=model, resolution="64", teacher_cfg=3.0)
        ext3.on_train_start(trainer3)
        assert len(cache_files(config)) == 16

        # clear cache before training removes everything first
        config4, _, trainer4, ext4 = build(tmp, pairs, batch_size=2, model=model, resolution="64")
        config4.clear_cache_before_training = True
        config4.latent_caching = True
        ext4.on_train_start(trainer4)
        assert len(cache_files(config4)) == 8

        # nothing trainable when no bucket fills a batch
        _, _, trainer5, ext5 = build(tmp, pairs[:1], batch_size=2, model=model, resolution="64")
        try:
            ext5.on_train_start(trainer5)
            raise AssertionError("expected an error")
        except ValueError as e:
            assert "batch size" in str(e)
    print("cache and epochs OK")


def test_mechanics():
    with tempfile.TemporaryDirectory() as tmp:
        pairs = [{"student": "a cat", "teacher": "a fluffy orange cat on a red sofa, soft light"},
                 {"student": "a house", "teacher": "an old wooden house in a snowy forest at dusk"}]
        config, model, trainer, ext = build(tmp, pairs, batch_size=2)
        base_state = {k: v.clone() for k, v in model.transformer.state_dict().items()}

        # make the LoRA non-trivial, so "LoRA off" is observable
        for p in model.transformer_lora.parameters():
            with torch.no_grad():
                p.normal_(0, 0.5)

        ext.on_train_start(trainer)
        entry = torch.load(ext.entry_paths[0], weights_only=True)
        assert entry["latent"].shape == (1, 16, 1, 8, 8), entry["latent"].shape

        # the cached latent was generated by the base model (LoRA off)
        model.transformer_lora.remove_hook_from_module()
        g = torch.Generator().manual_seed((int(os.path.basename(ext.entry_paths[0])[:12], 16) + 42) % (2 ** 63))
        negative = fake_encode(None, [""])
        expected = ext.adapter.generate(entry["teacher"], negative, 2.0, 3, 64, 64, g)
        model.transformer_lora.hook_to_module()
        assert torch.allclose(expected, entry["latent"], atol=1e-5), "teacher latent was not made by the base model"

        # gradients reach only LoRA parameters
        loss = ext.compute_loss(trainer, None, TrainProgress())
        loss.backward()
        lora = {id(p) for p in model.transformer_lora.parameters()}
        assert all(p.grad is None for p in model.transformer.parameters() if id(p) not in lora)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer_lora.parameters())
        assert sum(ext._band_count.values()) == 2

        for k, v in model.transformer.state_dict().items():
            if k in base_state:
                assert torch.equal(v, base_state[k]), k
    print("mechanics OK")


def test_identity_and_learning():
    with tempfile.TemporaryDirectory() as tmp:
        # identical prompts: student == teacher at init (lora_up is zero-initialized)
        _, _, trainer, ext = build(tmp, [{"student": "same prompt", "teacher": "same prompt"}])
        ext.on_train_start(trainer)
        loss = ext.compute_loss(trainer, None, TrainProgress())
        assert loss.item() < 1e-8, loss.item()

    with tempfile.TemporaryDirectory() as tmp:
        _, model, trainer, ext = build(tmp, [{"student": "x", "teacher": "a very detailed description of x"}])
        ext.on_train_start(trainer)
        optimizer = torch.optim.Adam(list(model.transformer_lora.parameters()), lr=3e-3)
        progress = TrainProgress()
        losses = []
        for _ in range(150):
            optimizer.zero_grad()
            loss = ext.compute_loss(trainer, None, progress)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress.next_step(1)
        first, last = sum(losses[:15]) / 15, sum(losses[-15:]) / 15
        print(f"loss: first 15 steps {first:.4f} -> last 15 steps {last:.4f}")
        assert last < first * 0.6, (first, last)
    print("learning OK")


def test_weight_schedule():
    with tempfile.TemporaryDirectory() as tmp:
        pair = [{"student": "a", "teacher": "b"}]
        _, _, _, ext = build(tmp, pair, loss_weight=2.0, stop_after_steps=100, decay_to_zero=True)
        assert ext.current_weight(0) == 2.0
        assert abs(ext.current_weight(50) - 1.0) < 1e-9
        assert ext.current_weight(100) == 0.0
        _, _, _, ext = build(tmp, pair, loss_weight=2.0, stop_after_steps=100)
        assert ext.current_weight(99) == 2.0 and ext.current_weight(100) == 0.0
    print("schedule OK")


if __name__ == "__main__":
    test_buckets()
    test_weight_schedule()
    test_cache_and_epochs()
    test_mechanics()
    test_identity_and_learning()
    print("ALL OK")
