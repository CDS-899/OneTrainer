"""
CPU test of the context distillation extension with a tiny randomly initialized Anima transformer.

Run: python -m tests.fork.test_context_distillation_cpu
Checks the mechanics only (no real model is involved):
  - teacher latents are generated with the LoRA switched off
  - only LoRA parameters receive gradients
  - identical short/dense prompts give ~zero loss at init (LoRA starts as identity)
  - the student learns to imitate the dense prompt (loss goes down)
  - the base model is unchanged afterwards
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
from modules.trainer.extension.ContextDistillation import ContextDistillation, parse_resolution
from modules.trainer.extension.FlowModelAdapter import AnimaFlowAdapter
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch

from diffusers import CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler

TEXT_DIM = 32
TEXT_LEN = 8


def fake_encode(self, prompts):
    # deterministic pseudo text embedding per prompt
    out = []
    for p in prompts:
        seed = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        out.append(torch.randn(1, TEXT_LEN, TEXT_DIM, generator=g))
    return torch.cat(out)


def build(tmpdir, pairs, **cd_overrides):
    torch.manual_seed(0)
    path = os.path.join(tmpdir, "pairs.jsonl")
    with open(path, "w") as f:
        for short, dense in pairs:
            f.write(json.dumps({"short": short, "dense": dense}) + "\n")

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
    cd = config.context_distillation
    cd.enabled = True
    cd.prompts_path = path
    cd.pool_size = 3
    cd.pool_refresh_every = 2
    cd.teacher_steps = 3
    cd.teacher_cfg = 2.0
    cd.batch_size = 2
    for k, v in cd_overrides.items():
        setattr(cd, k, v)

    model = AnimaModel(ModelType.ANIMA)
    model.transformer = CosmosTransformer3DModel(
        in_channels=16, out_channels=16, num_attention_heads=2, attention_head_dim=32, num_layers=2,
        text_embed_dim=TEXT_DIM, adaln_lora_dim=8, max_size=(4, 32, 32), extra_pos_embed_type=None,
        encoder_hidden_states_channels=TEXT_DIM, crossattn_proj_in_channels=TEXT_DIM,
    )
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


def test_parse_resolution():
    assert parse_resolution("512", 64) == (512, 512)
    assert parse_resolution("768x512", 64) == (512, 768)
    assert parse_resolution("512, 768", 64) == (512, 512)
    assert parse_resolution("500", 64) == (448, 448)


def test_mechanics():
    with tempfile.TemporaryDirectory() as tmp:
        pairs = [("a cat", "a fluffy orange cat on a red sofa, soft light"),
                 ("a house", "an old wooden house in a snowy forest at dusk")]
        config, model, trainer, ext = build(tmp, pairs)
        base_state = {k: v.clone() for k, v in model.transformer.state_dict().items()}

        # make the LoRA non-trivial, so "LoRA off" is observable
        for p in model.transformer_lora.parameters():
            with torch.no_grad():
                p.normal_(0, 0.5)

        ext.on_train_start(trainer)
        assert len(ext.pool) == 3, len(ext.pool)
        assert ext.pool.latents[0].shape == (1, 16, 1, 8, 8), ext.pool.latents[0].shape

        # teacher (prior_model) prediction must equal a model without any LoRA hooks
        x = torch.randn(1, 16, 1, 8, 8)
        sigma = torch.tensor([0.5])
        cond = ext.dense_conditioning[0]
        with torch.no_grad(), setup_prior(trainer):
            v_teacher = ext.adapter.velocity(x, sigma, cond)
        model.transformer_lora.remove_hook_from_module()
        with torch.no_grad():
            v_base = ext.adapter.velocity(x, sigma, cond)
        model.transformer_lora.hook_to_module()
        with torch.no_grad():
            v_student = ext.adapter.velocity(x, sigma, cond)
        assert torch.allclose(v_teacher, v_base, atol=1e-5), "teacher is not the base model"
        assert not torch.allclose(v_student, v_base, atol=1e-3), "LoRA has no effect, test is meaningless"

        # gradients reach only LoRA parameters
        progress = TrainProgress()
        loss = ext.compute_loss(trainer, None, progress)
        loss.backward()
        assert all(p.grad is None for p in model.transformer.parameters() if not is_lora(model, p))
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer_lora.parameters())

        # base weights untouched
        for k, v in model.transformer.state_dict().items():
            if k in base_state:
                assert torch.equal(v, base_state[k]), k
    print("mechanics OK")


def test_identity_and_learning():
    with tempfile.TemporaryDirectory() as tmp:
        # identical prompts: student == teacher at init (lora_up is zero-initialized)
        config, model, trainer, ext = build(tmp, [("same prompt", "same prompt")])
        ext.on_train_start(trainer)
        loss = ext.compute_loss(trainer, None, TrainProgress())
        assert loss.item() < 1e-8, loss.item()

        # different prompts: the student should learn to imitate the dense prompt
        config, model, trainer, ext = build(
            tmp, [("x", "a very detailed description of x")], pool_refresh_every=0,
        )
        ext.on_train_start(trainer)
        params = list(model.transformer_lora.parameters())
        optimizer = torch.optim.Adam(params, lr=3e-3)
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
        _, _, _, ext = build(tmp, [("a", "b")], loss_weight=2.0, stop_after_steps=100, decay_to_zero=True)
        assert ext.current_weight(0) == 2.0
        assert abs(ext.current_weight(50) - 1.0) < 1e-9
        assert ext.current_weight(100) == 0.0
        _, _, _, ext = build(tmp, [("a", "b")], loss_weight=2.0, stop_after_steps=100)
        assert ext.current_weight(99) == 2.0 and ext.current_weight(100) == 0.0
    print("schedule OK")


def setup_prior(trainer):
    cfg = SimpleNamespace(training_method=TrainingMethod.LORA)
    return trainer.model_setup.prior_model(trainer.model, cfg)


def is_lora(model, p):
    return any(p is q for q in model.transformer_lora.parameters())


if __name__ == "__main__":
    test_parse_resolution()
    test_weight_schedule()
    test_mechanics()
    test_identity_and_learning()
    print("ALL OK")
