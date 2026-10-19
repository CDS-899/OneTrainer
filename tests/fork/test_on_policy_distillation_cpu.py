"""
CPU tests of SMD distillation (smd_distillation) with a tiny random Anima transformer.

Run: python -m tests.fork.test_on_policy_distillation_cpu
  - prompt file parsing
  - a batch of prompts runs through the student's own Euler trajectory (LoRA on, CFG 1), one step per training step
  - the teacher is the model with the LoRA off, with CFG and (for SMD) a negative-prompt reference
  - only LoRA parameters receive gradients
  - learning: after training, the few-step student lands closer to the many-step CFG teacher from the same noise
    (also as CFG-only distillation: more student steps, no repulsion)
  - epochs, and standalone training in GenericTrainer
"""
import json
import os
import tempfile
from contextlib import contextmanager

from modules.trainer.extension.FlowModelAdapter import AnimaFlowAdapter
from modules.trainer.extension.OnPolicyDistillation import SmdDistillation, load_prompts
from modules.util.TrainProgress import TrainProgress
from tests.fork.test_context_distillation_cpu import build as build_cd
from tests.fork.test_context_distillation_cpu import fake_encode
from tests.fork.test_depth_anchor_cpu import RecordingWriter
from tests.fork.test_generic_trainer_cpu import make_trainer

import torch


def write_prompts(tmp, prompts):
    path = os.path.join(tmp, "prompts.jsonl")
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    return path


def build(tmp, prompts, batch_size=1, **overrides):
    # reuse the context distillation test setup (tiny model with a LoRA, fake trainer), then switch the block
    config, model, trainer, _ = build_cd(tmp, [{"student": "x", "teacher": "y"}], batch_size=batch_size)
    config.context_distillation.enabled = False
    block = config.smd_distillation
    block.enabled = True
    block.prompts_path = write_prompts(tmp, prompts)
    for k, v in overrides.items():
        setattr(block, k, v)
    ext = SmdDistillation(config)
    return config, model, trainer, ext


def test_prompt_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_prompts(tmp, [{"prompt": "a cat"}, {"prompt": {"subject": "dog"}, "ar": "3:4"}])
        prompts = load_prompts(path, "test")
        assert prompts[0].prompt == "a cat" and prompts[0].ar is None
        assert prompts[1].prompt == '{"subject": "dog"}' and prompts[1].ar == (3.0, 4.0)
        bad = write_prompts(tmp, [{"text": "x"}])
        try:
            load_prompts(bad, "test")
            raise AssertionError("expected an error")
        except ValueError:
            pass
    print("prompt file OK")


@contextmanager
def counting_prior(trainer, calls):
    original = trainer.model_setup.prior_model

    @contextmanager
    def prior_model(model, config):
        calls.append(1)
        with original(model, config):
            yield

    trainer.model_setup.prior_model = prior_model
    try:
        yield
    finally:
        trainer.model_setup.prior_model = original


def test_trajectory_and_teacher():
    with tempfile.TemporaryDirectory() as tmp:
        config, model, trainer, ext = build(tmp, [{"prompt": "a cat"}, {"prompt": "a house"}], batch_size=2,
                                            student_steps=3, queries=2)
        for p in model.transformer_lora.parameters():
            with torch.no_grad():
                p.normal_(0, 0.3)
        ext.on_train_start(trainer)
        ext.on_epoch_start(trainer, TrainProgress())
        assert ext.standalone_epoch_length() == 3  # 1 batch x 3 student steps

        # the trajectory is the student's own Euler sampling at CFG 1 (LoRA on), from seeded noise
        progress = TrainProgress()
        calls = []
        with counting_prior(trainer, calls):
            loss = ext.compute_loss(trainer, None, progress)
        trajectory = ext._trajectory
        conditioning = trajectory['conditioning']
        g = torch.Generator().manual_seed(42 * 1_000_003 + 0)
        x = torch.randn(ext.adapter.latent_shape(2, 64, 64), generator=g)
        sigmas = ext.schedule
        with torch.no_grad():
            v = ext.adapter.velocity(x, sigmas[0].expand(2), conditioning).float()
        assert torch.allclose(trajectory['latents'], x + (sigmas[1] - sigmas[0]) * v, atol=1e-5)
        assert trajectory['step'] == 1
        # one prior-model (LoRA off) context per query, each with a CFG pass pair inside
        assert len(calls) == 2, calls

        # only LoRA parameters get gradients
        loss.backward()
        lora = {id(p) for p in model.transformer_lora.parameters()}
        assert all(p.grad is None for p in model.transformer.parameters() if id(p) not in lora)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer_lora.parameters())

        # the trajectory ends after student_steps training steps, the next one starts from new noise
        ext.compute_loss(trainer, None, progress)
        ext.compute_loss(trainer, None, progress)
        assert ext._trajectory is None

        # the teacher target is the base model (LoRA off) with CFG against the negative prompt
        target, negative = ext._teacher(trainer, x, sigmas[0].expand(2), conditioning)
        model.transformer_lora.remove_hook_from_module()
        with torch.no_grad():
            cond = ext.adapter.velocity(x, sigmas[0].expand(2), conditioning).float()
            neg = ext.adapter.velocity(x, sigmas[0].expand(2), fake_encode(None, [""]).expand_as(conditioning)).float()
        model.transformer_lora.hook_to_module()
        assert torch.allclose(negative, neg, atol=1e-5)
        assert torch.allclose(target, neg + 5.0 * (cond - neg), atol=1e-4)
    print("trajectory and teacher OK")


def student_vs_teacher(ext, trainer, conditioning, noise, teacher_steps, cfg):
    adapter = ext.adapter
    with torch.no_grad():
        x = noise.clone()
        for i in range(ext.steps):
            v = adapter.velocity(x, ext.schedule[i].expand(x.shape[0]), conditioning).float()
            x = x + (ext.schedule[i + 1] - ext.schedule[i]) * v
        with trainer.model_setup.prior_model(trainer.model, ext.config):
            sigmas = adapter.sampling_sigmas(teacher_steps)
            y = noise.clone()
            negative = ext.negative.expand_as(conditioning)
            for i in range(teacher_steps):
                s = sigmas[i].expand(y.shape[0])
                c = adapter.velocity(y, s, conditioning).float()
                n = adapter.velocity(y, s, negative).float()
                y = y + (sigmas[i + 1] - sigmas[i]) * (n + cfg * (c - n))
    return float((x - y).pow(2).mean().sqrt())


def test_learning():
    for kind in ("few-step", "cfg-only"):
        with tempfile.TemporaryDirectory() as tmp:
            overrides = {"student_steps": 2, "teacher_cfg": 3.0} if kind == "few-step" else \
                {"student_steps": 6, "teacher_cfg": 3.0, "repulsion_weight": 0.0}
            config, model, trainer, ext = build(tmp, [{"prompt": "a red house"}], **overrides)
            ext.on_train_start(trainer)
            ext.on_epoch_start(trainer, TrainProgress())
            conditioning = torch.load(ext.entry_paths[0], weights_only=True)
            noise = torch.randn(ext.adapter.latent_shape(1, 64, 64), generator=torch.Generator().manual_seed(7))
            teacher_steps = 12
            before = student_vs_teacher(ext, trainer, conditioning, noise, teacher_steps, 3.0)

            optimizer = torch.optim.Adam(list(model.transformer_lora.parameters()), lr=2e-3)
            progress = TrainProgress()
            for _ in range(240):
                optimizer.zero_grad()
                ext.compute_loss(trainer, None, progress).backward()
                optimizer.step()
                progress.next_step(1)
            after = student_vs_teacher(ext, trainer, conditioning, noise, teacher_steps, 3.0)
            print(f"{kind}: distance of the {ext.steps}-step CFG-1 student to the {teacher_steps}-step CFG-3 teacher: "
                  f"{before:.4f} -> {after:.4f}")
            assert after < before * 0.7, (before, after)
    print("learning OK")


def test_standalone_training():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model = make_trainer(tmp, cd_enabled=False, standalone=False)
        config = trainer.config
        config.smd_distillation.enabled = True
        config.smd_distillation.prompts_path = write_prompts(tmp, [{"prompt": f"prompt {i}"} for i in range(3)])
        config.smd_distillation.student_steps = 2
        ext = SmdDistillation(config)
        trainer.extensions = [ext]
        trainer.data_loader = None
        trainer.tensorboard = RecordingWriter()
        AnimaFlowAdapter.encode_prompts = fake_encode
        before = [p.detach().clone() for p in model.transformer_lora.parameters()]
        base = {k: v.clone() for k, v in model.transformer.state_dict().items() if "lora" not in k}
        trainer.train()
        # 2 epochs x 3 prompts (batch size 1) x 2 student steps
        assert model.train_progress.global_step == 12, model.train_progress.global_step
        assert any(not torch.equal(a, p) for a, p in zip(before, model.transformer_lora.parameters(), strict=True))
        for k, v in model.transformer.state_dict().items():
            if k in base:
                assert torch.equal(v, base[k]), k
        tb = trainer.tensorboard.scalars
        assert "loss/smd_distillation" in tb and "smd_distillation/repulsion_distance" in tb, tb.keys()
    print("standalone training OK")


if __name__ == "__main__":
    test_prompt_file()
    test_trajectory_and_teacher()
    test_learning()
    test_standalone_training()
    print("ALL OK")
