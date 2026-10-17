"""
CPU tests of SOAR with a tiny randomly initialized Anima transformer.

Run: python -m tests.fork.test_soar_cpu
  - a correction point from a rollout that lands on the training trajectory has the normal target
  - one sampling step later on a schedule, between and on schedule points
  - the extra passes see the model exactly as the training step does (same conditioning, same timestep)
  - SOAR steps: normal loss scaled by 1 / (1 + weight * N), N separately back-propagated correction losses
  - integrated in GenericTrainer: trains only the LoRA, logs, every_n_steps, fused back pass, empty prompt cache
"""
import os
import tempfile

from modules.trainer.extension.FlowModelAdapter import AnimaFlowAdapter
from modules.trainer.extension.Soar import Soar, correction_point
from modules.util.TrainProgress import TrainProgress
from tests.fork.test_depth_anchor_cpu import RecordingWriter
from tests.fork.test_generic_trainer_cpu import make_trainer

import torch


def test_correction_point():
    torch.manual_seed(0)
    clean, noise = torch.randn(3, 16, 1, 4, 4), torch.randn(3, 16, 1, 4, 4)
    rollout_sigma = torch.tensor([0.3, 0.5, 0.0])
    sigma = torch.tensor([0.6, 0.5, 0.9])
    b = rollout_sigma.view(-1, 1, 1, 1, 1)
    on_trajectory = (1 - b) * clean + b * noise
    latent, target = correction_point(clean, noise, on_trajectory, rollout_sigma, sigma)
    s = sigma.view(-1, 1, 1, 1, 1)
    assert torch.allclose(latent, (1 - s) * clean + s * noise, atol=1e-5)
    assert torch.allclose(target, noise - clean, atol=1e-4)

    # an off-trajectory rollout: the target leads from the point straight to the clean latent
    off = on_trajectory + 0.1 * torch.randn_like(clean)
    latent, target = correction_point(clean, noise, off, rollout_sigma, sigma)
    assert torch.allclose(latent - s * target, clean, atol=1e-5)
    print("correction point OK")


def test_next_sigma():
    adapter = object.__new__(AnimaFlowAdapter)
    schedule = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])
    for sigma, expected in [(1.0, 0.9), (0.7, 0.4), (0.4, 0.0), (0.8, 0.55), (0.2, 0.0), (0.0, 0.0)]:
        got = float(adapter.next_sampling_sigma(torch.tensor([sigma]), schedule))
        assert abs(got - expected) < 1e-6, (sigma, got, expected)
    print("next sampling sigma OK")


def soar_trainer(tmp, fused_back_pass=False, **overrides):
    trainer, model = make_trainer(tmp, cd_enabled=False, standalone=False, fused_back_pass=fused_back_pass)
    soar = trainer.config.soar
    soar.enabled = True
    soar.rollout_steps = 4
    for k, v in overrides.items():
        setattr(soar, k, v)
    ext = Soar(trainer.config)
    trainer.extensions = [ext]
    return trainer, model, ext


def test_matches_training_step():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model, ext = soar_trainer(tmp)
        ext.on_train_start(trainer)
        batch = next(iter(trainer.data_loader.get_data_loader()))
        progress = TrainProgress(global_step=3)
        with torch.no_grad():
            out = trainer.model_setup.predict(model, batch, trainer.config, progress)
            adapter = ext.adapter
            clean = adapter.clean_latents(batch).float()
            noise = out['target'].float() + clean
            sigma = adapter.sigma_from_timestep(out['timestep']).view(-1, 1, 1, 1, 1)
            noisy = (1 - sigma) * clean + sigma * noise
            again = adapter.velocity_at_timestep(noisy, out['timestep'], adapter.batch_conditioning(batch, progress))
        assert torch.allclose(again, out['predicted'], atol=1e-5), (again - out['predicted']).abs().max()
    print("extra passes match the training step OK")


def test_step_losses():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model, ext = soar_trainer(tmp, correction_points=3, weight=0.5, every_n_steps=2)
        ext.on_train_start(trainer)
        batch = next(iter(trainer.data_loader.get_data_loader()))
        image_loss = torch.tensor(2.0)
        for step, fires in ((0, True), (1, False), (2, True)):
            progress = TrainProgress(global_step=step)
            out = trainer.model_setup.predict(model, batch, trainer.config, progress)
            adjusted = ext.adjust_image_loss(trainer, batch, out, image_loss, progress)
            losses = ext.compute_loss(trainer, batch, progress)
            if not fires:
                assert adjusted is image_loss and losses is None
                continue
            assert torch.isclose(adjusted, image_loss / 2.5), adjusted  # 1 + 0.5 * 3
            count = 0
            for loss in losses:
                assert loss.requires_grad and loss.dim() == 0
                model.transformer.zero_grad(set_to_none=True)
                loss.backward()
                assert any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.transformer_lora.parameters())
                count += 1
            assert count == 3
    print("SOAR step losses OK")


def test_integration(fused_back_pass=False):
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model, ext = soar_trainer(tmp, fused_back_pass=fused_back_pass, correction_points=2)
        trainer.tensorboard = RecordingWriter()
        encoded = []
        original = AnimaFlowAdapter.encode_prompts

        def counting_encode(self, prompts):
            encoded.append(list(prompts))
            return original(self, prompts)

        AnimaFlowAdapter.encode_prompts = counting_encode
        try:
            before = [p.detach().clone() for p in model.transformer_lora.parameters()]
            base = {k: v.clone() for k, v in model.transformer.state_dict().items() if "lora" not in k}
            trainer.train()
            assert any(not torch.equal(a, p) for a, p in zip(before, model.transformer_lora.parameters(), strict=True))
            for k, v in model.transformer.state_dict().items():
                if k in base:
                    assert torch.equal(v, base[k]), f"base weight changed: {k}"
            tb = trainer.tensorboard.scalars
            assert "loss/soar" in tb and "soar/rollout_error" in tb, tb.keys()
            assert all(v == v and v >= 0 for v in tb["loss/soar"])
            assert encoded == [[""]], encoded
            assert os.path.isfile(ext._empty_prompt_path())

            # a second run reuses the cached empty prompt (unless "clear cache before training" is on)
            trainer.config.clear_cache_before_training = False
            ext2 = Soar(trainer.config)
            ext2.on_train_start(trainer)
            assert encoded == [[""]] and torch.equal(ext2.empty_conditioning, ext.empty_conditioning)
        finally:
            AnimaFlowAdapter.encode_prompts = original
    print(f"integration (fused_back_pass={fused_back_pass}) OK, soar losses "
          f"{[round(v, 4) for v in tb['loss/soar']]}, rollout error {[round(v, 4) for v in tb['soar/rollout_error']]}")


if __name__ == "__main__":
    test_correction_point()
    test_next_sigma()
    test_matches_training_step()
    test_step_losses()
    test_integration()
    test_integration(fused_back_pass=True)
    print("ALL OK")
