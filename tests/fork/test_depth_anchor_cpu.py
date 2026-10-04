"""
CPU tests of the depth anchor.

Run: python -m tests.fork.test_depth_anchor_cpu
  - the depth loss is scale and shift invariant and detects structure differences
  - x0 recovered from predict() outputs equals x_t - sigma * v
  - the real taew2_1 tiny decoder decodes single-frame latents, differentiably
    (set TAEW2_1 to its .safetensors path; skipped when not available)
  - both depth model families (randomly initialized) run and pass gradients to the input
  - integrated in GenericTrainer: trains only the LoRA, logs per-band losses and a preview, loss_split alternates
"""
import os
import tempfile

from modules.trainer.extension.DepthAnchor import DepthAnchor, depth_anchor_loss
from modules.trainer.extension.FlowModelAdapter import AnimaFlowAdapter, TinyDecoder
from modules.trainer.extension.perceptual.depth_models import DepthEstimator
from modules.util.TrainProgress import TrainProgress
from tests.fork.test_generic_trainer_cpu import make_trainer

import torch
from torch import nn

TAEW2_1 = os.environ.get("TAEW2_1", "/tmp/claude-0/taehv/safetensors/taew2_1.safetensors")


def test_loss():
    torch.manual_seed(0)
    depth = torch.rand(2, 32, 32) + torch.linspace(0, 1, 32)[None, None, :]
    assert depth_anchor_loss(depth * 3.0 + 7.0, depth, 0.5).max() < 1e-5, "not scale/shift invariant"
    other = torch.flip(depth, dims=[2])
    assert depth_anchor_loss(other, depth, 0.5).min() > 0.1
    assert depth_anchor_loss(other, depth, 0.0).shape == (2,)
    print("loss OK")


def test_x0_recovery():
    torch.manual_seed(0)
    model = type("M", (), {})()
    model.scale_latents = lambda x: x
    model.noise_scheduler = type("S", (), {"config": {"num_train_timesteps": 1000}})()
    adapter = AnimaFlowAdapter(model, None, None, torch.device("cpu"))
    x0 = torch.randn(3, 16, 1, 4, 4)
    noise = torch.randn_like(x0)
    timestep = torch.tensor([10, 500, 990])
    sigma = ((timestep + 1) / 1000).view(-1, 1, 1, 1, 1)
    x_t = (1 - sigma) * x0 + sigma * noise
    v_pred = torch.randn_like(x0)
    out = {"timestep": timestep, "target": noise - x0, "predicted": v_pred}
    predicted, clean, s = adapter.predicted_clean_latents({"latent_image": x0}, out)
    assert torch.allclose(predicted, x_t - sigma * v_pred, atol=1e-5)
    assert torch.allclose(clean, x0) and torch.allclose(s, sigma.flatten())
    print("x0 recovery OK")


def test_tiny_decoder():
    if not os.path.isfile(TAEW2_1):
        print("tiny decoder SKIPPED (no taew2_1 weights)")
        return
    decoder = TinyDecoder(TAEW2_1, "taew2_1")
    latents = torch.randn(2, 16, 1, 8, 12, requires_grad=True)
    images = decoder(latents)
    assert images.shape == (2, 3, 64, 96), images.shape
    assert images.min() >= 0 and images.max() <= 1
    images.mean().backward()
    assert latents.grad.abs().sum() > 0
    assert not any(p.requires_grad for p in decoder.parameters())
    print("tiny decoder OK")


def random_v2() -> nn.Module:
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
    cfg = DepthAnythingConfig(
        backbone_config={"model_type": "dinov2", "hidden_size": 384, "num_attention_heads": 6, "num_hidden_layers": 4,
                         "out_indices": [1, 2, 3, 4], "image_size": 518, "patch_size": 14,
                         "reshape_hidden_states": False},
        fusion_hidden_size=64, neck_hidden_sizes=[48, 96, 192, 384], reassemble_hidden_size=384, head_hidden_size=32,
    )
    return DepthAnythingForDepthEstimation(cfg)


def random_estimator(resolution=70) -> DepthEstimator:
    estimator = DepthEstimator(random_v2(), "v2", resolution)
    estimator.requires_grad_(False)
    return estimator.eval()


def test_depth_models():
    images = torch.rand(2, 3, 64, 96, requires_grad=True)
    depth = random_estimator(70)(images)
    assert depth.dim() == 3 and depth.shape[0] == 2, depth.shape
    depth.mean().backward()
    assert images.grad.abs().sum() > 0
    try:
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
    except ImportError:
        print("depth models OK (Depth Anything 3 SKIPPED, not installed)")
        return
    net = create_object(load_config(MODEL_REGISTRY["da3-small"]))
    images = torch.rand(1, 3, 64, 96, requires_grad=True)
    depth = DepthEstimator(net, "v3", 70).eval()(images)
    assert depth.dim() == 3, depth.shape
    depth.mean().backward()
    assert images.grad.abs().sum() > 0
    print("depth models OK (v2 + v3)")


def make_anchor_trainer(tmp, **overrides):
    trainer, model = make_trainer(tmp, cd_enabled=False, standalone=False)
    da = trainer.config.depth_anchor
    da.enabled = True
    da.depth_resolution = 70
    da.preview_every = 2
    for k, v in overrides.items():
        setattr(da, k, v)
    anchor = DepthAnchor(trainer.config)
    anchor.on_train_start = lambda t: _start(anchor, t)
    trainer.extensions = [anchor]
    return trainer, model, anchor


def _start(anchor, trainer):
    anchor.adapter = AnimaFlowAdapter(trainer.model, trainer.model_setup, trainer.config, trainer.train_device)
    anchor.decoder = TinyDecoder(TAEW2_1, "taew2_1")
    anchor.depth = random_estimator(70)


class RecordingWriter:
    def __init__(self):
        self.scalars, self.images = {}, {}

    def add_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, []).append(value)

    def add_image(self, tag, image, step):
        self.images.setdefault(tag, []).append(image)

    def close(self):
        pass


def test_integration():
    if not os.path.isfile(TAEW2_1):
        print("integration SKIPPED (no taew2_1 weights)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model, anchor = make_anchor_trainer(tmp)
        trainer.tensorboard = RecordingWriter()
        base = {k: v.clone() for k, v in model.transformer.state_dict().items() if "lora" not in k}
        before = [p.detach().clone() for p in model.transformer_lora.parameters()]
        trainer.train()
        assert any(not torch.equal(a, p) for a, p in zip(before, model.transformer_lora.parameters(), strict=True))
        for k, v in model.transformer.state_dict().items():
            if k in base:
                assert torch.equal(v, base[k]), f"base weight changed: {k}"
        tb = trainer.tensorboard
        assert "loss/depth_anchor" in tb.scalars, tb.scalars.keys()
        assert any(k.startswith("depth_anchor/loss_") for k in tb.scalars)
        preview = tb.images["depth_anchor/preview"][0]
        assert preview.shape == (3, 64, 256), preview.shape  # 4 tiles of 64x64
        print(f"integration OK, depth losses {[round(v, 4) for v in tb.scalars['loss/depth_anchor']]}")

    # the anchor's gradient alone reaches the LoRA
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model, anchor = make_anchor_trainer(tmp, weight=1.0)
        anchor.on_train_start(trainer)
        batch = next(iter(trainer.data_loader.get_data_loader()))
        progress = TrainProgress()
        output = trainer.model_setup.predict(model, batch, trainer.config, progress)
        zero = torch.zeros((), requires_grad=True)
        anchor_only = anchor.adjust_image_loss(trainer, batch, output, zero, progress)
        anchor_only.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer_lora.parameters())
        print("anchor gradient OK")


def test_schedule():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, a = make_anchor_trainer(tmp, every_n_steps=1)
        assert all(a.fires(TrainProgress(global_step=s)) for s in range(6))
        _, _, a = make_anchor_trainer(tmp, every_n_steps=3)
        assert [a.fires(TrainProgress(global_step=s)) for s in range(7)] == [True, False, False, True, False, False, True]
        _, _, a = make_anchor_trainer(tmp, every_n_steps=2, loss_split=True)
        assert [a.fires(TrainProgress(global_step=s)) for s in range(6)] == [False, True, False, True, False, True]
        try:
            make_anchor_trainer(tmp, every_n_steps=1, loss_split=True)
            raise AssertionError("expected an error")
        except ValueError:
            pass

        # loss_split: anchor steps carry only the anchor, diffusion steps are untouched
        trainer, model, a = make_anchor_trainer(tmp, every_n_steps=2, loss_split=True)
        a.on_train_start(trainer)
        batch = next(iter(trainer.data_loader.get_data_loader()))
        diffusion = torch.tensor(5.0)
        for step, expect_diffusion in ((0, True), (1, False)):
            progress = TrainProgress(global_step=step)
            output = trainer.model_setup.predict(model, batch, trainer.config, progress)
            result = a.adjust_image_loss(trainer, batch, output, diffusion, progress)
            assert (result is diffusion) == expect_diffusion, step
        # previews happen on anchored steps even when they never land on a multiple of preview_every
        trainer, model, a = make_anchor_trainer(tmp, every_n_steps=2, loss_split=True, preview_every=4)
        a.on_train_start(trainer)
        previews = []
        for step in range(12):
            progress = TrainProgress(global_step=step)
            output = trainer.model_setup.predict(model, batch, trainer.config, progress)
            a.adjust_image_loss(trainer, batch, output, diffusion, progress)
            if a._preview is not None:
                previews.append(step)
                a._preview = None
        assert previews == [1, 5, 9], previews
    print("schedule OK")


if __name__ == "__main__":
    test_loss()
    test_x0_recovery()
    test_tiny_decoder()
    test_depth_models()
    test_schedule()
    test_integration()
    print("ALL OK")
