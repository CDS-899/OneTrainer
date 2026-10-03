"""
CPU test of the refactored GenericTrainer loop with a tiny randomly initialized Anima transformer.

Run: python -m tests.fork.test_generic_trainer_cpu
  - plain LoRA training on image batches still works (no extension)
  - image batches + context distillation in the same step
  - standalone context distillation (no image data set)
"""
import json
import os
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace

from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.AnimaLoRASetup import AnimaLoRASetup
from modules.trainer.extension.ContextDistillation import ContextDistillation
from modules.trainer.extension.FlowModelAdapter import AnimaFlowAdapter
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from tests.fork.test_context_distillation_cpu import TEXT_DIM, TEXT_LEN, fake_encode

import torch

from diffusers import CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler


class FakeDataSet:
    def __init__(self, n):
        self.n = n

    def start_next_epoch(self):
        pass

    def approximate_length(self):
        return self.n


class FakeDataLoader:
    def __init__(self, n):
        self.data_set = FakeDataSet(n)

    def get_data_set(self):
        return self.data_set

    def get_data_loader(self):
        g = torch.Generator().manual_seed(1)
        for _ in range(self.data_set.n):
            yield {
                'latent_image': torch.randn(1, 16, 1, 8, 8, generator=g),
                'text_encoder_hidden_state': torch.randn(1, TEXT_LEN, TEXT_DIM, generator=g),
                'concept_type': [ConceptType.STANDARD.value],
                'loss_weight': torch.ones(1),
            }


def make_trainer(tmp, cd_enabled, standalone, image_steps=4, fused_back_pass=False):
    torch.manual_seed(0)
    config = TrainConfig.default_values()
    config.model_type = ModelType.ANIMA
    config.training_method = TrainingMethod.LORA
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.train_dtype = DataType.FLOAT_32
    config.lora_weight_dtype = DataType.FLOAT_32
    config.lora_rank = 4
    config.lora_alpha = 4.0
    config.layer_filter = "attn1,attn2,ff"
    config.resolution = "64"
    config.workspace_dir = os.path.join(tmp, "ws")
    config.cache_dir = os.path.join(tmp, "cache")
    config.tensorboard = False
    config.epochs = 2
    config.batch_size = 1
    config.learning_rate = 1e-3
    config.text_encoder.train = False
    config.sample_after_unit = TimeUnit.NEVER
    config.backup_after_unit = TimeUnit.NEVER
    config.save_every_unit = TimeUnit.NEVER
    if fused_back_pass:
        from modules.util.enum.Optimizer import Optimizer
        config.optimizer.optimizer = Optimizer.ADAMW
        config.optimizer.fused_back_pass = True

    pairs = os.path.join(tmp, "pairs.jsonl")
    with open(pairs, "w") as f:
        f.write(json.dumps({"short": "x", "dense": "a detailed description of x"}) + "\n")
    cd = config.context_distillation
    cd.enabled = cd_enabled
    cd.prompts_path = pairs
    cd.standalone = standalone
    cd.standalone_epoch_length = 5
    cd.pool_size = 2
    cd.pool_refresh_every = 3
    cd.teacher_steps = 2

    model = AnimaModel(ModelType.ANIMA)
    model.train_config = config
    model.transformer = CosmosTransformer3DModel(
        in_channels=16, out_channels=16, num_attention_heads=2, attention_head_dim=32, num_layers=2,
        text_embed_dim=TEXT_DIM, adaln_lora_dim=8, max_size=(4, 32, 32), extra_pos_embed_type=None,
        encoder_hidden_states_channels=TEXT_DIM, crossattn_proj_in_channels=TEXT_DIM,
    )
    model.text_encoder = torch.nn.Linear(1, 1)
    model.text_conditioner = torch.nn.Linear(1, 1)
    model.vae = torch.nn.Linear(1, 1)
    model.vae.config = SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16, z_dim=16)
    model.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    model.train_dtype = DataType.FLOAT_32
    model.autocast_context = nullcontext()
    model.materialize_only = lambda *a, **k: None
    model.materialize = lambda *a, **k: None
    model.evict = lambda *a, **k: None
    AnimaFlowAdapter.encode_prompts = fake_encode

    setup = AnimaLoRASetup(torch.device("cpu"), torch.device("cpu"), False)
    setup.setup_model(model, config)

    trainer = GenericTrainer(config, TrainCallbacks(), TrainCommands())
    trainer.model = model
    trainer.model_setup = setup
    trainer.extensions = [ContextDistillation(config)] if cd_enabled else []
    trainer.data_loader = None if (cd_enabled and standalone) else FakeDataLoader(image_steps)
    trainer.model_sampler = None
    trainer.sample_queue = []
    trainer.previous_sample_time = -1
    trainer.parameters = model.parameters.parameters()
    return trainer, model


def lora_snapshot(model):
    return [p.detach().clone() for p in model.transformer_lora.parameters()]


def run(cd_enabled, standalone, expected_steps, fused_back_pass=False):
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model = make_trainer(tmp, cd_enabled, standalone, fused_back_pass=fused_back_pass)
        before = lora_snapshot(model)
        base = {k: v.clone() for k, v in model.transformer.state_dict().items() if "lora" not in k}
        trainer.train()
        after = lora_snapshot(model)
        assert model.train_progress.global_step == expected_steps, model.train_progress.global_step
        assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True)), "LoRA did not change"
        for k, v in model.transformer.state_dict().items():
            if k in base:
                assert torch.equal(v, base[k]), f"base weight changed: {k}"
        if cd_enabled:
            assert trainer.extensions[0]._generated >= 2
        trainer.tensorboard.close()
    print(f"cd={cd_enabled} standalone={standalone} fused_back_pass={fused_back_pass}: {expected_steps} steps OK")


if __name__ == "__main__":
    run(cd_enabled=False, standalone=False, expected_steps=8)   # 2 epochs x 4 image batches
    run(cd_enabled=True, standalone=False, expected_steps=8)    # same, plus distillation each step
    run(cd_enabled=True, standalone=True, expected_steps=10)    # 2 epochs x standalone_epoch_length 5
    run(cd_enabled=True, standalone=False, expected_steps=8, fused_back_pass=True)
    print("ALL OK")
