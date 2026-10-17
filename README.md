# OneTrainer fork: experimental training features

A personal fork of [OneTrainer](https://github.com/Nerogar/OneTrainer) with experimental training methods.
Everything else (models, UI, layer offloading, int8, compile, fused back pass) is upstream OneTrainer and works as
usual; see the [upstream README](https://github.com/Nerogar/OneTrainer#readme) and wiki for general usage.

## What this fork adds

- **Context distillation**: teaches a LoRA that a short prompt should behave like a long one, from text pairs only
  (no images). Uses: prompt adherence, concept warm-start before image training.
- **Depth anchor** (experimental): adds a depth-structure loss to image training. In tests on Anima it did not help
  and cost likeness, so it is off by default.
- **SOAR** (experimental): the model also learns to recover from its own sampling steps (Tencent HY-SOAR), as an
  addition to normal image training.
- **Sample prompt cache**: sampling during training reuses text embeddings of prompts it has already encoded.
- **Trainer extensions**: a small hook system in `GenericTrainer` that the features above plug into. Plain LoRA and
  fine-tune training is unchanged.

Supported model so far: Anima (adding one is a small adapter class).

Settings, file formats and what to watch in TensorBoard: [FORK.md](FORK.md).

## Install

```bat
git clone -b dev https://github.com/CDS-899/OneTrainer OneTrainer_Fork
cd OneTrainer_Fork
install.bat
```

Linux: `./install.sh`. Update with `git pull`, then `update.bat` / `./update.sh`.

## License

AGPL-3.0, like upstream OneTrainer. See [LICENSE.txt](LICENSE.txt).
