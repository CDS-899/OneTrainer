# OneTrainer fork: distillation experiments

Experimental training features on top of OneTrainer, built so they keep every OneTrainer optimization
(layer offloading, int8 / compressed weights, compile, fused back pass) and leave normal LoRA and
fine-tune training unchanged.

## Branches

| Branch | What it is |
|---|---|
| `master` | Plain upstream `Nerogar/OneTrainer`, never modified here |
| `opt-stack` | Upstream PR #1694 (dxqb's optimization stack + LTX 2.5), unchanged |
| `dev` | `opt-stack` + this fork's features. **Use this one.** |

## Install

Windows:

```bat
cd D:\AI
git clone -b dev https://github.com/CDS-899/OneTrainer OneTrainer_Fork
cd OneTrainer_Fork
install.bat
```

Linux: the same with `./install.sh`. Update later with `git pull` followed by `update.bat` / `./update.sh`.

## Feature: context distillation (image-free)

Teaches a LoRA that one prompt (**student**) should behave like another (**teacher**), using only text pairs.
No images needed.

- **Student**: base model + LoRA, sees the `student` prompt (what you will type).
- **Teacher**: the same base model with the LoRA switched off, sees the `teacher` prompt (what it should copy).
- The student learns to predict the teacher's velocity at the same noisy latent.

The noisy latents come from one **teacher latent per pair**, generated once from the teacher prompt (never decoded
to an image) and cached like text embeddings. Every time a pair is trained, its latent is re-noised with fresh
noise at a timestep from the normal *Timestep Distribution* setting, so all noise levels get trained. The cached
latent only decides *where* the student is trained; the target is always the teacher's live prediction, at CFG 1,
so the LoRA learns behavior and is sampled with your usual steps and CFG.

Uses:
- **Prompt adherence**: make short prompts produce what the model makes from detailed descriptions.
- **Concept warm-start**: `"Taras Shevchenko"` → an appearance description. Run it for a few hundred steps before
  (or mixed into the start of) normal image training, so image training starts close to the target.

### Prompt file

`.jsonl`, one pair per line:

```json
{"student": "a magical girl casting a spell", "teacher": "A beautiful anime magical girl with flowing pastel pink twintails ...", "ar": "3:4"}
```

- `ar` (width:height) is optional. Without it the pair is square; with it, it goes to the nearest of OneTrainer's
  own aspect buckets at the target resolution (the same buckets images use).
- `teacher` may be a JSON object (structured prompts); it is passed to the model as compact JSON.
- A pair with the same prompt twice keeps that prompt like the base model (against concept bleeding,
  e.g. `"a middle-aged man on a bench"` → itself next to `"Taras Shevchenko on a bench"` → description).
- Generating pairs (templates, LLM rewrites, assigning `ar`) is a job for a preparation script, not the trainer.

Example: `docs/fork/example_pairs.jsonl`.

### Running

- **UI**: load the preset `#anima LoRA context distillation`, set `context_distillation.prompts_path` in the saved
  config JSON, train as usual. Settings without a widget in the UI survive saving and loading.
- **CLI**:
  ```bat
  venv\Scripts\python.exe scripts\train.py --config-path my_config.json --config-value context_distillation.prompts_path=D:/pairs.jsonl
  ```

Modes:
- `standalone: true`: trains only on the prompt pairs, no concepts / images. An epoch is every pair once.
- `standalone: false`: every image-training step also gets one distillation batch (separate backward pass, so the
  memory peak does not grow, except with fused back pass where both losses share one backward). The pairs run
  through their own epochs.
- Warm-start: `stop_after_steps: 300` (optionally `decay_to_zero: true`) with normal image concepts.

Batch size, epochs and the timestep distribution come from the main settings. Batches are grouped by bucket and
an incomplete batch is dropped each epoch (shuffled per epoch, like images), so every bucket needs at least
batch-size pairs.

### Cache

Before the first step, each pair's prompts are encoded and its teacher latent generated (`teacher_steps` sampling
steps with CFG), then saved to `<cache dir>/context_distillation/`. Later runs reuse it and need no text encoder
at all. Entries are regenerated automatically when the pair, its bucket, `teacher_steps`, `teacher_cfg`,
`negative_prompt`, `seed`, the base model or its dtypes change. *Clear cache before training* clears it too.

### Settings (`context_distillation` block)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch |
| `prompts_path` | `""` | Prompt pairs file |
| `loss_weight` | `1.0` | Weight of the distillation loss (matters when mixed with image training) |
| `standalone` | `false` | Train without an image data set |
| `stop_after_steps` | `0` | Stop distilling after N steps (0 = never) |
| `decay_to_zero` | `false` | Fade the weight linearly to 0 at `stop_after_steps` |
| `resolution` | `""` | Same format as the training resolution (`"512"`, `"512,1024"`, `"768x512"`). Empty = training resolution |
| `teacher_steps` | `20` | Sampling steps for the cached teacher latents (set like your normal sampling) |
| `teacher_cfg` | `4.0` | CFG for the cached teacher latents (set like your normal sampling) |
| `negative_prompt` | `""` | Negative prompt for that CFG |
| `seed` | `42` | Seed for the cached latents and training noise |

Cost per step: one teacher forward (no gradients) plus one student forward and backward.

### What to watch

- TensorBoard `loss/context_distillation` should go down. `context_distillation/loss_{high,mid,low}_noise`
  split it by noise level (sigma > 2/3, 1/3–2/3, < 1/3): high noise is where composition and identity are
  decided, low noise only refines details. In standalone mode `loss/train_step` is the same curve.
- Samples: add sample prompts using **student** prompts (they should move toward the teacher content), and one
  prompt that is **not** in the file (it should stay like the base model).

Sampling during training reuses the text embeddings of sample prompts it has already encoded (while the text encoder
is frozen), so the text encoder is only moved to the GPU when a new sample prompt appears (Anima, Krea 2).

Supported models: Anima. Adding one = a small adapter class in `modules/trainer/extension/FlowModelAdapter.py`.

## Feature: depth anchor

Adds a 3D-structure loss to normal image training. Each step, the model's predicted clean image (computed from the
step's own outputs, no extra model forward) and the training image are decoded by a tiny latent decoder and run
through a frozen depth estimator; the difference between the two depth maps is added to the loss. Gradients flow
through the depth estimator and the decoder into the LoRA. It teaches shape and geometry while ignoring colors,
textures and lighting, and it does not mask or reweight the normal diffusion loss.

- The training image's depth is computed on the fly (no gradients), so crops and flips need no special handling and
  nothing is cached.
- The loss compares depth *shapes*: both maps are normalized (median / mean absolute deviation), then
  L1 + a multi-scale gradient term (as in MiDaS).
- Tiny decoder: madebyollin's TAEHV (`taew2_1` for Anima, which uses the Qwen-Image / Wan 2.1 latent space),
  downloaded once (~22 MB) into `<cache dir>/perceptual_models/`.

### Depth models

| `depth_model` | Size | License |
|---|---|---|
| `depth-anything-v2-small` (default) | 25M | Apache-2.0 |
| `depth-anything-v2-base` / `-large` | 98M / 335M | CC-BY-NC-4.0 |
| `da3-small` / `da3-base` | 0.08B / 0.12B (DA3 README) | Apache-2.0 |
| `da3-mono-large` (single-image relative depth) | 0.35B | Apache-2.0 |

Depth Anything V2 works out of the box. Depth Anything 3 is optional and must be installed **without** its
dependencies (they would replace OneTrainer's numpy / torch packages):

```bat
venv\Scripts\pip install --no-deps "git+https://github.com/ByteDance-Seed/Depth-Anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4" addict einops
```

A bigger depth model gives stronger gradients, so `weight` needs tuning per model.

### Settings (`depth_anchor` block)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch |
| `weight` | `0.1` | Weight of the depth loss |
| `depth_model` | `depth-anything-v2-small` | See the table above |
| `depth_resolution` | `518` | Longer side of the depth model input (multiple of 14) |
| `min_noise` / `max_noise` | `0.0` / `1.0` | Only samples with a noise level in this range get the anchor |
| `every_n_steps` | `1` | Apply on every n-th optimizer step |
| `loss_split` | `false` | Anchor steps use only the anchor loss (needs `every_n_steps` ≥ 2; 2 = alternate diffusion / anchor steps) |
| `gradient_weight` | `0.5` | Weight of the multi-scale gradient term |
| `decoder_path` | `""` | Tiny decoder weights; empty = download the default |
| `preview_every` | `100` | TensorBoard preview every n steps (0 = never) |

The preset `#anima LoRA depth anchor` uses the perceptual-repo recipe: `every_n_steps: 2` + `loss_split: true`
(alternating diffusion and anchor steps). For an A/B test, train the same config with `depth_anchor.enabled` off.

### What to watch

- `loss/depth_anchor` and `depth_anchor/loss_{high,mid,low}_noise`.
- `depth_anchor/preview`: training image | its depth | predicted image | predicted depth (noise level in
  `depth_anchor/preview_sigma`). Check that the depth maps look sensible on your images: depth models are trained
  mostly on photos.
- Only image batches with the normal flow-matching target are anchored (prior-prediction samples are skipped).

## Code layout

| File | Purpose |
|---|---|
| `modules/trainer/GenericTrainer.py` | Small refactor: the step's losses come from `_iter_losses()` (image loss + extensions), each back-propagated separately; extensions can adjust the image loss in the same graph (`adjust_image_loss`); standalone extensions can replace the image data set. Plain LoRA training produces bit-identical weights to upstream. |
| `modules/trainer/extension/TrainingExtension.py` | Base class for extra training logic |
| `modules/trainer/extension/ContextDistillation.py` | Context distillation |
| `modules/trainer/extension/FlowModelAdapter.py` | Per-model operations (text encoding, velocity, latent shape) |
| `modules/util/config/ContextDistillationConfig.py` | The `context_distillation` config block |
| `modules/trainer/extension/DepthAnchor.py` | Depth anchor |
| `modules/trainer/extension/perceptual/` | Depth model wrappers, vendored TAEHV tiny decoder (MIT) |
| `modules/trainer/extension/noise_bands.py` | Per-noise-band loss logging |
| `modules/util/config/DepthAnchorConfig.py` | The `depth_anchor` config block |
| `modules/modelSampler/SamplePromptCache.py` | Reuses sample prompt embeddings between sampling rounds |
| `tests/fork/` | CPU tests with a tiny random Anima transformer |

CPU tests (no GPU or model download needed):

```
python -m tests.fork.test_context_distillation_cpu
python -m tests.fork.test_generic_trainer_cpu
python -m tests.fork.test_prompt_files_and_cache_cpu
python -m tests.fork.test_depth_anchor_cpu
```
