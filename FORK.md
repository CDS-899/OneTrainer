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

**Status: experimental, off by default.** Tested on Anima (person LoRA, int8, Depth Anything V2 Small): at weights
0.01–0.02 it had no measurable effect; at 0.2 it slowed likeness learning while improving the depth loss by only
~1–3%. At high noise the predicted image is a blur whose depth can not be matched, so most of the anchor's gradient
is noise. The old alternating mode (`loss_split`) blocked likeness entirely. `depth_anchor/push_ratio` (measured at
the model output) greatly overstates the anchor's real influence on the LoRA.

Adds a 3D-structure loss to normal image training. Each step, the model's predicted clean image (computed from the
step's own outputs, no extra model forward) and the training image are decoded by a tiny latent decoder and run
through a frozen depth estimator; the difference between the two depth maps is added to the loss. Gradients flow
through the depth estimator and the decoder into the LoRA. It teaches shape and geometry while ignoring colors,
textures and lighting, and it does not mask or reweight the normal diffusion loss.

- The training image's depth is computed on the fly (no gradients), so crops and flips need no special handling and
  nothing is cached.
- The loss compares depth *shapes*, as in ai-toolkit-perceptual / MiDaS: the predicted map is fitted onto the
  training image's map (least-squares scale and shift), then compared by a multi-scale depth-gradient term
  (edges and surface slopes) and optionally an L1 term. Fitting only the prediction keeps the loss stable when the
  prediction is a blur (high noise).
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
| `weight` | `0.005` | Weight of the depth loss |
| `depth_model` | `depth-anything-v2-small` | See the table above |
| `depth_resolution` | `518` | Longer side of the depth model input (multiple of 14) |
| `min_noise` / `max_noise` | `0.0` / `1.0` | Only samples with a noise level in this range get the anchor |
| `every_n_steps` | `1` | Apply on every n-th optimizer step |
| `loss_split` | `false` | Anchor steps use only the anchor loss (needs `every_n_steps` ≥ 2; 2 = alternate diffusion / anchor steps) |
| `ssi_weight` | `0.0` | Weight of the aligned L1 term |
| `gradient_weight` | `1.0` | Weight of the multi-scale gradient term |
| `gradient_scales` | `6` | Number of scales for the gradient term |
| `decoder_path` | `""` | Tiny decoder weights; empty = download the default |
| `preview_every` | `100` | TensorBoard preview every n steps (0 = never) |

The preset `#anima LoRA depth anchor` adds the anchor to the normal loss on every step (`weight: 0.01`, Depth
Anything V2 Small) and skips low-noise samples (`min_noise: 0.3`): there the predicted image is already almost the
training image, so depth has nothing left to correct and the step belongs to fine detail.

**Avoid `loss_split` for subjects.** Adam scales every update to a similar size whatever the loss value, so an
anchor-only step is a full-size step on depth no matter how small `weight` is: the weight stops mattering, half of
the training goes to depth, and likeness is learned much slower. In sum mode (the default) the weight really sets
the balance.

**Tuning `weight`**: train the same config (seed, LR, steps) with weights like 0, 0.005, 0.02, 0.05 and keep the
largest one that still learns likeness about as fast as 0. Bigger depth models give stronger gradients and need a
smaller weight (the perceptual repo uses 0.005 with V2 Large).

### What to watch

- `loss/depth_anchor` and `depth_anchor/loss_{high,mid,low}_noise`.
- `depth_anchor/push_ratio` (logged with each preview): ratio of the anchor's and the diffusion loss's gradients at
  the model output. It overstates the anchor's real influence by a large factor (see Status), so treat it as a
  rough relative number only.
- `depth_anchor/preview`: training image | its depth | predicted image | predicted depth (noise level in
  `depth_anchor/preview_sigma`). Check that the depth maps look sensible on your images: depth models are trained
  mostly on photos.
- Only image batches with the normal flow-matching target are anchored (prior-prediction samples are skipped).

## Feature: SOAR (self-correction during image training)

From Tencent's HY-SOAR (arXiv 2604.12617), written from the paper and its algorithm. Normal training only shows the
model noisy versions of real images, but when sampling it continues from its own slightly wrong states, and small
errors pile up. On SOAR steps, after the normal prediction:

1. **Rollout**: from the step's noisy image the model takes one sampling step itself, with CFG against the empty
   prompt (one step of a `rollout_steps` schedule).
2. **Correction points**: that state is re-noised toward the same noise, at random noise levels up to pure noise.
3. Each point is trained toward the velocity that leads from it straight back to the real image. If the model's step
   was perfect, this is exactly the normal target, so SOAR never pulls away from the training images.

The loss of a SOAR step is `(normal + weight * sum(points)) / (1 + weight * N)`, as in the paper (points counted over
the samples SOAR applies to). For a "final stage", resume a normally trained LoRA or fine-tune with `soar.enabled` on.

**Difference from the paper: no SOAR at low noise.** Samples whose training noise level is below `min_noise` get
normal training only, and a rollout step that ends at zero noise is never corrected. Near zero noise the correction
target divides a small texture error by a tiny noise level: in the first Anima test (batch 2, SOAR at all noise
levels) these points caused loss spikes of 20-40x and a regular mesh texture in samples. The paper trains with batch
32, where such points average out. Low-noise steps only refine texture, so little is lost.

Cost per SOAR step (compared to a normal step; first Anima test: 1.85x with 1 point and CFG): one extra forward pass without gradients for CFG (none with
`rollout_cfg: 1`; the conditional half reuses the normal prediction), plus one forward and backward per correction
point. Roughly 2.3x with 1 point and CFG, 3.3x with 2. Each point is back-propagated on its own, so the memory peak
stays that of a normal step (except with fused back pass, where all losses of a step share one backward). The extra
passes use the same batch shape as the normal step, so compiled graphs are reused.

The empty prompt is encoded once and cached in `<cache dir>/soar/` (cleared by *Clear cache before training*, then
re-encoded at the next start). Needs a frozen text encoder. Prior-prediction batches are skipped.

### Settings (`soar` block)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch |
| `correction_points` | `1` | Correction points per SOAR step: the main cost dial (paper: 6) |
| `weight` | `1.0` | Weight of each point relative to the normal loss (paper: 1) |
| `rollout_cfg` | `4.5` | CFG of the rollout step against the empty prompt. `1` = no CFG, no extra pass |
| `rollout_steps` | `20` | The rollout takes one step of a sampling schedule with this many steps |
| `every_n_steps` | `1` | SOAR on every n-th optimizer step, normal training on the others |
| `min_noise` | `0.2` | SOAR only for samples at this training noise level or above; the others get normal training |

Prefer more steps with few points over many points every few steps: the points of one step share one image, noise
and rollout, so they are strongly correlated.

### What to watch

- `loss/soar` and `soar/loss_{high,mid,low}_noise`: the correction loss. It is larger than the normal loss because
  low-noise points ask for the whole remaining error at once.
- `soar/rollout_error`: RMS distance between the model's own rollout step and the training trajectory, the error
  SOAR corrects. It should shrink as training goes on.
- Samples: compare with a normal run at the same steps, for likeness and for hands, eyes and mushy details.

Preset: `#anima LoRA soar`.

### Test results

Anima person LoRA (int8, batch 2, 1000 steps), SOAR at all noise levels (before `min_noise`) vs normal training,
same seed: at the same step SOAR was closer to the training image for the exact training prompt (step 950), about
even at equal GPU time. Other prompts showed a regular mesh texture, traced to the low-noise correction points, which
`min_noise` now excludes. Not yet re-tested.

## Feature: SMD distillation (`smd_distillation`)

Prompt-only distillation of a LoRA that samples faster: the core of SMD (*Simplified Matching Distillation*,
arXiv 2610.18472, written from the paper; its training-only Pivot head is not included: on images it changed scores
by under 1%).

- **Student**: base model + LoRA, samples with plain Euler on the model's own schedule, at CFG 1.
- **Teacher**: the same model with the LoRA off, at CFG `teacher_cfg` against `negative_prompt`.
- The student runs its own trajectory from noise. For each of its steps it makes one velocity prediction, the
  teacher is asked at random points along that step (at the states the student's prediction leads to), and the
  student's prediction learns to match. So it is trained exactly on the states it visits when sampling.
- One training step trains one student step: a batch of prompts goes through all steps in consecutive training
  steps, then the next batch starts from fresh noise. An epoch runs every prompt once.
- No helper "fake" model and no discriminator (the parts that make DMD / TDM unstable): plain regression onto the
  teacher.

`student_steps: 4` (default) distills steps and CFG at once. With `student_steps` at your normal step count (e.g. 20)
it is CFG-only distillation: same steps and look, 2x faster sampling (set `repulsion_weight: 0` for a plain copy of
the teacher).

Sample the result with Euler, the same step count and CFG 1 (set the sample settings accordingly; in ComfyUI use the
model's normal schedule). The training is standalone (no image data set); batch size and epochs come from the main
settings. Requires LoRA training.

### Prompt file

`.jsonl`, one prompt per line, `"ar"` (width:height) optional:

```json
{"prompt": "a lone samurai walking through a bamboo forest in the rain", "ar": "3:4"}
```

The paper found ~300 prompts already enough and even a single prompt to work, but the *kind* of prompts matters:
training on text-heavy prompts improved text rendering the most. Cover what the model should be good at.
Prompts are encoded once and cached in `<cache dir>/smd_distillation/` (about 1 MB each for Anima). Example: `docs/fork/example_prompts.jsonl`.

### Settings

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch |
| `prompts_path` | `""` | Prompt file |
| `resolution` | `""` | Same format as the training resolution; empty = the training resolution |
| `student_steps` | `4` | Sampling steps of the distilled LoRA |
| `teacher_cfg` | `5.0` | Teacher CFG (paper: 5 for Qwen-Image) |
| `repulsion_weight` | `0.2` | Push away from the teacher's negative-prompt prediction (paper: 0.2) |
| `queries` | `2` | Teacher queries per student step |
| `negative_prompt` | `""` | Negative prompt for the teacher's CFG and the repulsion |
| `seed` | `42` | Seed for noise and query points |

Cost per training step: one student forward and backward plus, per query, two teacher forwards without gradients
(one when `teacher_cfg` is 1 and there is no repulsion). The paper trained Qwen-Image for 1,000 iterations at batch 16,
learning rate 1e-4, rank-256 LoRA. Preset: `#anima LoRA smd distillation`.

### What to watch

- `loss/smd_distillation` and its noise bands: the distance between the student's and the
  teacher's velocity, by the noise level where the teacher was asked. The high-noise band decides composition.
- `smd_distillation/loss_step_N` (up to 8 student steps): the same per student step, 1 = the first step from noise.
- `smd_distillation/repulsion_distance`: distance from the negative-prompt prediction; it should grow a little.
- Samples at the student's step count and CFG 1.

## Code layout

| File | Purpose |
|---|---|
| `modules/trainer/GenericTrainer.py` | Small refactor: the step's losses come from `_iter_losses()` (image loss + extensions), each back-propagated separately; extensions can adjust the image loss in the same graph (`adjust_image_loss`); standalone extensions can replace the image data set. Plain LoRA training produces bit-identical weights to upstream. |
| `modules/trainer/extension/TrainingExtension.py` | Base class for extra training logic |
| `modules/trainer/extension/ContextDistillation.py` | Context distillation |
| `modules/trainer/extension/FlowModelAdapter.py` | Per-model operations (text encoding, velocity, latent shape) |
| `modules/util/config/ContextDistillationConfig.py` | The `context_distillation` config block |
| `modules/trainer/extension/DepthAnchor.py` | Depth anchor |
| `modules/trainer/extension/Soar.py` | SOAR |
| `modules/trainer/extension/OnPolicyDistillation.py` | SMD distillation |
| `modules/util/config/SmdDistillationConfig.py` | The `smd_distillation` config block |
| `modules/util/config/SoarConfig.py` | The `soar` config block |
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
python -m tests.fork.test_soar_cpu
python -m tests.fork.test_on_policy_distillation_cpu
```
