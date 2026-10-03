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

## Code layout

| File | Purpose |
|---|---|
| `modules/trainer/GenericTrainer.py` | Small refactor: the step's losses come from `_iter_losses()` (image loss + extensions), each back-propagated separately; standalone extensions can replace the image data set. Plain LoRA training produces bit-identical weights to upstream. |
| `modules/trainer/extension/TrainingExtension.py` | Base class for extra training logic |
| `modules/trainer/extension/ContextDistillation.py` | Context distillation |
| `modules/trainer/extension/FlowModelAdapter.py` | Per-model operations (text encoding, velocity, latent shape) |
| `modules/util/config/ContextDistillationConfig.py` | The `context_distillation` config block |
| `modules/modelSampler/SamplePromptCache.py` | Reuses sample prompt embeddings between sampling rounds |
| `tests/fork/` | CPU tests with a tiny random Anima transformer |

CPU tests (no GPU or model download needed):

```
python -m tests.fork.test_context_distillation_cpu
python -m tests.fork.test_generic_trainer_cpu
python -m tests.fork.test_prompt_files_and_cache_cpu
```
