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

Teaches a LoRA that a **short prompt** should behave like a **dense prompt**, using only text pairs.
No images, no pre-generated dataset.

- **Student**: base model + LoRA, sees the short prompt.
- **Teacher**: the same base model with the LoRA switched off, sees the dense prompt.
- The student learns to predict the teacher's velocity at the same noisy latent.

The noisy latents come from a small pool of teacher latents, generated from the dense prompts during
training (never decoded to images) and re-noised with fresh noise at a random noise level every step.
The pool is refreshed gradually, so the student is trained at all noise levels, on ever-new points.
The pool only decides *where* the student is trained; the target is always the teacher's live prediction.

Uses:
- **Prompt adherence**: make short prompts produce what the model makes from detailed descriptions.
- **Concept warm-start**: `"Taras Shevchenko"` → a dense appearance description. Run it for a few hundred
  steps before (or mixed into the start of) normal image training, so image training starts close to the
  target instead of from nothing.

### Prompt file

`.jsonl` (one object per line) or `.json` (a list of objects):

```json
{"short": "Taras Shevchenko, portrait", "dense": "portrait of a middle-aged Ukrainian man with a bald head, long drooping mustache, ..."}
```

A `dense` value may itself be a JSON object (structured prompts); it is passed to the model as compact JSON.

Two more entry types, for teaching a **trigger word** (example: `docs/fork/trigger_example.jsonl`):

```json
{"trigger": "Taras Shevchenko",
 "description": ["a bald middle-aged man with a long drooping mustache, ...", "another wording of the same look"],
 "contexts": ["{subject} reading a book in a garden", "an oil painting of {subject}"],
 "contexts_file": "contexts_person.txt",
 "generic": "a middle-aged man"}
{"preserve": ["a woman reading a book in a garden", "a cat sleeping on a windowsill"]}
```

- Every context × description gives one pair: short = context with `{subject}` → trigger, dense = the same context
  with `{subject}` → description. Many varied contexts teach the trigger itself rather than one fixed scene.
- `contexts_file` (one context per line, `#` comments, path relative to the prompts file) and `contexts` are combined.
- `generic` adds, per context, a pair where the prompt with the generic words is its own target. This keeps
  "a middle-aged man" meaning what it meant before, so the concept does not bleed into every man.
- `preserve` entries do the same for any prompt you want to stay unchanged.

The preset `#anima LoRA trigger distillation` runs the example file. Good sample prompts: a trigger prompt that is
**not** in the contexts, the same prompt with the generic words, and an unrelated prompt.

### Running

- **UI**: load the preset `#anima LoRA context distillation`, change `context_distillation.prompts_path`
  in the saved config JSON, train as usual. Settings without a widget in the UI survive saving and loading.
- **CLI**:
  ```bat
  venv\Scripts\python.exe scripts\train.py --config-path my_config.json --config-value context_distillation.prompts_path=D:/pairs.jsonl
  ```

Modes:
- `standalone: true`: trains only on the prompt pairs. No concepts / images are used.
  An epoch is `standalone_epoch_length` steps.
- `standalone: false`: every image-training step also gets a distillation step (separate backward pass,
  so the memory peak does not grow, except with fused back pass where both losses share one backward).
- Warm-start: `stop_after_steps: 300` (optionally `decay_to_zero: true`) with normal image concepts.

### Settings (`context_distillation` block)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch |
| `prompts_path` | `""` | Prompt pairs file |
| `loss_weight` | `1.0` | Weight of the distillation loss |
| `batch_size` | `1` | Distillation samples per step |
| `standalone` | `false` | Train without an image data set |
| `standalone_epoch_length` | `100` | Steps per epoch in standalone mode |
| `stop_after_steps` | `0` | Stop distilling after N steps (0 = never) |
| `decay_to_zero` | `false` | Fade the weight linearly to 0 at `stop_after_steps` |
| `resolution` | `""` | Teacher latent resolution, `"512"` or `"768x512"`. Empty = training resolution |
| `pool_size` | `16` | Teacher latents kept in memory |
| `pool_refresh_every` | `8` | Generate one new teacher latent every N steps (0 = never) |
| `teacher_steps` | `20` | Sampling steps per teacher latent |
| `teacher_cfg` | `4.0` | CFG for generating teacher latents |
| `negative_prompt` | `""` | Negative prompt for that CFG |
| `target_cfg` | `1.0` | CFG of the training target. 1.0 = plain conditional prediction (recommended) |
| `seed` | `42` | Seed for pool generation and sampling |

Cost: the initial pool costs `pool_size × teacher_steps` CFG forwards before the first step. Each refresh costs
`teacher_steps` CFG forwards, i.e. on average `teacher_steps × 2 / pool_refresh_every` extra forwards per step.
Every distillation step costs one teacher forward (no gradients) plus one student forward + backward.

### What to watch

- TensorBoard `loss/context_distillation` should go down. `context_distillation/teacher_latents_generated`
  shows that the pool refreshes.
- Samples: add sample prompts using the **short** prompts (they should move toward the dense content), and one
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
