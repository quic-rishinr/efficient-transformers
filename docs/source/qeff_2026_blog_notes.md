# QEFF 2026 Blog Notes

This note is the evidence and review companion for [`qeff_2026_blog.md`](qeff_2026_blog.md). It is not intended to read like the published post. It exists to keep the public-facing draft accurate and to track editorial decisions.

## Scope

The current draft is a present-tense showcase of QEFF as it exists in mainline in March 2026.

It deliberately does **not** frame the post as a 2024-vs-2026 retrospective. The user requested a feature-led blog focused on what QEFF can do now, so the draft was rewritten around today's capability surface rather than a historical comparison.

## Included Feature Areas

These areas are explicitly covered in the blog because they are both material and evidenced in the repo:

- Diffusers
  - `#604` added FLUX support.
  - `#669` added Wan Lightning.
  - `#788` added Wan image-to-video support.
- Export and compile improvements
  - `#621` added ONNX subfunctions.
  - `#640` optimized ONNX transform memory and time.
  - `#521` deleted the model after export to save memory.
  - `#674` added memory profiling.
  - `#861`, `#873`, `#821`, and related commits improved large-model export behavior, reuse, and OOM resilience.
  - `#620` / `#471de6f5` added proxy-model export support.
  - `#609` reduced Platform SDK dependency for some QPC-generation paths.
- Auto classes and public API surface
  - `QEFFAutoModelForImageTextToText`
  - `QEFFAutoModelForSpeechSeq2Seq`
  - `QEFFAutoModelForCTC`
  - `QEFFAutoModelForSequenceClassification`
  - `QEffAutoPeftModelForCausalLM`
  - `QEffAutoLoraModelForCausalLM`
  - `QEffFluxPipeline`
  - `QEffWanPipeline`
  - `QEffWanImageToVideoPipeline`
- Runtime and serving
  - Continuous batching
  - Compute Context Length (CCL)
  - SpD and multi-projection heads
  - Prompt-lookup decoding
  - On-device sampling and guided decoding
  - BlockedKV attention
  - Prefix caching
  - Disaggregated serving
- Fine-tuning
  - `QEfficient.cloud.finetune`
  - QAIC and GPU device paths
  - DDP and multi-node training
  - checkpoint resume
  - custom dataset preprocessing
  - gradient accumulation / gradient checkpointing
- Model and modality expansion
  - VLMs
  - embeddings
  - audio
  - diffusion
  - sequence classification
  - PEFT / finite adapters
  - AWQ / GPTQ / FP8 / GGUF

## Source Backbone

Primary repo sources used:

- `docs/source/release_docs.md`
- `docs/source/qeff_autoclasses.md`
- `docs/source/diffuser_classes.md`
- `docs/source/supported_features.rst`
- `docs/source/validate.md`
- `docs/source/quick_start.md`
- `docs/source/finetune.md`
- examples under:
  - `examples/diffusers`
  - `examples/image_text_to_text`
  - `examples/performance`
  - `examples/audio`
  - `examples/embeddings`
  - `examples/sequence_classification`
  - `examples/peft`
  - `examples/disagg_serving`
  - `examples/onboarding_guide`

External source used only for framing cross-check:

- DeepWiki overview page for the repo

## Key Milestone Commits Reviewed

- `614c4cd4` Continuous batching (`#73`)
- `0ef68296` PEFT LoRA (`#85`)
- `afb4645d` AWQ + GPTQ (`#101`)
- `34386eda` Finite LoRAX (`#153`)
- `dc2c509f` / `85467631` QNN compilation support (`#171`, `#187`)
- `d0ee7bce` VLM support (`#267`)
- `3de40722` Disaggregated serving (`#365`)
- `504a850b` QNN compilation path support in `QEFFBaseModel` (`#374`)
- `5c471b66` SpD and multi-projection heads (`#306`)
- `2514c0b6` embedding-model upgrades (`#424`)
- `5576a9d8` Whisper support (`#271`)
- `ca4828c9` Wav2Vec2 onboarding (`#571`)
- `f4ff8035` CCL (`#576`)
- `30c334b3` ONNX subfunctions (`#621`)
- `cc4340d7` BlockedKV (`#618`)
- `4fa07308` ONNX transform memory and time optimization (`#640`)
- `0daa5326` Guided decoding for on-device sampling (`#624`)
- `b78e03f8` initial HF-trainer-based fine-tuning structure (`#626`)
- `c75a6374` checkpoint resume via epochs (`#614`)
- `c76d5eac` multi-node DDP training (`#708`)
- `2535829d` Diffusers / FLUX (`#604`)
- `80b26ebd` Wan Lightning (`#669`)
- `544327a7` Sequence classification (`#729`)
- `815309ec` Qwen3-MoE export OOM risk reduction (`#821`)
- `763fedda` large-model ONNX export improvements (`#861`)
- `60005ac3` ONNX export reuse and subfunction compile coverage expansion (`#873`)
- `ea234987` Wan I2V (`#788`)

## Deliberate Omissions

The draft intentionally de-emphasizes or omits:

- CI maintenance, workflow cleanup, and version bumps
- bug-fix-only model onboardings that do not materially change public capability
- hard performance claims with specific numbers unless the repo itself documents them
- external framework comparisons that would require benchmark evidence not present in the repo
- a timeline-heavy narrative that turns the post into a changelog instead of a showcase

## Review Passes Applied

### Pass 1: completeness

Checked that the draft covered the features explicitly requested by the user:

- Diffusers
- export-time reduction and export-path improvements
- new auto classes
- CCL
- SpD

Then expanded to adjacent must-keep features:

- continuous batching
- on-device sampling and guided decoding
- BlockedKV
- disaggregated serving
- fine-tuning
- VLMs
- embeddings
- audio
- sequence classification
- PEFT / finite adapters
- AWQ / GPTQ / FP8 / GGUF

### Pass 2: framing correction

The original draft over-indexed on "what changed from 2024 to 2026."

The revised draft removes that structure and instead leads with:

1. current workload coverage
2. current serving/runtime depth
3. current export/compile maturity
4. current API surface
5. explicit supported-model proof points
6. a grounded differentiation section based on stack completeness rather than hype

### Pass 3: anti-hallucination

Removed or avoided:

- unsupported benchmark numbers
- unsupported claims about exact speedup ratios
- unsupported claims about usage volume or production adoption
- vague "best-in-class" language without direct evidence

### Pass 4: editorial tightening

The final draft is shaped around one argument:

QEFF is most compelling today because it combines modality breadth, serving depth, deployment-path maturity, and adaptation workflows in a single stack.

## Supporting Visuals

- `image/qeff_2026_capability_map.svg`
- `image/qeff_2026_feature_stack.svg`

## Missing Inputs If You Want a More Aggressive Publish Version

The current draft is publishable without inventing data. If you want a more benchmark-heavy or more overtly competitive version, the safest additions would be:

- an internally validated export-time delta for ONNX subfunctions and later export optimizations
- a validated throughput or TTFT delta for at least one of:
  - CCL
  - BlockedKV
  - SpD
  - on-device sampling
  - disaggregated GPT-OSS serving
- one internally approved comparison chart against a prior QEFF baseline or another deployment path

## Suggested Final QA Before Publication

- Reconfirm the preferred branding between `QEFF`, `QEfficient`, and `Efficient Transformers`.
- Reconfirm whether the post should mention `Cloud AI 100` only, or also broader QNN portability language.
- If Qualcomm marketing wants stronger competitiveness language, insert approved benchmark numbers rather than stronger adjectives.
