# Qfficient Transformers: A Unified Deployment Stack for Multimodal AI on Cloud AI 100

At Qualcomm, we built Qfficient Transformers to make Cloud AI 100 deployment practical across real production workloads, not just isolated model demos. Qfficient Transformers now provides one engineering path from Hugging Face checkpoints to optimized execution for text generation, vision-language models, diffusion pipelines, embeddings, audio, classification, and adapter-based customization.

This post focuses on what is in the stack now: the capabilities we have added, the model depth we support, and the runtime and export improvements that materially reduce deployment friction.

![Qfficient Transformers capability map](image/qeff_2026_capability_map.png)

## What Qfficient Transformers Covers End-to-End

From Qualcomm's perspective, the most important Qfficient Transformers change is stack completeness.

- `Text generation` through `QEFFAutoModelForCausalLM` with continuous batching, CCL, speculative decoding paths, on-device sampling, BlockedKV, prefix caching, disaggregated serving support, quantization options including AWQ, GPTQ, and FP8, plus GGUF execution paths.
- `Vision-language` through `QEFFAutoModelForImageTextToText` with single-QPC and dual-QPC execution, multi-image handling, and continuous batching support for VLM deployment.
- `Diffusion` through `QEffFluxPipeline`, `QEffWanPipeline`, and `QEffWanImageToVideoPipeline` for image generation, text-to-video, and image-to-video flows.
- `Embeddings` through `QEFFAutoModel` with sentence embedding on AI 100, flexible pooling, and compilation across multiple sequence lengths with runtime graph selection.
- `Audio` through `QEFFAutoModelForSpeechSeq2Seq` and `QEFFAutoModelForCTC` for Whisper and Wav2Vec2-style ASR pipelines.
- `Classification` through `QEFFAutoModelForSequenceClassification`, including safety and prompt-guard style tasks.
- `Adaptation` through `QEffAutoPeftModelForCausalLM` and `QEffAutoLoraModelForCausalLM` for PEFT and finite adapter workflows.
- `Fine-tuning` through `QEfficient.cloud.finetune` with QAIC and GPU options, PEFT mode, distributed DDP and multi-node support, checkpoint resume, custom dataset preprocessing, and gradient checkpointing.

## Strong Model Coverage That Matches Production Reality

We have expanded validated coverage across mainstream and high-impact model families:

- `Text generation`: Llama 3.x, GPT-OSS, Qwen3-MoE, Gemma, Granite, Mixtral/Codestral, StarCoder2, OLMo2, Molmo, SwiftKV, and Grok-1.
- `Vision-language`: LLaVA, Llama 3.2 Vision, Llama-4 Scout, Granite Vision, Gemma3, Qwen2.5-VL, Mistral 3.1, InternVL, and Molmo.
- `Embeddings and safety`: BGE, E5, MPNet, Granite Embeddings, NomicBERT, and Llama Prompt Guard.
- `Audio and media`: Whisper, Wav2Vec2, FLUX.1-schnell, Wan 2.2 T2V, and Wan 2.2 I2V.

The important point is not only model count. It is that validated model documentation is explicit about execution entry points and, for VLMs, single-QPC versus dual-QPC capability.

## Runtime Improvements That Move the Throughput Needle

Qfficient Transformers now optimizes serving behavior as a first-class concern:

- `Continuous batching` in primary runtime flows, including VLM scenarios.
- `Compute Context Length (CCL)` to specialize prefill and decode context handling.
- `Speculative decoding` paths including draft-based SpD, prompt-lookup decoding, and multi-projection-head flows.
- `On-device sampling` plus guided decoding to reduce host-device traffic.
- `BlockedKV` and `prefix caching` improvements for decode-path efficiency.
- `Disaggregated serving` support for separate prefill/decode deployments, especially relevant to large and MoE-style workloads.

![Qfficient Transformers feature stack](image/qeff_2026_feature_stack.png)

## Export and Compile Improvements for Faster Iteration

We have invested heavily in reducing export and compile overhead:

- `QEfficient.cloud.infer` as the high-level path for export, compile, and run, with artifact reuse for existing ONNX/QPC outputs.
- `QNN compilation` integrated into high-level auto-model flows instead of isolated side tooling.
- `ONNX subfunctions` support across CausalLM, VLM, and diffuser workflows to reduce repeated export structure cost.
- ONNX transform improvements focused on memory and time reduction, including large-model export stability work and OOM risk reduction.
- Memory profiling and related export diagnostics to improve operational tuning.
- Reduced dependency on Platform SDK for QPC generation workflows.

This is where Qfficient Transformers behaves like deployment infrastructure, not only a conversion layer.

## Why Qfficient Transformers Is Genuinely Competitive

Our strongest competitive position is structural:

- `One stack across many workloads`: text, VLMs, diffusion, embeddings, audio, classification, adaptation, and fine-tuning.
- `Serving features in the default path`: CCL, SpD, BlockedKV, on-device sampling, prefix caching, and disaggregated serving are integrated where teams actually deploy.
- `Lower iteration cost`: reusable artifacts, high-level infer APIs, ONNX subfunctions, and export-path optimizations reduce the time from model onboarding to production-grade runs.

This is a concrete advantage for teams deploying on Cloud AI 100 because the hard problems are usually at the seams between model onboarding, runtime behavior, and operational iteration.

## How to Use the Stack

### Unified infer flow with ONNX subfunctions

```bash
python -m QEfficient.cloud.infer \
    --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --batch_size 1 \
    --prompt_len 32 \
    --ctx_len 128 \
    --num_cores 16 \
    --device_group [0] \
    --prompt "Summarize the deployment stack" \
    --use-onnx-subfunctions
```

### Dual-QPC vision-language inference

```bash
python examples/image_text_to_text/basic_vlm_inference.py \
    --model-name meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --image-url "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg" \
    --query "Describe the image and explain the scene" \
    --kv-offload \
    --prefill-seq-len 128 \
    --ctx-len 3000 \
    --num-cores 16 \
    --num-devices 1
```

### FLUX image generation

```python
import torch
from QEfficient import QEffFluxPipeline

pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

result = pipeline(
    prompt="A laughing girl",
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.manual_seed(42),
    parallel_compile=True,
)

result.images[0].save("girl_laughing.png")
```

### Distributed fine-tuning on QAIC

```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 -m QEfficient.cloud.finetune \
    --device qaic \
    --enable_ddp \
    --use-peft \
    --num_epochs 2 \
    --model_name "meta-llama/Llama-3.2-1B"
```

## Closing

From Qualcomm's point of view, Qfficient Transformers has become a full deployment stack for Cloud AI 100: broad model and modality coverage, runtime features that matter in production, and export and compile workflows built for faster iteration. That combination is what makes Qfficient Transformers practical for teams moving from model evaluation to sustained serving and adaptation at scale.
