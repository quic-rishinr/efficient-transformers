# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Full end-to-end memory profiling harness for Qwen3-VL-MoE disagg mode."""

import argparse
import copy
import ctypes
import gc
import json
import os
import shutil
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SCRIPT = REPO_ROOT / "examples/image_text_to_text/models/qwen3_vl_moe/qwen3_vl_disagg_mode.py"
DEFAULT_HF_HUB_CACHE = "/home/huggingface_hub"
DEFAULT_QEFF_HOME = "/home/rishinr/qwen3vl_235b_full_layer_disagg_e2e_profile"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"


def _set_required_environment() -> None:
    os.environ.setdefault("HF_HUB_CACHE", DEFAULT_HF_HUB_CACHE)
    os.environ.setdefault("QEFF_HOME", DEFAULT_QEFF_HOME)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    tmpdir = Path(os.environ["QEFF_HOME"]) / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmpdir))


_set_required_environment()

from profile_qwen3_5_disagg_mode import (  # noqa: E402
    DetailedProfiler,
    QEffMemoryProfiler,
    install_global_hooks,
    install_model_hooks,
    write_outputs,
)


def _maybe_reduce_vision_depth(config: Any, vision_depth: Optional[int]) -> None:
    if vision_depth is None or not hasattr(config, "vision_config"):
        return

    config.vision_config.depth = vision_depth
    deepstack_indexes = list(getattr(config.vision_config, "deepstack_visual_indexes", []))
    valid_deepstack_indexes = [idx for idx in deepstack_indexes if idx < vision_depth]
    if deepstack_indexes and not valid_deepstack_indexes:
        valid_deepstack_indexes = [vision_depth - 1]
    if deepstack_indexes:
        config.vision_config.deepstack_visual_indexes = valid_deepstack_indexes


def _make_local_image(width: int, height: int):
    from PIL import Image

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = x
    image[..., 1] = y
    image[..., 2] = ((image[..., 0].astype(np.uint16) + image[..., 1].astype(np.uint16)) // 2).astype(np.uint8)
    image[: height // 3, : width // 3, :] = (220, 40, 40)
    image[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3, :] = (40, 180, 80)
    image[2 * height // 3 :, 2 * width // 3 :, :] = (50, 90, 220)
    return Image.fromarray(image, mode="RGB")


def _release_unused_host_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


@contextmanager
def _skip_missing_key_initialization_for_layerwise_filtering():
    import transformers.modeling_utils as modeling_utils

    original = modeling_utils.PreTrainedModel._initialize_missing_keys

    def _no_initialize_missing_keys(self, *args, **kwargs):
        return None

    modeling_utils.PreTrainedModel._initialize_missing_keys = _no_initialize_missing_keys
    try:
        yield
    finally:
        modeling_utils.PreTrainedModel._initialize_missing_keys = original


def _update_retained_states(target_inputs: Dict[str, Any], source_outputs: Dict[str, Any], num_hidden_layers: int) -> None:
    for layer_idx in range(num_hidden_layers):
        target_inputs[f"past_key.{layer_idx}"] = source_outputs[f"past_key.{layer_idx}_RetainedState"]
        target_inputs[f"past_value.{layer_idx}"] = source_outputs[f"past_value.{layer_idx}_RetainedState"]


def _select_next_token_id(logits: np.ndarray) -> int:
    if logits.ndim == 2:
        token_scores = logits[0]
    elif logits.ndim == 3:
        token_scores = logits[0, -1]
    else:
        raise ValueError(f"Unsupported logits shape for token selection: {logits.shape}")
    return int(np.argmax(token_scores))


def _quarantine_layerwise_artifacts(qpc_paths: Dict[str, Any], qpc_key: str, suffix: str) -> Optional[str]:
    qpc_path = qpc_paths.get(qpc_key) if isinstance(qpc_paths, dict) else None
    if qpc_path is None:
        return None

    export_root = Path(qpc_path).parent.parent
    candidates = []
    for pattern in ("merged_*.onnx", "layer_*_0.onnx.data", "pref_*.onnx"):
        candidates.extend(export_root.glob(pattern))
    candidates.extend(path for path in (export_root / "final_data", export_root / "onnx_layerwise_tmp") if path.exists())
    if not candidates:
        return None

    quarantine_dir = export_root / f"{suffix}_layerwise_artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        shutil.move(str(path), str(quarantine_dir / path.name))
    return str(quarantine_dir)


def run_profile(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(os.environ["QEFF_HOME"]) / "memory_profiles" / args.run_id
    )
    profiler = QEffMemoryProfiler(
        sampling_interval=args.sampling_interval,
        output_file=str(output_dir / "memory_timeline.png"),
        verbose=args.verbose,
        enable_cpu_monitoring=True,
        enable_disk_monitoring=True,
        track_child_processes=True,
        child_scan_interval=args.child_scan_interval,
    )
    detailed_profiler = DetailedProfiler(profiler)
    output_paths: Dict[str, str] = {}
    vision_qpc_path = None
    prefill_qpc_path = None
    decode_qpc_path = None

    run_metadata: Dict[str, Any] = {
        "target_script": str(TARGET_SCRIPT),
        "started_at": datetime.now().isoformat(),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "qeff_home": os.environ.get("QEFF_HOME"),
        "tmpdir": os.environ.get("TMPDIR"),
        "hf_hub_enable_hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"),
        "sampling_interval": args.sampling_interval,
        "child_scan_interval": args.child_scan_interval,
        "model_id": args.model_id,
        "local_files_only": not args.allow_download,
        "requested_text_layers": args.text_layers,
        "requested_vision_depth": args.vision_depth,
        "layerwise": args.layerwise,
        "layerwise_window_size": args.layerwise_window_size,
        "skip_missing_key_initialization_for_layerwise_filtering": True,
        "compile_order": ["vision", "decode", "prefill"],
        "runtime_decode_steps": args.runtime_decode_steps,
            "compile_options": {
            "batch_size": args.batch_size,
            "prefill_seq_len": args.prefill_seq_len,
            "ctx_len": args.ctx_len,
            "num_cores": args.num_cores,
            "vision_num_devices": args.vision_num_devices,
            "prefill_num_devices": args.prefill_num_devices,
            "decode_num_devices": args.decode_num_devices,
            "height": args.height,
            "width": args.width,
            "mxfp6_matmul": args.mxfp6_matmul,
            "mxint8_kv_cache": args.mxint8_kv_cache,
            "export_dtype": "float16",
            "host_memory_mode": "layerwise" if args.layerwise else "full",
            "aic_enable_depth_first": True,
            "split_model_io": True,
            "use_onnx_subfunctions": True,
            "mos": 1,
        },
    }

    profiler.start_monitoring()
    try:
        install_global_hooks(detailed_profiler)

        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoConfig, AutoProcessor

        from QEfficient import QEFFAutoModelForImageTextToText
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        local_files_only = not args.allow_download
        detailed_profiler.call("setup: torch.manual_seed", torch.manual_seed, args.random_seed)

        config = detailed_profiler.call(
            "hf config: AutoConfig.from_pretrained",
            AutoConfig.from_pretrained,
            args.model_id,
            local_files_only=local_files_only,
        )
        original_text_layers = config.text_config.num_hidden_layers
        original_vision_depth = getattr(config.vision_config, "depth", None)
        detailed_profiler.mark("config: mutate dtype/optional layer limits")
        config.dtype = "float16"
        config.torch_dtype = torch.float16
        if args.text_layers is not None:
            config.text_config.num_hidden_layers = args.text_layers
        _maybe_reduce_vision_depth(config, args.vision_depth)
        resolved_text_layers = config.text_config.num_hidden_layers
        resolved_vision_depth = getattr(config.vision_config, "depth", None)
        detailed_profiler.mark("post: config: mutate dtype/optional layer limits")

        run_metadata.update(
            {
                "original_text_layers": original_text_layers,
                "original_vision_depth": original_vision_depth,
                "resolved_text_layers": resolved_text_layers,
                "resolved_vision_depth": resolved_vision_depth,
                "vision_export_text_layers": 1,
            }
        )

        tokenizer = detailed_profiler.call(
            "hf tokenizer: AutoTokenizer.from_pretrained",
            transformers.AutoTokenizer.from_pretrained,
            args.model_id,
            local_files_only=local_files_only,
        )
        processor = detailed_profiler.call(
            "hf processor: AutoProcessor.from_pretrained",
            AutoProcessor.from_pretrained,
            args.model_id,
            local_files_only=local_files_only,
        )

        vision_config = copy.deepcopy(config)
        vision_config.text_config.num_hidden_layers = 1
        qeff_vision_model = detailed_profiler.call(
            "qeff load vision-only wrapper: QEFFAutoModelForImageTextToText.from_pretrained",
            QEFFAutoModelForImageTextToText.from_pretrained,
            args.model_id,
            attn_implementation="eager",
            kv_offload=True,
            config=vision_config,
            dtype=torch.float16,
            layerwise=False,
            local_files_only=local_files_only,
        )
        install_model_hooks(qeff_vision_model, detailed_profiler)

        vision_qpc_path = detailed_profiler.call(
            "compile request",
            qeff_vision_model.compile,
            component_name="vision",
            batch_size=args.batch_size,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            height=args.height,
            width=args.width,
            num_cores=args.num_cores,
            num_devices=args.vision_num_devices,
            mos=1,
            mxfp6_matmul=args.mxfp6_matmul,
            mxint8_kv_cache=args.mxint8_kv_cache,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            layerwise=False,
        )
        detailed_profiler.mark("vision: cleanup vision-only wrapper")
        del qeff_vision_model
        del vision_config
        detailed_profiler.call("release host memory after vision compile", _release_unused_host_memory)

        language_load_kwargs = {
            "attn_implementation": "eager",
            "kv_offload": True,
            "config": config,
            "dtype": torch.float16,
            "layerwise": args.layerwise,
            "local_files_only": local_files_only,
        }
        if args.layerwise:
            language_load_kwargs["layerwise_window_size"] = args.layerwise_window_size
        qeff_model = detailed_profiler.call(
            "qeff load language wrapper: QEFFAutoModelForImageTextToText.from_pretrained",
            QEFFAutoModelForImageTextToText.from_pretrained,
            args.model_id,
            **language_load_kwargs,
        )
        install_model_hooks(qeff_model, detailed_profiler)

        with _skip_missing_key_initialization_for_layerwise_filtering():
            decode_qpc_path = detailed_profiler.call(
                "compile request",
                qeff_model.compile,
                component_name="decode",
                batch_size=args.batch_size,
                prefill_seq_len=1,
                ctx_len=args.ctx_len,
                height=args.height,
                width=args.width,
                num_cores=args.num_cores,
                num_devices=args.decode_num_devices,
                mxfp6_matmul=args.mxfp6_matmul,
                mxint8_kv_cache=args.mxint8_kv_cache,
                split_model_io=True,
                mos=1,
                aic_enable_depth_first=True,
                prefill_only=False,
                skip_vision=True,
                use_onnx_subfunctions=True,
                layerwise=args.layerwise,
                layerwise_window_size=args.layerwise_window_size,
                offload_pt_weights=False,
            )
        decode_layerwise_artifacts = detailed_profiler.call(
            "decode: quarantine layerwise artifacts",
            _quarantine_layerwise_artifacts,
            decode_qpc_path,
            "lang_decode_qpc_path",
            "decode",
            component_name="decode",
        )
        run_metadata["decode_layerwise_artifacts"] = decode_layerwise_artifacts
        qeff_model.lang_model.onnx_path = None

        with _skip_missing_key_initialization_for_layerwise_filtering():
            prefill_qpc_path = detailed_profiler.call(
                "compile request",
                qeff_model.compile,
                component_name="prefill",
                batch_size=args.batch_size,
                prefill_seq_len=args.prefill_seq_len,
                ctx_len=args.ctx_len,
                height=args.height,
                width=args.width,
                num_cores=args.num_cores,
                num_devices=args.prefill_num_devices,
                mxfp6_matmul=args.mxfp6_matmul,
                mxint8_kv_cache=args.mxint8_kv_cache,
                retain_full_kv=True,
                split_model_io=True,
                mos=1,
                aic_enable_depth_first=True,
                prefill_only=True,
                enable_chunking=True,
                skip_vision=True,
                use_onnx_subfunctions=True,
                layerwise=args.layerwise,
                layerwise_window_size=args.layerwise_window_size,
                offload_pt_weights=True,
            )

        lang_prefill_session = detailed_profiler.call(
            "QAICInferenceSession: prefill",
            QAICInferenceSession,
            prefill_qpc_path.get("lang_prefill_qpc_path"),
            component_name="runtime",
        )
        lang_decode_session = detailed_profiler.call(
            "QAICInferenceSession: decode",
            QAICInferenceSession,
            decode_qpc_path.get("lang_decode_qpc_path"),
            component_name="runtime",
        )
        vision_session = detailed_profiler.call(
            "QAICInferenceSession: vision",
            QAICInferenceSession,
            vision_qpc_path.get("vision_qpc_path"),
            component_name="runtime",
        )

        image = detailed_profiler.call(
            "build deterministic local image",
            _make_local_image,
            args.width,
            args.height,
            component_name="runtime",
        )
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]
        ] * args.batch_size

        texts = detailed_profiler.call(
            "processor.apply_chat_template",
            lambda: [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages],
            component_name="runtime",
        )
        image_inputs, video_inputs = detailed_profiler.call(
            "process_vision_info",
            process_vision_info,
            messages,
            component_name="runtime",
        )
        inputs = detailed_profiler.call(
            "processor multimodal inputs",
            processor,
            component_name="runtime",
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = detailed_profiler.call(
            "prepare_inputs_for_generation",
            qeff_model.model.prepare_inputs_for_generation,
            component_name="runtime",
            inputs=inputs,
            prefill_seq_len=args.prefill_seq_len,
            batch_size=args.batch_size,
        )

        pad_token_id = 1
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -args.prefill_seq_len)
        padded_len = num_chunks * args.prefill_seq_len
        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        for key, value in inputs.items():
            inputs[key] = np.array(value)

        vision_inputs = {
            key: value
            for key, value in inputs.items()
            if key
            in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
        }
        vision_inputs.update(
            {
                key: vision_inputs[key].astype("float16")
                for key in {"pixel_values", "image_masks"}
                if key in vision_inputs
            }
        )
        vision_outputs = detailed_profiler.call(
            "vision_session.run", vision_session.run, vision_inputs, component_name="runtime"
        )

        lang_inputs = {key: value for key, value in inputs.items() if key not in vision_inputs}
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask")
        else:
            lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

        lang_inputs["image_idx"] = np.array([[0]])
        lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]
        lang_inputs["deepstack_features"] = vision_outputs["deepstack_features"]

        all_outputs = []
        chunk_inputs = lang_inputs.copy()
        outputs = None
        detailed_profiler.call(
            "lang_prefill_session.set_buffers vision outputs",
            lang_prefill_session.set_buffers,
            vision_outputs,
            component_name="runtime",
        )
        for chunk_idx in range(num_chunks):
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][
                :, chunk_idx * args.prefill_seq_len : (chunk_idx + 1) * args.prefill_seq_len
            ]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., chunk_idx * args.prefill_seq_len : (chunk_idx + 1) * args.prefill_seq_len
            ]
            outputs = detailed_profiler.call(
                f"lang_prefill_session.run chunk {chunk_idx}",
                lang_prefill_session.run,
                chunk_inputs,
                component_name="runtime",
            )
            _update_retained_states(chunk_inputs, outputs, resolved_text_layers)
            chunk_inputs["image_idx"] = outputs["image_idx_output"]

        if outputs is None:
            raise RuntimeError("Prefill produced no outputs because num_chunks was zero")

        first_token = _select_next_token_id(outputs["logits"])
        all_outputs.append(first_token)
        decode_inputs = {
            "input_ids": np.array(first_token).reshape(1, 1),
            "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
        }
        _update_retained_states(decode_inputs, outputs, resolved_text_layers)

        decode_out = detailed_profiler.call(
            "lang_decode_session.run first token",
            lang_decode_session.run,
            decode_inputs,
            component_name="runtime",
        )
        next_token = _select_next_token_id(decode_out["logits"])
        all_outputs.append(next_token)
        pos_id = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
        loop_decode_inputs = {
            "input_ids": np.array(next_token).reshape(1, 1),
            "position_ids": pos_id,
        }
        _update_retained_states(loop_decode_inputs, decode_out, resolved_text_layers)

        for token_idx in range(max(args.runtime_decode_steps - 1, 0)):
            decode_out = detailed_profiler.call(
                f"lang_decode_session.run token {token_idx + 2}",
                lang_decode_session.run,
                loop_decode_inputs,
                component_name="runtime",
            )
            next_token = _select_next_token_id(decode_out["logits"])
            all_outputs.append(next_token)
            pos_id += 1
            _update_retained_states(loop_decode_inputs, decode_out, resolved_text_layers)
            loop_decode_inputs.update({"input_ids": np.array(next_token).reshape(1, 1), "position_ids": pos_id})

        decoded_output = detailed_profiler.call(
            "tokenizer.decode",
            tokenizer.decode,
            all_outputs,
            component_name="runtime",
        )

        run_metadata.update(
            {
                "vision_qpc_path": str(vision_qpc_path),
                "prefill_qpc_path": str(prefill_qpc_path),
                "decode_qpc_path": str(decode_qpc_path),
                "num_prefill_chunks": int(num_chunks),
                "padded_input_len": int(padded_len),
                "generated_token_count": len(all_outputs),
                "generated_token_ids": all_outputs,
                "decoded_output": decoded_output,
                "completed_at": datetime.now().isoformat(),
                "status": "success",
            }
        )
    except Exception as exc:
        run_metadata.update(
            {
                "completed_at": datetime.now().isoformat(),
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "vision_qpc_path": str(vision_qpc_path),
                "prefill_qpc_path": str(prefill_qpc_path),
                "decode_qpc_path": str(decode_qpc_path),
            }
        )
        raise
    finally:
        if profiler.monitoring:
            profiler.stop_monitoring()
        output_paths = write_outputs(profiler, output_dir, run_metadata)
        print(json.dumps({"metadata": run_metadata, "outputs": output_paths}, indent=2, default=str))

    return {"metadata": run_metadata, "outputs": output_paths}


def parse_args() -> argparse.Namespace:
    run_id = datetime.now().strftime("qwen3_vl_235b_full_layer_disagg_e2e_%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=run_id)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--sampling-interval", type=float, default=0.05)
    parser.add_argument("--child-scan-interval", type=float, default=0.1)
    parser.add_argument("--text-layers", type=int, default=None)
    parser.add_argument("--vision-depth", type=int, default=None)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--vision-num-devices", type=int, default=1)
    parser.add_argument("--prefill-num-devices", type=int, default=1)
    parser.add_argument("--decode-num-devices", type=int, default=1)
    parser.add_argument("--height", type=int, default=354)
    parser.add_argument("--width", type=int, default=536)
    parser.add_argument(
        "--mxfp6-matmul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MXFP6 MatMul weight compression for compile. Use --no-mxfp6-matmul only for fp16 QPC experiments.",
    )
    parser.add_argument(
        "--mxint8-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MXINT8 KV-cache custom IO. Use --no-mxint8-kv-cache for fully fp16 IO.",
    )
    parser.add_argument(
        "--layerwise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use layerwise export to lower host RAM. Default is full export to measure true fp16 host peak.",
    )
    parser.add_argument("--layerwise-window-size", type=int, default=1)
    parser.add_argument("--runtime-decode-steps", type=int, default=8)
    parser.add_argument("--prompt", default="Describe all the colors seen in the image.")
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    start_monotonic = time.monotonic()
    try:
        run_profile(parse_args())
    finally:
        elapsed = time.monotonic() - start_monotonic
        print(f"Detailed Qwen3-VL-MoE full-layer disagg e2e profiling elapsed time: {elapsed:.1f}s")
