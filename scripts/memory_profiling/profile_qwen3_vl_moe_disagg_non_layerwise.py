# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Memory profiling harness for non-layerwise Qwen3-VL-MoE disagg compile."""

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SCRIPT = REPO_ROOT / "examples/image_text_to_text/models/qwen3_vl_moe/qwen3_vl_disagg_mode.py"
DEFAULT_HF_HUB_CACHE = "/home/huggingface_hub"
DEFAULT_QEFF_HOME = "/home/rishinr/qwen3vl_235b_2layer_non_layerwise_disagg_profile"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"


def _set_required_environment() -> None:
    os.environ.setdefault("HF_HUB_CACHE", DEFAULT_HF_HUB_CACHE)
    os.environ.setdefault("QEFF_HOME", DEFAULT_QEFF_HOME)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


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
        "hf_hub_enable_hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"),
        "sampling_interval": args.sampling_interval,
        "child_scan_interval": args.child_scan_interval,
        "model_id": args.model_id,
        "local_files_only": not args.allow_download,
        "text_layers": args.text_layers,
        "vision_depth": args.vision_depth,
        "layerwise": False,
        "compile_components": ["vision", "decode", "prefill"],
        "compile_options": {
            "mxfp6_matmul": args.mxfp6_matmul,
            "mxint8_kv_cache": args.mxint8_kv_cache,
            "host_memory_mode": "full",
        },
    }

    profiler.start_monitoring()
    try:
        install_global_hooks(detailed_profiler)

        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor

        from QEfficient import QEFFAutoModelForImageTextToText

        local_files_only = not args.allow_download
        detailed_profiler.call("setup: torch.manual_seed", torch.manual_seed, args.random_seed)
        config = detailed_profiler.call(
            "hf config: AutoConfig.from_pretrained",
            AutoConfig.from_pretrained,
            args.model_id,
            local_files_only=local_files_only,
        )
        detailed_profiler.mark("config: mutate dtype/text_layers")
        config.dtype = "float16"
        config.torch_dtype = torch.float16
        config.text_config.num_hidden_layers = args.text_layers
        _maybe_reduce_vision_depth(config, args.vision_depth)
        detailed_profiler.mark("post: config: mutate dtype/text_layers")

        detailed_profiler.call(
            "hf tokenizer: AutoTokenizer.from_pretrained",
            transformers.AutoTokenizer.from_pretrained,
            args.model_id,
            local_files_only=local_files_only,
        )
        detailed_profiler.call(
            "hf processor: AutoProcessor.from_pretrained",
            AutoProcessor.from_pretrained,
            args.model_id,
            local_files_only=local_files_only,
        )

        qeff_model = detailed_profiler.call(
            "qeff load: QEFFAutoModelForImageTextToText.from_pretrained",
            QEFFAutoModelForImageTextToText.from_pretrained,
            args.model_id,
            attn_implementation="eager",
            kv_offload=True,
            config=config,
            dtype=torch.float16,
            layerwise=False,
            local_files_only=local_files_only,
        )
        install_model_hooks(qeff_model, detailed_profiler)

        vision_qpc_path = detailed_profiler.call(
            "compile request",
            qeff_model.compile,
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
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            layerwise=False,
        )

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
            layerwise=False,
            offload_pt_weights=False,
        )

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
            layerwise=False,
            offload_pt_weights=True,
        )
        run_metadata.update(
            {
                "vision_qpc_path": str(vision_qpc_path),
                "prefill_qpc_path": str(prefill_qpc_path),
                "decode_qpc_path": str(decode_qpc_path),
                "completed_at": datetime.now().isoformat(),
                "status": "success",
                "mode": "compile_only",
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
    run_id = datetime.now().strftime("qwen3_vl_235b_2layer_non_layerwise_disagg_%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=run_id)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--sampling-interval", type=float, default=0.05)
    parser.add_argument("--child-scan-interval", type=float, default=0.1)
    parser.add_argument("--text-layers", type=int, default=2)
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
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    start_monotonic = time.monotonic()
    try:
        run_profile(parse_args())
    finally:
        elapsed = time.monotonic() - start_monotonic
        print(f"Detailed Qwen3-VL-MoE non-layerwise disagg profiling elapsed time: {elapsed:.1f}s")
