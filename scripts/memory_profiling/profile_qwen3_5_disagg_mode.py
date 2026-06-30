# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Detailed memory profiling harness for qwen3_5_disagg_mode.py."""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SCRIPT = REPO_ROOT / "examples/image_text_to_text/models/qwen3_5_moe/qwen3_5_disagg_mode.py"
DEFAULT_HF_HUB_CACHE = "/home/huggingface_hub"
DEFAULT_QEFF_HOME = "/home/rishinr/qwen3vl_memory_1121_disagg"


def _set_required_environment() -> None:
    os.environ.setdefault("HF_HUB_CACHE", DEFAULT_HF_HUB_CACHE)
    os.environ.setdefault("QEFF_HOME", DEFAULT_QEFF_HOME)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


_set_required_environment()

from profiler import QEffMemoryProfiler  # noqa: E402


class DetailedProfiler:
    """QEffMemoryProfiler adapter with scoped phase names."""

    def __init__(self, profiler: QEffMemoryProfiler):
        self.profiler = profiler
        self._component_stack: List[str] = []

    @property
    def current_component(self) -> Optional[str]:
        return self._component_stack[-1] if self._component_stack else None

    def scoped_name(self, operation_name: str) -> str:
        component_name = self.current_component
        return f"{component_name}: {operation_name}" if component_name else operation_name

    @contextmanager
    def component(self, component_name: Optional[str]):
        if component_name is None:
            yield
            return
        self._component_stack.append(component_name)
        try:
            yield
        finally:
            self._component_stack.pop()

    def sample_now(self) -> None:
        sample = self.profiler.metrics_collector.collect_sample()
        self.profiler.samples.append(sample)
        self.profiler._update_peaks(sample)

    def mark(self, operation_name: str) -> None:
        self.sample_now()
        self.profiler.mark_operation(self.scoped_name(operation_name))

    def call(
        self,
        operation_name: str,
        func: Callable[..., Any],
        *args: Any,
        component_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        with self.component(component_name):
            self.mark(operation_name)
            try:
                return func(*args, **kwargs)
            finally:
                self.mark(f"post: {operation_name}")


def _short_command(command: Any) -> str:
    if isinstance(command, (list, tuple)) and command:
        executable = str(command[0])
        if "qaic-compile" in executable:
            return "qaic-compile"
        return Path(executable).name
    if isinstance(command, str):
        first_token = command.split(maxsplit=1)[0] if command.split() else command
        if "qaic-compile" in first_token:
            return "qaic-compile"
        return Path(first_token).name
    return "unknown"


def install_global_hooks(detailed_profiler: DetailedProfiler) -> SimpleNamespace:
    """Patch expensive calls so memory is attributed to the active component."""
    import onnx
    import torch.onnx

    from QEfficient.base.onnx_transforms import OnnxTransformPipeline

    originals = SimpleNamespace(
        onnx_load=onnx.load,
        onnx_save=onnx.save,
        subprocess_run=subprocess.run,
        torch_onnx_export=torch.onnx.export,
        onnx_transform_apply=OnnxTransformPipeline.apply,
    )

    def profiled_onnx_load(*args: Any, **kwargs: Any) -> Any:
        return detailed_profiler.call("onnx.load", originals.onnx_load, *args, **kwargs)

    def profiled_onnx_save(*args: Any, **kwargs: Any) -> Any:
        return detailed_profiler.call("onnx.save", originals.onnx_save, *args, **kwargs)

    def profiled_subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        command = args[0] if args else kwargs.get("args")
        return detailed_profiler.call(
            f"subprocess.run: {_short_command(command)}", originals.subprocess_run, *args, **kwargs
        )

    def profiled_torch_onnx_export(*args: Any, **kwargs: Any) -> Any:
        return detailed_profiler.call("torch.onnx.export", originals.torch_onnx_export, *args, **kwargs)

    def profiled_onnx_transform_apply(self: Any, *args: Any, **kwargs: Any) -> Any:
        transform_names = ",".join(transform.__name__ for transform in getattr(self, "transforms", [])) or "none"
        return detailed_profiler.call(
            f"onnx transforms: {transform_names}", originals.onnx_transform_apply, self, *args, **kwargs
        )

    onnx.load = profiled_onnx_load
    onnx.save = profiled_onnx_save
    subprocess.run = profiled_subprocess_run
    torch.onnx.export = profiled_torch_onnx_export
    OnnxTransformPipeline.apply = profiled_onnx_transform_apply
    return originals


def _wrap_instance_method(
    instance: Any, method_name: str, operation_name: str, detailed_profiler: DetailedProfiler
) -> None:
    if not hasattr(instance, method_name):
        return
    original = getattr(instance, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return detailed_profiler.call(operation_name, original, *args, **kwargs)

    setattr(instance, method_name, wrapped)


def install_model_hooks(qeff_model: Any, detailed_profiler: DetailedProfiler) -> None:
    """Patch model methods while preserving outer vision/prefill/decode scope."""
    _wrap_instance_method(qeff_model, "export", "dual-qpc export orchestration", detailed_profiler)
    _wrap_instance_method(qeff_model, "generate", "qeff generate", detailed_profiler)

    if hasattr(qeff_model, "vision_model"):
        _wrap_instance_method(qeff_model.vision_model, "export", "vision export", detailed_profiler)
        _wrap_instance_method(qeff_model.vision_model, "_compile", "vision compile wrapper", detailed_profiler)
        _wrap_instance_method(qeff_model.vision_model, "transform", "vision pytorch transform", detailed_profiler)
        _wrap_instance_method(
            qeff_model.vision_model,
            "_offload_model_weights",
            "vision offload pytorch weights",
            detailed_profiler,
        )

    if hasattr(qeff_model, "lang_model"):
        _wrap_instance_method(qeff_model.lang_model, "export", "language export", detailed_profiler)
        _wrap_instance_method(qeff_model.lang_model, "_compile", "language compile wrapper", detailed_profiler)
        _wrap_instance_method(qeff_model.lang_model, "transform", "language pytorch transform", detailed_profiler)
        _wrap_instance_method(
            qeff_model.lang_model,
            "_offload_model_weights",
            "language offload pytorch weights",
            detailed_profiler,
        )

    model = getattr(qeff_model, "model", None)
    if model is not None:
        _wrap_instance_method(model, "get_dummy_inputs", "build dummy inputs", detailed_profiler)
        _wrap_instance_method(model, "get_onnx_dynamic_axes", "build dynamic axes", detailed_profiler)
        _wrap_instance_method(model, "get_output_names", "build output names", detailed_profiler)
        _wrap_instance_method(
            model, "prepare_inputs_for_generation", "prepare inputs for generation", detailed_profiler
        )


def _load_target_constants() -> SimpleNamespace:
    spec = importlib.util.spec_from_file_location("qwen3_5_disagg_constants", TARGET_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to inspect target script: {TARGET_SCRIPT}")
    # Avoid importing because the target script executes compile/inference at module scope.
    return SimpleNamespace(
        model_id="Qwen/Qwen3.6-35B-A3B",
        layerwise=False,
        layerwise_window_size=1,
        prefill_seq_len=64,
        ctx_len=4096,
        batch_size=1,
        generation_len=256,
    )


def _update_retained_states(
    target_inputs: Dict[str, Any], source_outputs: Dict[str, Any], layer_types: List[str]
) -> None:
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            target_inputs[f"past_key.{layer_idx}"] = source_outputs[f"past_key.{layer_idx}_RetainedState"]
            target_inputs[f"past_value.{layer_idx}"] = source_outputs[f"past_value.{layer_idx}_RetainedState"]
        else:
            target_inputs[f"conv_state.{layer_idx}"] = source_outputs[f"conv_state.{layer_idx}_RetainedState"]
            target_inputs[f"recurrent_state.{layer_idx}"] = source_outputs[f"recurrent_state.{layer_idx}_RetainedState"]


def _phase_rows(profiler: QEffMemoryProfiler) -> List[Dict[str, Any]]:
    if not profiler.samples or len(profiler.operations) < 2:
        return []
    start_time = profiler.samples[0].timestamp
    rows: List[Dict[str, Any]] = []
    operations = profiler.operations + [(profiler.samples[-1].timestamp, "_end")]

    for index, ((operation_time, operation_name), (next_time, _)) in enumerate(zip(operations, operations[1:])):
        phase_samples = [sample for sample in profiler.samples if operation_time <= sample.timestamp <= next_time]
        if not phase_samples:
            phase_samples = [
                min(profiler.samples, key=lambda sample: abs((sample.timestamp - operation_time).total_seconds()))
            ]
        rss_values = [sample.rss_mb for sample in phase_samples]
        vms_values = [sample.vms_mb for sample in phase_samples]
        cpu_values = [sample.cpu_percent for sample in phase_samples]
        peak_sample = max(phase_samples, key=lambda sample: sample.rss_mb)
        first_sample = phase_samples[0]
        last_sample = phase_samples[-1]
        rows.append(
            {
                "index": index + 1,
                "operation": operation_name,
                "start_s": round((operation_time - start_time).total_seconds(), 3),
                "end_s": round((next_time - start_time).total_seconds(), 3),
                "duration_s": round((next_time - operation_time).total_seconds(), 3),
                "rss_start_mb": round(first_sample.rss_mb, 3),
                "rss_end_mb": round(last_sample.rss_mb, 3),
                "rss_delta_mb": round(last_sample.rss_mb - first_sample.rss_mb, 3),
                "rss_min_mb": round(min(rss_values), 3),
                "rss_avg_mb": round(sum(rss_values) / len(rss_values), 3),
                "rss_peak_mb": round(max(rss_values), 3),
                "rss_peak_s": round((peak_sample.timestamp - start_time).total_seconds(), 3),
                "vms_peak_mb": round(max(vms_values), 3),
                "cpu_avg_percent": round(sum(cpu_values) / len(cpu_values), 3),
                "cpu_peak_percent": round(max(cpu_values), 3),
                "disk_read_delta_mb": round(last_sample.disk_read_mb - first_sample.disk_read_mb, 3),
                "disk_write_delta_mb": round(last_sample.disk_write_mb - first_sample.disk_write_mb, 3),
                "sample_count": len(phase_samples),
            }
        )
    return rows


def _sample_rows(profiler: QEffMemoryProfiler) -> Iterable[Dict[str, Any]]:
    if not profiler.samples:
        return []
    start_time = profiler.samples[0].timestamp
    return [
        {
            "time_s": round((sample.timestamp - start_time).total_seconds(), 3),
            "timestamp": sample.timestamp.isoformat(),
            "rss_mb": round(sample.rss_mb, 3),
            "vms_mb": round(sample.vms_mb, 3),
            "cpu_percent": round(sample.cpu_percent, 3),
            "disk_read_mb": round(sample.disk_read_mb, 3),
            "disk_write_mb": round(sample.disk_write_mb, 3),
            "disk_read_rate_mb_s": round(sample.disk_read_rate, 3),
            "disk_write_rate_mb_s": round(sample.disk_write_rate, 3),
        }
        for sample in profiler.samples
    ]


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(profiler: QEffMemoryProfiler, output_dir: Path, run_metadata: Dict[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "memory_report.txt"
    phase_csv_path = output_dir / "component_phases.csv"
    sample_csv_path = output_dir / "samples_timeseries.csv"
    raw_json_path = output_dir / "profile_summary.json"
    graph_path = output_dir / "memory_timeline.png"

    report_path.write_text(profiler.get_memory_report(), encoding="utf-8")
    phase_rows = _phase_rows(profiler)
    sample_rows = list(_sample_rows(profiler))
    _write_csv(phase_csv_path, phase_rows)
    _write_csv(sample_csv_path, sample_rows)
    profiler.generate_memory_graph(str(graph_path))
    raw_json_path.write_text(
        json.dumps(
            {
                "metadata": run_metadata,
                "peak_rss_mb": profiler.peak_rss,
                "peak_operation": profiler.peak_operation,
                "operations": phase_rows,
                "samples": sample_rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return {
        "report": str(report_path),
        "phases_csv": str(phase_csv_path),
        "samples_csv": str(sample_csv_path),
        "summary_json": str(raw_json_path),
        "graph": str(graph_path),
    }


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
    target = _load_target_constants()

    run_metadata: Dict[str, Any] = {
        "target_script": str(TARGET_SCRIPT),
        "started_at": datetime.now().isoformat(),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "qeff_home": os.environ.get("QEFF_HOME"),
        "hf_hub_enable_hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"),
        "sampling_interval": args.sampling_interval,
        "model_id": args.model_id or target.model_id,
        "layerwise": target.layerwise,
        "layerwise_window_size": target.layerwise_window_size,
        "components": args.components,
        "language_compile_order": args.language_compile_order,
        "text_layers": args.text_layers,
        "vision_depth": args.vision_depth,
    }
    output_paths: Dict[str, str] = {}

    profiler.start_monitoring()
    try:
        install_global_hooks(detailed_profiler)

        import numpy as np
        import requests
        import torch
        import transformers
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoConfig, AutoProcessor

        from QEfficient import QEFFAutoModelForImageTextToText
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        model_id = args.model_id or target.model_id
        detailed_profiler.mark("hf config setup")
        config = detailed_profiler.call("AutoConfig.from_pretrained", AutoConfig.from_pretrained, model_id)
        config.vision_config.depth = args.vision_depth
        config.text_config.num_hidden_layers = args.text_layers
        config.torch_dtype = "float16"
        layer_types = list(getattr(config.text_config, "layer_types", []))
        if len(layer_types) < config.text_config.num_hidden_layers:
            layer_types.extend(["full_attention"] * (config.text_config.num_hidden_layers - len(layer_types)))
        config.text_config.layer_types = layer_types[: config.text_config.num_hidden_layers]
        detailed_profiler.mark("post: hf config setup")

        qeff_model = detailed_profiler.call(
            "QEFFAutoModelForImageTextToText.from_pretrained",
            QEFFAutoModelForImageTextToText.from_pretrained,
            model_id,
            attn_implementation="eager",
            kv_offload=True,
            config=config,
            layerwise=target.layerwise,
        )
        install_model_hooks(qeff_model, detailed_profiler)
        tokenizer = detailed_profiler.call(
            "AutoTokenizer.from_pretrained", transformers.AutoTokenizer.from_pretrained, model_id
        )
        processor = detailed_profiler.call("AutoProcessor.from_pretrained", AutoProcessor.from_pretrained, model_id)

        selected_components = set(args.components)
        compile_errors = {}
        vision_qpc_path = None
        if "vision" in selected_components and not args.skip_vision:
            try:
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
                    num_devices=1,
                    mos=1,
                    mxfp6_matmul=True,
                    aic_enable_depth_first=True,
                    skip_vision=False,
                    split_model_io=True,
                    skip_lang=True,
                    use_onnx_subfunctions=True,
                    layerwise=target.layerwise,
                    layerwise_window_size=target.layerwise_window_size,
                )
            except Exception as exc:
                compile_errors["vision"] = repr(exc)
                if not (args.compile_only and args.continue_on_compile_error):
                    raise

        def compile_decode() -> Any:
            try:
                return detailed_profiler.call(
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
                    mxfp6_matmul=True,
                    mxint8_kv_cache=False,
                    retain_full_kv=True,
                    split_model_io=True,
                    mos=1,
                    aic_enable_depth_first=True,
                    prefill_only=False,
                    skip_vision=True,
                    use_onnx_subfunctions=True,
                    layerwise=target.layerwise,
                    layerwise_window_size=target.layerwise_window_size,
                    offload_pt_weights=False,
                )
            except Exception as exc:
                compile_errors["decode"] = repr(exc)
                if not (args.compile_only and args.continue_on_compile_error):
                    raise
                return None

        def compile_prefill() -> Any:
            try:
                return detailed_profiler.call(
                    "compile request",
                    qeff_model.compile,
                    component_name="prefill",
                    batch_size=args.batch_size,
                    prefill_seq_len=args.prefill_seq_len,
                    ctx_len=args.ctx_len,
                    height=args.height,
                    width=args.width,
                    num_cores=args.num_cores,
                    num_devices=1,
                    mxfp6_matmul=False,
                    mxint8_kv_cache=False,
                    retain_full_kv=True,
                    split_model_io=True,
                    mos=1,
                    user_tiled=True,
                    aic_enable_depth_first=False,
                    prefill_only=True,
                    enable_chunking=True,
                    skip_vision=True,
                    use_onnx_subfunctions=True,
                    layerwise=target.layerwise,
                    layerwise_window_size=target.layerwise_window_size,
                    offload_pt_weights=(args.language_compile_order == "decode-first"),
                )
            except Exception as exc:
                compile_errors["prefill"] = repr(exc)
                if not (args.compile_only and args.continue_on_compile_error):
                    raise
                return None

        decode_qpc_path = None
        prefill_qpc_path = None
        if args.language_compile_order == "decode-first":
            if "decode" in selected_components:
                decode_qpc_path = compile_decode()
            if "prefill" in selected_components:
                prefill_qpc_path = compile_prefill()
        else:
            if "prefill" in selected_components:
                prefill_qpc_path = compile_prefill()
            if "decode" in selected_components:
                decode_qpc_path = compile_decode()

        if args.compile_only:
            run_metadata.update(
                {
                    "vision_qpc_path": str(vision_qpc_path),
                    "prefill_qpc_path": str(prefill_qpc_path),
                    "decode_qpc_path": str(decode_qpc_path),
                    "compile_errors": compile_errors,
                    "completed_at": datetime.now().isoformat(),
                    "status": "failed" if compile_errors else "success",
                    "mode": "compile_only",
                }
            )
            return {"metadata": run_metadata, "outputs": output_paths}

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

        if args.skip_vision:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Tell me about yourself."}],
                }
            ]
            vision_session = None
        else:
            image = detailed_profiler.call(
                "download/open image",
                lambda: Image.open(requests.get(args.image_url, stream=True, timeout=30).raw),
                component_name="runtime",
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe all the colors seen in the image."},
                    ],
                }
            ]
            vision_session = detailed_profiler.call(
                "QAICInferenceSession: vision",
                QAICInferenceSession,
                vision_qpc_path.get("vision_qpc_path"),
                component_name="runtime",
            )

        messages = [messages] * args.batch_size
        texts = detailed_profiler.call(
            "processor.apply_chat_template",
            lambda: [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages
            ],
            component_name="runtime",
        )
        image_inputs, video_inputs = detailed_profiler.call(
            "process_vision_info", process_vision_info, messages, component_name="runtime"
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
        vision_outputs = {}
        if vision_inputs and vision_session is not None:
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
        if not args.skip_vision:
            lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]

        all_outputs = []
        chunk_inputs = lang_inputs.copy()
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
            _update_retained_states(chunk_inputs, outputs, config.text_config.layer_types)
            chunk_inputs["image_idx"] = outputs["image_idx_output"]

        all_outputs.append(np.argmax(outputs["logits"]))
        decode_inputs = {
            "input_ids": np.argmax(outputs["logits"]).reshape(1, 1),
            "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
        }
        _update_retained_states(decode_inputs, outputs, config.text_config.layer_types)
        decode_inputs["image_idx"] = outputs["image_idx_output"]
        if not args.skip_vision:
            decode_inputs["vision_embeds"] = outputs["vision_embeds_RetainedState"]

        decode_out = detailed_profiler.call(
            "lang_decode_session.run first token", lang_decode_session.run, decode_inputs, component_name="runtime"
        )
        all_outputs.append(np.argmax(decode_out["logits"]))
        pos_id = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
        loop_decode_inputs = {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
        _update_retained_states(loop_decode_inputs, decode_out, config.text_config.layer_types)
        loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]
        if not args.skip_vision:
            loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]

        for token_idx in range(args.generation_len - 2):
            decode_out = detailed_profiler.call(
                f"lang_decode_session.run token {token_idx + 2}",
                lang_decode_session.run,
                loop_decode_inputs,
                component_name="runtime",
            )
            all_outputs.append(np.argmax(decode_out["logits"]))
            pos_id += 1
            _update_retained_states(loop_decode_inputs, decode_out, config.text_config.layer_types)
            loop_decode_inputs.update(
                {
                    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
                    "position_ids": pos_id,
                }
            )

        decoded_output = detailed_profiler.call(
            "tokenizer.decode", tokenizer.decode, all_outputs, component_name="runtime"
        )
        run_metadata.update(
            {
                "vision_qpc_path": str(vision_qpc_path),
                "prefill_qpc_path": str(prefill_qpc_path),
                "decode_qpc_path": str(decode_qpc_path),
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
    run_id = datetime.now().strftime("qwen3_5_disagg_mode_%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=run_id)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--sampling-interval", type=float, default=0.1)
    parser.add_argument("--child-scan-interval", type=float, default=0.2)
    parser.add_argument("--vision-depth", type=int, default=4)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--prefill-seq-len", type=int, default=64)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--decode-num-devices", type=int, default=int(os.environ.get("QEFF_DECODE_NUM_DEVICES", "1")))
    parser.add_argument("--height", type=int, default=354)
    parser.add_argument("--width", type=int, default=536)
    parser.add_argument("--generation-len", type=int, default=256)
    parser.add_argument("--image-url", default="https://picsum.photos/id/237/536/354")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("vision", "prefill", "decode"),
        default=["vision", "prefill", "decode"],
    )
    parser.add_argument("--language-compile-order", choices=("prefill-first", "decode-first"), default="prefill-first")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--continue-on-compile-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    start_monotonic = time.monotonic()
    try:
        run_profile(parse_args())
    finally:
        elapsed = time.monotonic() - start_monotonic
        print(f"Detailed disagg profiling elapsed time: {elapsed:.1f}s")
