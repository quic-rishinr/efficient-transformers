# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Detailed memory profiling harness for qwen3_5_moe_layerwise_decode.py.

This intentionally keeps the example script unchanged while running the same
workflow with finer-grained operation markers around model loading, export,
compile, input preparation, and generation internals.
"""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SCRIPT = REPO_ROOT / "examples/image_text_to_text/models/qwen3_5_moe/qwen3_5_moe_layerwise_decode.py"
DEFAULT_HF_HUB_CACHE = "/home/huggingface_hub"
DEFAULT_QEFF_HOME = "/home/rishinr/qwen3vl_memory"


def _set_required_environment() -> None:
    os.environ.setdefault("HF_HUB_CACHE", DEFAULT_HF_HUB_CACHE)
    os.environ.setdefault("QEFF_HOME", DEFAULT_QEFF_HOME)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


_set_required_environment()

from profiler import QEffMemoryProfiler  # noqa: E402


class DetailedProfiler:
    """Small adapter over QEffMemoryProfiler with exact component summaries."""

    def __init__(self, profiler: QEffMemoryProfiler):
        self.profiler = profiler
        self._component_stack: List[str] = []

    @property
    def current_component(self) -> Optional[str]:
        return self._component_stack[-1] if self._component_stack else None

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
        self.profiler.mark_operation(operation_name)

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

    def scoped_name(self, operation_name: str) -> str:
        component_name = self.current_component
        return f"{component_name}: {operation_name}" if component_name else operation_name


def _load_target_module() -> Any:
    spec = importlib.util.spec_from_file_location("qwen3_5_moe_layerwise_decode_target", TARGET_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load target script: {TARGET_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    """Patch expensive library calls so their memory is visible as phases."""
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
        return detailed_profiler.call(detailed_profiler.scoped_name("onnx.load"), originals.onnx_load, *args, **kwargs)

    def profiled_onnx_save(*args: Any, **kwargs: Any) -> Any:
        return detailed_profiler.call(detailed_profiler.scoped_name("onnx.save"), originals.onnx_save, *args, **kwargs)

    def profiled_subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        command = args[0] if args else kwargs.get("args")
        operation_name = detailed_profiler.scoped_name(f"subprocess.run: {_short_command(command)}")
        return detailed_profiler.call(operation_name, originals.subprocess_run, *args, **kwargs)

    def profiled_torch_onnx_export(*args: Any, **kwargs: Any) -> Any:
        return detailed_profiler.call(
            detailed_profiler.scoped_name("torch.onnx.export"), originals.torch_onnx_export, *args, **kwargs
        )

    def profiled_onnx_transform_apply(self: Any, *args: Any, **kwargs: Any) -> Any:
        transform_names = ",".join(transform.__name__ for transform in getattr(self, "transforms", [])) or "none"
        operation_name = detailed_profiler.scoped_name(f"onnx transforms: {transform_names}")
        return detailed_profiler.call(operation_name, originals.onnx_transform_apply, self, *args, **kwargs)

    onnx.load = profiled_onnx_load
    onnx.save = profiled_onnx_save
    subprocess.run = profiled_subprocess_run
    torch.onnx.export = profiled_torch_onnx_export
    OnnxTransformPipeline.apply = profiled_onnx_transform_apply
    return originals


def _wrap_instance_method(
    instance: Any,
    method_name: str,
    operation_name: str,
    detailed_profiler: DetailedProfiler,
    component_name: Optional[str] = None,
) -> None:
    if not hasattr(instance, method_name):
        return
    original = getattr(instance, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return detailed_profiler.call(operation_name, original, *args, component_name=component_name, **kwargs)

    setattr(instance, method_name, wrapped)


def install_model_hooks(qeff_model: Any, detailed_profiler: DetailedProfiler) -> None:
    """Patch the loaded QEff model instance to expose component-level phases."""
    _wrap_instance_method(qeff_model, "export", "dual-qpc export orchestration", detailed_profiler, "dual-qpc")
    _wrap_instance_method(qeff_model, "generate", "qeff generate", detailed_profiler, "runtime")

    if hasattr(qeff_model, "vision_model"):
        _wrap_instance_method(qeff_model.vision_model, "export", "vision export", detailed_profiler, "vision")
        _wrap_instance_method(
            qeff_model.vision_model, "_compile", "vision compile wrapper", detailed_profiler, "vision"
        )
        _wrap_instance_method(
            qeff_model.vision_model, "transform", "vision pytorch transform", detailed_profiler, "vision"
        )
        _wrap_instance_method(
            qeff_model.vision_model,
            "_offload_model_weights",
            "vision offload pytorch weights",
            detailed_profiler,
            "vision",
        )

    if hasattr(qeff_model, "lang_model"):
        _wrap_instance_method(qeff_model.lang_model, "export", "language export", detailed_profiler, "language")
        _wrap_instance_method(
            qeff_model.lang_model, "_compile", "language compile wrapper", detailed_profiler, "language"
        )
        _wrap_instance_method(
            qeff_model.lang_model, "transform", "language pytorch transform", detailed_profiler, "language"
        )
        _wrap_instance_method(
            qeff_model.lang_model,
            "_offload_model_weights",
            "language offload pytorch weights",
            detailed_profiler,
            "language",
        )

    model = getattr(qeff_model, "model", None)
    if model is not None:
        _wrap_instance_method(model, "get_dummy_inputs", "build dummy inputs", detailed_profiler, "dual-qpc")
        _wrap_instance_method(model, "get_onnx_dynamic_axes", "build dynamic axes", detailed_profiler, "dual-qpc")
        _wrap_instance_method(model, "get_output_names", "build output names", detailed_profiler, "dual-qpc")
        _wrap_instance_method(
            model, "prepare_inputs_for_generation", "prepare inputs for generation", detailed_profiler
        )


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
    rows = []
    for sample in profiler.samples:
        rows.append(
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
        )
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    profiler: QEffMemoryProfiler,
    output_dir: Path,
    run_metadata: Dict[str, Any],
    graph_filename: str = "memory_timeline.png",
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "memory_report.txt"
    phase_csv_path = output_dir / "component_phases.csv"
    sample_csv_path = output_dir / "samples_timeseries.csv"
    raw_json_path = output_dir / "profile_summary.json"
    graph_path = output_dir / graph_filename

    report = profiler.get_memory_report()
    report_path.write_text(report, encoding="utf-8")
    _write_csv(phase_csv_path, _phase_rows(profiler))
    _write_csv(sample_csv_path, _sample_rows(profiler))
    profiler.generate_memory_graph(str(graph_path))

    raw_json_path.write_text(
        json.dumps(
            {
                "metadata": run_metadata,
                "peak_rss_mb": profiler.peak_rss,
                "peak_operation": profiler.peak_operation,
                "operations": _phase_rows(profiler),
                "samples": list(_sample_rows(profiler)),
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

    run_metadata: Dict[str, Any] = {
        "target_script": str(TARGET_SCRIPT),
        "started_at": datetime.now().isoformat(),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "qeff_home": os.environ.get("QEFF_HOME"),
        "hf_hub_enable_hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"),
        "sampling_interval": args.sampling_interval,
    }
    qpc_path = None
    generated_ids = None
    decoded_output = None
    output_paths: Dict[str, str] = {}

    profiler.start_monitoring()
    try:
        detailed_profiler.mark("imports: target example module")
        target = _load_target_module()
        detailed_profiler.mark("post: imports: target example module")

        install_global_hooks(detailed_profiler)

        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor

        from QEfficient import QEFFAutoModelForImageTextToText

        model_id = args.model_id or target.MODEL_ID
        run_metadata.update(
            {
                "model_id": model_id,
                "layerwise": target.LAYERWISE,
                "torch_dtype": str(target.TORCH_DTYPE),
                "random_seed": target.RANDOM_SEED,
            }
        )

        detailed_profiler.call("setup: torch.manual_seed", torch.manual_seed, target.RANDOM_SEED)

        config = detailed_profiler.call("hf config: AutoConfig.from_pretrained", AutoConfig.from_pretrained, model_id)
        detailed_profiler.mark("config: mutate dtype/depth/layers")
        config.torch_dtype = target.TORCH_DTYPE
        config.vision_config.depth = args.vision_depth
        config.text_config.num_hidden_layers = args.text_layers
        detailed_profiler.mark("post: config: mutate dtype/depth/layers")

        tokenizer = detailed_profiler.call(
            "hf tokenizer: AutoTokenizer.from_pretrained", transformers.AutoTokenizer.from_pretrained, model_id
        )
        processor = detailed_profiler.call(
            "hf processor: AutoProcessor.from_pretrained", AutoProcessor.from_pretrained, model_id
        )

        qeff_model = detailed_profiler.call(
            "qeff load: QEFFAutoModelForImageTextToText.from_pretrained",
            QEFFAutoModelForImageTextToText.from_pretrained,
            model_id,
            attn_implementation="eager",
            kv_offload=True,
            config=config,
            dtype=target.TORCH_DTYPE,
            layerwise=target.LAYERWISE,
        )
        install_model_hooks(qeff_model, detailed_profiler)

        detailed_profiler.mark("compile orchestration: qeff_model.compile")
        qpc_path = qeff_model.compile(
            batch_size=1,
            prefill_seq_len=1,
            ctx_len=args.ctx_len,
            num_cores=args.num_cores,
            num_devices=args.num_devices,
            height=args.height,
            width=args.width,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            skip_vision=True,
            split_retained_state_io=True,
            use_onnx_subfunctions=True,
            mos=1,
            layerwise=target.LAYERWISE,
            layerwise_window_size=1,
        )
        detailed_profiler.mark("post: compile orchestration: qeff_model.compile")

        batch_size = 1
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]
        messages = [messages] * batch_size

        inputs = detailed_profiler.call(
            "processor: apply_chat_template",
            processor.apply_chat_template,
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = detailed_profiler.call(
            "model: prepare_inputs_for_generation",
            qeff_model.model.prepare_inputs_for_generation,
            inputs=inputs,
            prefill_seq_len=args.runtime_prefill_seq_len,
            batch_size=batch_size,
        )
        output = detailed_profiler.call(
            "qeff_model.generate", qeff_model.generate, inputs=inputs, generation_len=args.generation_len
        )
        generated_ids = (
            output.generated_ids.tolist() if hasattr(output.generated_ids, "tolist") else str(output.generated_ids)
        )
        decoded_output = detailed_profiler.call(
            "decode: tokenizer.batch_decode", tokenizer.batch_decode, output.generated_ids
        )

        run_metadata.update(
            {
                "qpc_path": str(qpc_path),
                "generated_ids": generated_ids,
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
                "qpc_path": str(qpc_path),
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
    run_id = datetime.now().strftime("qwen3_5_moe_layerwise_decode_%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=run_id)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--sampling-interval", type=float, default=0.1)
    parser.add_argument("--child-scan-interval", type=float, default=0.2)
    parser.add_argument("--vision-depth", type=int, default=4)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--height", type=int, default=354)
    parser.add_argument("--width", type=int, default=536)
    parser.add_argument("--runtime-prefill-seq-len", type=int, default=128)
    parser.add_argument("--generation-len", type=int, default=100)
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    start_monotonic = time.monotonic()
    try:
        result = run_profile(parse_args())
    finally:
        elapsed = time.monotonic() - start_monotonic
        print(f"Detailed profiling elapsed time: {elapsed:.1f}s")
