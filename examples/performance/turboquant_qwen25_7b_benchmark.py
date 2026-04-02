# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Qwen2.5-7B compression benchmark with TurboQuant-style emulation.

This script compares runtime/compile variants on QAIC:
1) `mxfp6` (`mxfp6_matmul=True`)
2) `mxint8` (`mxint8_kv_cache=True`)
3) `mxfp6_mxint8` (both flags enabled)

It also includes a software-only TurboQuant-style KV emulation report:
- random rotation + scalar quantization
- 1-bit QJL residual signs

Note:
- The emulation section is algorithmic only; it does not imply dedicated
  hardware TurboQuant kernels in qaic-compile.
"""

import argparse
import json
import os
import traceback
from datetime import datetime
from math import sqrt
from time import perf_counter
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.turboquant import TurboQuantConfig, TurboQuantizer
from QEfficient.utils import load_hf_tokenizer


def build_prompt_of_token_length(tokenizer, target_tokens: int) -> str:
    seed_text = "TurboQuant prototype benchmark prompt for compression efficiency. "
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("Tokenizer returned empty tokenization for seed text.")
    repeated_ids = (seed_ids * ((target_tokens // len(seed_ids)) + 2))[:target_tokens]
    return tokenizer.decode(repeated_ids, skip_special_tokens=True)


def get_qpc_size_bytes(qpc_path: str) -> int:
    total = 0
    for root, _, files in os.walk(qpc_path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def collect_exec_metrics(exec_info) -> Dict[str, float]:
    perf = exec_info.perf_metrics
    return {
        "prefill_time_sec": float(perf.prefill_time),
        "decode_tok_per_sec_per_batch": float(perf.decode_perf),
        "total_tok_per_sec_per_batch": float(perf.total_perf),
        "e2e_time_sec": float(perf.total_time),
        "decode_tok_per_sec_aggregate": float(perf.decode_perf * exec_info.batch_size),
        "total_tok_per_sec_aggregate": float(perf.total_perf * exec_info.batch_size),
    }


def run_variant(
    model,
    tokenizer,
    variant_name: str,
    compile_kwargs: Dict,
    prompts: List[str],
    generation_len: int,
    device_id: int,
) -> Dict:
    result = {
        "variant": variant_name,
        "compile_kwargs": compile_kwargs,
        "compile_success": False,
        "run_success": False,
    }

    t0 = perf_counter()
    try:
        qpc_path = model.compile(**compile_kwargs)
        t1 = perf_counter()
        result["compile_wall_time_sec"] = t1 - t0
        result["compile_success"] = True
        result["compile_time_sec"] = t1 - t0
        result["qpc_path"] = str(qpc_path)
        result["qpc_size_bytes"] = get_qpc_size_bytes(str(qpc_path))
    except Exception as exc:
        t1 = perf_counter()
        result["compile_wall_time_sec"] = t1 - t0
        result["compile_error"] = f"{type(exc).__name__}: {exc}"
        result["compile_traceback"] = traceback.format_exc(limit=20)
        return result

    try:
        t0 = perf_counter()
        exec_info = model.generate(
            tokenizer=tokenizer,
            prompts=prompts,
            generation_len=generation_len,
            device_id=[device_id],
        )
        t1 = perf_counter()
        result["run_success"] = True
        result["run_wall_time_sec"] = t1 - t0
        result["run_metrics"] = collect_exec_metrics(exec_info)
    except Exception as exc:
        result["run_error"] = f"{type(exc).__name__}: {exc}"
        result["run_traceback"] = traceback.format_exc(limit=20)

    return result


def _safe_head_dim(model_cfg) -> int:
    head_dim = getattr(model_cfg, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    hidden_size = int(getattr(model_cfg, "hidden_size"))
    num_attention_heads = int(getattr(model_cfg, "num_attention_heads"))
    return hidden_size // num_attention_heads


def _randn(shape, gen: torch.Generator, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.randn(shape, generator=gen, dtype=dtype).to(device)


def run_turboquant_emulation(
    model_name: str,
    batch_size: int,
    ctx_len: int,
    max_layers: int,
    query_len: int,
    bits: float,
    seed: int,
    device: str,
) -> Dict:
    t0 = perf_counter()
    model_cfg = AutoConfig.from_pretrained(model_name)
    head_dim = _safe_head_dim(model_cfg)
    num_kv_heads = int(getattr(model_cfg, "num_key_value_heads", getattr(model_cfg, "num_attention_heads", 1)))
    num_layers = int(getattr(model_cfg, "num_hidden_layers", 1))
    eval_layers = max(1, min(max_layers, num_layers))
    eval_query_len = max(1, min(query_len, ctx_len))

    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        run_device = torch.device("cpu")

    quantizer = TurboQuantizer(TurboQuantConfig(bits=bits, seed=seed))

    total_original_bytes = 0
    total_estimated_bytes = 0
    quant_time_sec = 0.0
    dequant_time_sec = 0.0
    layer_metrics = []

    for layer_idx in range(eval_layers):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 97 * (layer_idx + 1))

        kv_shape = (batch_size, num_kv_heads, ctx_len, head_dim)
        k = _randn(kv_shape, gen=gen, dtype=torch.float16, device=run_device)
        v = _randn(kv_shape, gen=gen, dtype=torch.float16, device=run_device)
        q = _randn((batch_size, num_kv_heads, eval_query_len, head_dim), gen=gen, dtype=torch.float16, device=run_device)

        qt0 = perf_counter()
        packed = quantizer.quantize_kv_cache(key=k, value=v)
        qt1 = perf_counter()
        quant_time_sec += qt1 - qt0

        dqt0 = perf_counter()
        k_rec, v_rec = quantizer.dequantize_kv_cache(packed, output_dtype=torch.float16)
        dqt1 = perf_counter()
        dequant_time_sec += dqt1 - dqt0

        original_bytes = (k.numel() + v.numel()) * k.element_size()
        estimated_bytes = packed["k"].estimated_compressed_bytes() + packed["v"].estimated_compressed_bytes()
        total_original_bytes += int(original_bytes)
        total_estimated_bytes += int(estimated_bytes)

        k_mse = float(torch.mean((k.float() - k_rec.float()) ** 2).item())
        v_mse = float(torch.mean((v.float() - v_rec.float()) ** 2).item())
        k_mae = float(torch.mean(torch.abs(k.float() - k_rec.float())).item())
        v_mae = float(torch.mean(torch.abs(v.float() - v_rec.float())).item())

        logits_ref = torch.matmul(q.float(), k.float().transpose(-1, -2)) / sqrt(head_dim)
        logits_rec = torch.matmul(q.float(), k_rec.float().transpose(-1, -2)) / sqrt(head_dim)
        logits_mse = float(torch.mean((logits_ref - logits_rec) ** 2).item())
        logits_mae = float(torch.mean(torch.abs(logits_ref - logits_rec)).item())
        logits_cos = float(
            F.cosine_similarity(logits_ref.reshape(1, -1), logits_rec.reshape(1, -1), dim=-1).item()
        )

        layer_metrics.append(
            {
                "layer_idx": layer_idx,
                "k_mse": k_mse,
                "v_mse": v_mse,
                "k_mae": k_mae,
                "v_mae": v_mae,
                "logits_mse": logits_mse,
                "logits_mae": logits_mae,
                "logits_cosine": logits_cos,
                "estimated_compression_ratio": float(original_bytes / max(estimated_bytes, 1)),
            }
        )

    t1 = perf_counter()
    avg_k_mse = float(sum(m["k_mse"] for m in layer_metrics) / len(layer_metrics))
    avg_v_mse = float(sum(m["v_mse"] for m in layer_metrics) / len(layer_metrics))
    avg_logits_mse = float(sum(m["logits_mse"] for m in layer_metrics) / len(layer_metrics))
    avg_logits_cos = float(sum(m["logits_cosine"] for m in layer_metrics) / len(layer_metrics))

    return {
        "enabled": True,
        "note": "Software emulation of TurboQuant-style KV compression (not QAIC kernel path).",
        "model_head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "num_hidden_layers_total": num_layers,
        "num_layers_evaluated": eval_layers,
        "ctx_len": ctx_len,
        "query_len": eval_query_len,
        "batch_size": batch_size,
        "device": str(run_device),
        "turboquant_config": {
            "bits": bits,
            "seed": seed,
            "stats_dtype": "float16",
        },
        "timing": {
            "quant_time_sec_total": quant_time_sec,
            "dequant_time_sec_total": dequant_time_sec,
            "wall_time_sec_total": t1 - t0,
        },
        "compression": {
            "original_bytes_total": total_original_bytes,
            "estimated_compressed_bytes_total": total_estimated_bytes,
            "estimated_ratio_total": float(total_original_bytes / max(total_estimated_bytes, 1)),
        },
        "quality_avg": {
            "k_mse": avg_k_mse,
            "v_mse": avg_v_mse,
            "logits_mse": avg_logits_mse,
            "logits_cosine": avg_logits_cos,
        },
        "quality_per_layer": layer_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B compression benchmark prototype")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt-len", type=int, default=1024, help="Prompt token length")
    parser.add_argument("--ctx-len", type=int, default=1024, help="Compile context length")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--generation-len", type=int, default=8)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--export-dir",
        type=str,
        default="/dev/shm/qeff_turboquant_qwen25_7b",
        help="Directory used for ONNX export and compiled artifacts",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="turboquant_qwen25_7b_benchmark_results.json",
        help="Where to write benchmark results",
    )
    parser.add_argument(
        "--skip-turboquant-emulation",
        action="store_true",
        help="Skip software TurboQuant emulation section.",
    )
    parser.add_argument(
        "--turboquant-bits",
        type=float,
        default=3.0,
        help="Total bit budget for TurboQuant emulation (default: 3.0).",
    )
    parser.add_argument(
        "--turboquant-max-layers",
        type=int,
        default=4,
        help="How many model layers to sample for emulation metrics.",
    )
    parser.add_argument(
        "--turboquant-query-len",
        type=int,
        default=64,
        help="Query length used for attention logit quality checks.",
    )
    parser.add_argument(
        "--turboquant-seed",
        type=int,
        default=2026,
        help="Random seed used for TurboQuant emulation.",
    )
    parser.add_argument(
        "--turboquant-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for TurboQuant emulation.",
    )
    args = parser.parse_args()

    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=args.model_name)
    prompt = build_prompt_of_token_length(tokenizer, args.prompt_len)
    prompts = [prompt] * args.batch_size

    model = QEFFAutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        continuous_batching=False,
    )

    os.makedirs(args.export_dir, exist_ok=True)
    export_start = perf_counter()
    onnx_path = model.export(
        export_dir=args.export_dir,
        use_onnx_subfunctions=True,
        offload_pt_weights=True,
    )
    export_end = perf_counter()

    base_compile = {
        "onnx_path": str(onnx_path),
        "prefill_seq_len": args.prompt_len,
        "ctx_len": args.ctx_len,
        "batch_size": args.batch_size,
        "num_cores": args.num_cores,
        "num_devices": args.num_devices,
        "mxint8_kv_cache": False,
        "use_onnx_subfunctions": True,
    }

    variants = [
        (
            "mxfp6",
            {
                **base_compile,
                "mxfp6_matmul": True,
                "mxint8_kv_cache": False,
            },
        ),
        (
            "mxint8",
            {
                **base_compile,
                "mxfp6_matmul": False,
                "mxint8_kv_cache": True,
            },
        ),
        (
            "mxfp6_mxint8",
            {
                **base_compile,
                "mxfp6_matmul": True,
                "mxint8_kv_cache": True,
            },
        ),
    ]

    report = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "model_name": args.model_name,
        "prompt_len": args.prompt_len,
        "ctx_len": args.ctx_len,
        "batch_size": args.batch_size,
        "generation_len": args.generation_len,
        "num_cores": args.num_cores,
        "num_devices": args.num_devices,
        "device_id": args.device_id,
        "export_dir": args.export_dir,
        "onnx_path": str(onnx_path),
        "export_time_sec": export_end - export_start,
        "results": [],
        "turboquant_emulation": {"enabled": False},
    }

    for variant_name, compile_kwargs in variants:
        compile_kwargs["compile_dir"] = os.path.join(args.export_dir, f"compile_{variant_name}")
        variant_result = run_variant(
            model=model,
            tokenizer=tokenizer,
            variant_name=variant_name,
            compile_kwargs=compile_kwargs,
            prompts=prompts,
            generation_len=args.generation_len,
            device_id=args.device_id,
        )
        report["results"].append(variant_result)

    if not args.skip_turboquant_emulation:
        report["turboquant_emulation"] = run_turboquant_emulation(
            model_name=args.model_name,
            batch_size=args.batch_size,
            ctx_len=args.ctx_len,
            max_layers=args.turboquant_max_layers,
            query_len=args.turboquant_query_len,
            bits=args.turboquant_bits,
            seed=args.turboquant_seed,
            device=args.turboquant_device,
        )

    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote benchmark report to: {args.output_json}")


if __name__ == "__main__":
    main()
