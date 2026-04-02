# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
TurboQuant-style KV-cache compression prototype.

This module emulates the two-stage algorithmic structure described by
Google TurboQuant / PolarQuant + QJL:
1) random rotation + scalar quantization (MSE-oriented stage),
2) 1-bit quantized JL correction on the residual (inner-product stage).

Important:
- This is a research prototype for emulation/analysis.
- It does not change qaic-compile behavior by itself.
- Bit packing is reported as estimated compressed size (software estimate).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, pi, prod, sqrt
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class TurboQuantConfig:
    bits: float = 3.0
    seed: int = 2026
    eps: float = 1e-6
    stats_dtype: torch.dtype = torch.float16
    max_supported_bits: float = 8.0

    def __post_init__(self):
        if self.bits < 1.0 or self.bits > self.max_supported_bits:
            raise ValueError(f"bits must be in [1.0, {self.max_supported_bits}]")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.stats_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("stats_dtype must be float16, bfloat16, or float32")


@dataclass
class TurboQuantTensor:
    idx: torch.Tensor
    x_norm: torch.Tensor
    qjl_sign: torch.Tensor
    gamma: torch.Tensor
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    bits: float
    mse_bits_low: int
    mse_bits_frac: float
    centroids_count: int
    head_dim: int

    def estimated_compressed_bytes(self) -> int:
        index_bits_per_value = max(int((self.centroids_count - 1).bit_length()), 1)
        index_bits_total = self.idx.numel() * index_bits_per_value
        qjl_bits_total = self.qjl_sign.numel()
        stats_bytes = (
            self.x_norm.numel() * self.x_norm.element_size()
            + self.gamma.numel() * self.gamma.element_size()
        )
        return ((index_bits_total + qjl_bits_total + 7) // 8) + stats_bytes

    def original_bytes(self) -> int:
        element_size = torch.tensor([], dtype=self.original_dtype).element_size()
        return prod(self.original_shape) * element_size

    def estimated_compression_ratio(self) -> float:
        return self.original_bytes() / max(self.estimated_compressed_bytes(), 1)


class TurboQuantizer:
    """
    TurboQuant-style quantizer for tensors where the last dimension is head_dim.
    """

    def __init__(self, config: Optional[TurboQuantConfig] = None):
        self.config = config or TurboQuantConfig()
        self._state_cache: Dict[Tuple[int, torch.device], Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _levels_from_mse_bits(mse_bits: int) -> int:
        return 1 if mse_bits <= 0 else 2**mse_bits

    def _build_centroids(self, head_dim: int, mse_bits: int) -> torch.Tensor:
        if mse_bits <= 0:
            return torch.tensor([0.0], dtype=torch.float32)

        dim = float(head_dim)
        if mse_bits == 1:
            val = sqrt(2.0 / pi) / sqrt(dim)
            return torch.tensor([-val, val], dtype=torch.float32)

        if mse_bits == 2:
            v1 = 0.453 / sqrt(dim)
            v2 = 1.51 / sqrt(dim)
            return torch.tensor([-v2, -v1, v1, v2], dtype=torch.float32)

        levels = 2**mse_bits
        grid_n = 4097
        x_grid = torch.linspace(-1.0, 1.0, grid_n, dtype=torch.float64)
        exp = (dim - 3.0) / 2.0
        density = torch.pow(torch.clamp(1.0 - (x_grid * x_grid), min=0.0), exp)

        centroids = torch.linspace(-1.0, 1.0, levels, dtype=torch.float64)
        for _ in range(40):
            mids = 0.5 * (centroids[:-1] + centroids[1:])
            bounds = torch.cat(
                [torch.tensor([-1.0], dtype=torch.float64), mids, torch.tensor([1.0], dtype=torch.float64)]
            )
            updated = torch.empty_like(centroids)
            for idx in range(levels):
                lo = bounds[idx].item()
                hi = bounds[idx + 1].item()
                if idx == 0:
                    mask = (x_grid >= lo) & (x_grid <= hi)
                else:
                    mask = (x_grid > lo) & (x_grid <= hi)
                denom = density[mask].sum()
                if float(denom) <= 0.0:
                    updated[idx] = centroids[idx]
                else:
                    updated[idx] = (x_grid[mask] * density[mask]).sum() / denom
            delta = torch.max(torch.abs(updated - centroids)).item()
            centroids = updated
            if delta < 1e-8:
                break

        return centroids.to(torch.float32)

    def _build_state(self, head_dim: int, device: torch.device) -> Dict[str, torch.Tensor]:
        key = (head_dim, device)
        if key in self._state_cache:
            return self._state_cache[key]

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.config.seed + head_dim * 31)

        random_matrix = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
        rotation, _ = torch.linalg.qr(random_matrix)
        qjl_matrix = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)

        mse_bits = self.config.bits - 1.0
        mse_bits_low = int(floor(mse_bits + 1e-12))
        mse_bits_frac = float(mse_bits - float(mse_bits_low))
        if mse_bits_frac < 1e-8:
            mse_bits_frac = 0.0

        centroids_low = self._build_centroids(head_dim=head_dim, mse_bits=mse_bits_low)
        centroids_high = None
        low_levels = self._levels_from_mse_bits(mse_bits_low)

        if mse_bits_frac > 0.0:
            centroids_high = self._build_centroids(head_dim=head_dim, mse_bits=mse_bits_low + 1)
            centroids = torch.cat([centroids_low, centroids_high], dim=0)
            high_levels = self._levels_from_mse_bits(mse_bits_low + 1)
        else:
            centroids = centroids_low
            high_levels = 0

        state = {
            "rotation": rotation.to(device=device),
            "qjl_matrix": qjl_matrix.to(device=device),
            "centroids": centroids.to(device=device),
            "low_levels": torch.tensor(low_levels, device=device),
            "high_levels": torch.tensor(high_levels, device=device),
            "mse_bits_low": torch.tensor(mse_bits_low, device=device),
            "mse_bits_frac": torch.tensor(mse_bits_frac, device=device),
        }
        self._state_cache[key] = state
        return state

    def _quantmse(
        self,
        y: torch.Tensor,
        centroids: torch.Tensor,
        low_levels: int,
        mse_bits_frac: float,
    ) -> torch.Tensor:
        if mse_bits_frac <= 0.0:
            dist = torch.abs(y.unsqueeze(-1) - centroids.view(1, 1, -1))
            return torch.argmin(dist, dim=-1)

        centroids_low = centroids[:low_levels]
        centroids_high = centroids[low_levels:]

        dist_low = torch.abs(y.unsqueeze(-1) - centroids_low.view(1, 1, -1))
        idx_low = torch.argmin(dist_low, dim=-1)

        dist_high = torch.abs(y.unsqueeze(-1) - centroids_high.view(1, 1, -1))
        idx_high = torch.argmin(dist_high, dim=-1)

        outlier_channels = max(0, min(y.shape[-1], int(round(mse_bits_frac * y.shape[-1]))))
        if outlier_channels == 0:
            return idx_low

        topk = torch.topk(torch.abs(y), outlier_channels, dim=-1, largest=True, sorted=False).indices
        mask = torch.zeros_like(idx_low, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk, value=True)

        idx = idx_low.clone()
        idx[mask] = idx_high[mask] + low_levels
        return idx

    def quantize(self, x: torch.Tensor) -> TurboQuantTensor:
        if x.ndim < 1:
            raise ValueError("input tensor must have at least 1 dimension")
        if not x.is_floating_point():
            raise ValueError("input tensor must be floating point")

        original_shape = tuple(x.shape)
        original_dtype = x.dtype
        head_dim = x.shape[-1]
        state = self._build_state(head_dim=head_dim, device=x.device)

        flat = x.reshape(-1, head_dim).to(torch.float32)
        x_norm = torch.linalg.norm(flat, dim=-1, keepdim=True).clamp(min=self.config.eps)
        x_unit = flat / x_norm

        rotation = state["rotation"].to(torch.float32)
        qjl_matrix = state["qjl_matrix"].to(torch.float32)
        centroids = state["centroids"].to(torch.float32)

        y = x_unit @ rotation.T
        low_levels = int(state["low_levels"].item())
        mse_bits_frac = float(state["mse_bits_frac"].item())
        idx = self._quantmse(
            y=y,
            centroids=centroids,
            low_levels=low_levels,
            mse_bits_frac=mse_bits_frac,
        )

        y_tilde = centroids[idx]
        x_tilde_unit = y_tilde @ rotation

        residual = x_unit - x_tilde_unit
        gamma = torch.linalg.norm(residual, dim=-1, keepdim=True).clamp(min=0.0)
        residual_proj = residual @ qjl_matrix.T
        qjl_sign = residual_proj >= 0.0

        idx_dtype = torch.int16 if centroids.numel() <= 32767 else torch.int32

        mse_bits_low = int(state["mse_bits_low"].item())
        return TurboQuantTensor(
            idx=idx.to(dtype=idx_dtype),
            x_norm=x_norm.to(dtype=self.config.stats_dtype),
            qjl_sign=qjl_sign,
            gamma=gamma.to(dtype=self.config.stats_dtype),
            original_shape=original_shape,
            original_dtype=original_dtype,
            bits=self.config.bits,
            mse_bits_low=mse_bits_low,
            mse_bits_frac=mse_bits_frac,
            centroids_count=int(centroids.numel()),
            head_dim=head_dim,
        )

    def dequantize(self, packed: TurboQuantTensor, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        device = packed.idx.device
        state = self._build_state(head_dim=packed.head_dim, device=device)
        rotation = state["rotation"].to(torch.float32)
        qjl_matrix = state["qjl_matrix"].to(torch.float32)
        centroids = state["centroids"].to(torch.float32)

        flat_size = packed.idx.numel() // packed.head_dim
        idx = packed.idx.reshape(flat_size, packed.head_dim).to(torch.long)
        y_tilde = centroids[idx]
        x_tilde_mse = y_tilde @ rotation

        sign = torch.where(
            packed.qjl_sign.reshape(flat_size, packed.head_dim),
            torch.tensor(1.0, device=device),
            torch.tensor(-1.0, device=device),
        ).to(torch.float32)
        gamma = packed.gamma.reshape(flat_size, 1).to(torch.float32)
        qjl_factor = sqrt(pi / 2.0) / max(packed.head_dim, 1)
        x_tilde_qjl = qjl_factor * gamma * (sign @ qjl_matrix)
        x_tilde_unit = x_tilde_mse + x_tilde_qjl

        x_norm = packed.x_norm.reshape(flat_size, 1).to(torch.float32)
        recon = (x_tilde_unit * x_norm).reshape(*packed.original_shape)
        out_dtype = output_dtype or packed.original_dtype
        return recon.to(out_dtype)

    def quantize_dequantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, TurboQuantTensor]:
        packed = self.quantize(x)
        recon = self.dequantize(packed)
        return recon, packed

    def quantize_kv_cache(self, key: torch.Tensor, value: torch.Tensor) -> Dict[str, TurboQuantTensor]:
        return {
            "k": self.quantize(key),
            "v": self.quantize(value),
        }

    def dequantize_kv_cache(
        self, compressed_kv: Dict[str, TurboQuantTensor], output_dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "k" not in compressed_kv or "v" not in compressed_kv:
            raise ValueError("compressed_kv must contain both 'k' and 'v'")
        return (
            self.dequantize(compressed_kv["k"], output_dtype=output_dtype),
            self.dequantize(compressed_kv["v"], output_dtype=output_dtype),
        )
