# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from math import sqrt

import torch

from QEfficient.transformers.quantizers.turboquant import TurboQuantConfig, TurboQuantizer


def test_turboquant_roundtrip_shape_and_dtype():
    torch.manual_seed(7)
    x = torch.randn(2, 8, 64, 128, dtype=torch.float16)
    quantizer = TurboQuantizer(TurboQuantConfig(bits=3.0, seed=7))
    recon, packed = quantizer.quantize_dequantize(x)

    assert recon.shape == x.shape
    assert recon.dtype == x.dtype
    assert packed.idx.dtype in (torch.int16, torch.int32)
    assert packed.qjl_sign.dtype == torch.bool


def test_turboquant_estimated_compression_ratio_is_gt_1():
    torch.manual_seed(11)
    x = torch.randn(1, 4, 128, 128, dtype=torch.float16)
    quantizer = TurboQuantizer(TurboQuantConfig(bits=3.0, seed=11))
    _, packed = quantizer.quantize_dequantize(x)

    assert packed.estimated_compression_ratio() > 1.0


def test_turboquant_reconstruction_and_logits_are_finite():
    torch.manual_seed(3)
    x = torch.randn(1, 4, 128, 128, dtype=torch.float16)
    quantizer = TurboQuantizer(TurboQuantConfig(bits=3.0, seed=3))
    recon, _ = quantizer.quantize_dequantize(x)

    mse = torch.mean((x.float() - recon.float()) ** 2).item()
    assert mse > 0.0
    assert torch.isfinite(torch.tensor(mse))
    assert mse < 5.0

    q = torch.randn(1, 4, 32, 128, dtype=torch.float16)
    logits_ref = torch.matmul(q.float(), x.float().transpose(-1, -2)) / sqrt(x.shape[-1])
    logits_rec = torch.matmul(q.float(), recon.float().transpose(-1, -2)) / sqrt(x.shape[-1])
    logit_mse = torch.mean((logits_ref - logits_rec) ** 2).item()
    assert torch.isfinite(torch.tensor(logit_mse))
    assert logit_mse < 2.5


def test_turboquant_invalid_config_raises():
    try:
        _ = TurboQuantConfig(bits=0.5)
        assert False, "Expected ValueError for invalid bits"
    except ValueError:
        pass


def test_turboquant_kv_cache_roundtrip():
    torch.manual_seed(19)
    k = torch.randn(1, 4, 64, 128, dtype=torch.float16)
    v = torch.randn(1, 4, 64, 128, dtype=torch.float16)
    quantizer = TurboQuantizer(TurboQuantConfig(bits=3.0, seed=19))
    packed = quantizer.quantize_kv_cache(k, v)
    k_rec, v_rec = quantizer.dequantize_kv_cache(packed, output_dtype=torch.float16)

    assert k_rec.shape == k.shape
    assert v_rec.shape == v.shape
