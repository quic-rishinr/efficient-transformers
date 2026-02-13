# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Accuracy Analysis Tools for QEfficient

This module provides comprehensive accuracy analysis and comparison tools
for evaluating QEfficient-optimized models against HuggingFace baseline models.

Main Components:
- model_comparison_metrics.py: Full comparison script with 6 metrics
- Baseline-only mode for perplexity evaluation
- Support for WikiText-2 and LAMBADA datasets

Usage:
    from efficient-transformers.scripts.accuracy_analysis import model_comparison_metrics
    
    # Or run directly:
    python model_comparison_metrics.py --model-name meta-llama/Llama-3.2-1B ...
"""

__version__ = "1.0.0"
__author__ = "Qualcomm Technologies, Inc."
__license__ = "BSD-3-Clause"

__all__ = [
    "model_comparison_metrics",
]
