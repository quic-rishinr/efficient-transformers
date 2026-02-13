#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Configuration file for model comparison metrics.

This file contains all configurable constants used in the model comparison script,
including visualization settings and metric threshold definitions.
"""

# Visualization constants
FIGURE_SIZE = (12, 6)
DPI = 300

# Metric expectation thresholds
METRIC_THRESHOLDS = {
    'perplexity_pct_change': {
        'excellent': 1.0,
        'good': 5.0,
        'acceptable': 10.0,
        'label': 'Perplexity % Change',
        'direction': 'lower',  # Lower is better
        'ideal': 0.0,
        'description': 'Lower is Better'
    },
    'logit_mse': {
        'excellent': 1e-4,
        'good': 1e-3,
        'acceptable': 1e-2,
        'label': 'Logit MSE',
        'direction': 'lower',  # Lower is better
        'ideal': 0.0,
        'description': 'Lower is Better'
    },
    'kl_divergence': {
        'excellent': 1e-4,
        'good': 0.01,
        'acceptable': 0.1,
        'label': 'KL Divergence',
        'direction': 'lower',  # Lower is better
        'ideal': 0.0,
        'description': 'Lower is Better'
    },
    'topk_overlap': {
        'excellent': 99.0,
        'good': 95.0,
        'acceptable': 90.0,
        'label': 'Top-K Overlap (%)',
        'direction': 'higher',  # Higher is better
        'ideal': 100.0,
        'description': 'Higher is Better'
    },
    'rank_correlation': {
        'excellent': 0.9999,
        'good': 0.99,
        'acceptable': 0.95,
        'label': 'Rank Correlation',
        'direction': 'higher',  # Higher is better
        'ideal': 1.0,
        'description': 'Higher is Better'
    },
    'next_token_accuracy': {
        'excellent': 99.0,
        'good': 95.0,
        'acceptable': 90.0,
        'label': 'Next Token Accuracy (%)',
        'direction': 'higher',  # Higher is better
        'ideal': 100.0,
        'description': 'Higher is Better'
    }
}
