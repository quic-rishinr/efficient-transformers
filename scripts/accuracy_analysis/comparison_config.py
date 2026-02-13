#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Configuration constants for model comparison metrics and visualization.
"""

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300

# Metric thresholds for determining quality levels
METRIC_THRESHOLDS = {
    'perplexity_pct_change': {
        'excellent': 1.0,    # < 1% change is excellent
        'good': 5.0,         # < 5% change is good
        'acceptable': 15.0,  # < 15% change is acceptable
        'description': 'Lower is better',
        'ideal': '0%'
    },
    'logit_mse': {
        'excellent': 1e-6,   # Very low MSE
        'good': 1e-4,        # Low MSE
        'acceptable': 1e-2,  # Moderate MSE
        'description': 'Lower is better',
        'ideal': '0'
    },
    'kl_divergence': {
        'excellent': 1e-6,   # Very low KL divergence
        'good': 0.01,        # Low KL divergence
        'acceptable': 0.1,   # Moderate KL divergence
        'description': 'Lower is better',
        'ideal': '0'
    },
    'topk_overlap': {
        'excellent': 95.0,   # > 95% overlap is excellent
        'good': 85.0,        # > 85% overlap is good
        'acceptable': 70.0,  # > 70% overlap is acceptable
        'description': 'Higher is better',
        'ideal': 100.0
    },
    'rank_correlation': {
        'excellent': 0.9999, # Very high correlation
        'good': 0.99,        # High correlation
        'acceptable': 0.95,  # Good correlation
        'description': 'Higher is better',
        'ideal': '1.0'
    },
    'next_token_accuracy': {
        'excellent': 95.0,   # > 95% accuracy is excellent
        'good': 85.0,        # > 85% accuracy is good
        'acceptable': 70.0,  # > 70% accuracy is acceptable
        'description': 'Higher is better',
        'ideal': 100.0
    }
}
