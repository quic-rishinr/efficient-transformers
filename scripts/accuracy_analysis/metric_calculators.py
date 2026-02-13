#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Metric Calculator Classes for Model Comparison

This module contains various metric calculators used to compare model outputs:
- MetricCalculator: Base class for all metric calculators
- PerplexityCalculator: Calculate perplexity for both models
- LogitMSECalculator: Calculate Mean Squared Error between logits
- KLDivergenceCalculator: Calculate KL Divergence between probability distributions
- TopKOverlapCalculator: Calculate overlap in top-k predicted tokens
- RankCorrelationCalculator: Calculate Spearman rank correlation
- NextTokenAccuracyCalculator: Calculate next-token prediction accuracy
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


class MetricCalculator:
    """Base class for metric calculators."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
    
    def compute(self, baseline_logits: torch.Tensor, qeff_logits: torch.Tensor) -> float:
        """Compute metric for a single sample."""
        raise NotImplementedError
    
    def add_result(self, value: float):
        """Add a result to the list."""
        self.results.append(value)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.results:
            return {}
        
        return {
            'mean': np.mean(self.results),
            'std': np.std(self.results),
            'min': np.min(self.results),
            'max': np.max(self.results),
            'median': np.median(self.results),
            'count': len(self.results)
        }


class PerplexityCalculator(MetricCalculator):
    """Calculate perplexity for both models."""
    
    def __init__(self):
        super().__init__("Perplexity")
        self.baseline_losses = []
        self.qeff_losses = []
        self.baseline_perplexities = []  # Store per-sample perplexities
        self.qeff_perplexities = []      # Store per-sample perplexities
    
    def compute(self, baseline_logits: torch.Tensor, qeff_logits: torch.Tensor, target_ids: torch.Tensor) -> Tuple[float, float]:
        """
        Compute perplexity for both models.
        
        Args:
            baseline_logits: Baseline logits [seq_len, vocab_size]
            qeff_logits: QEfficient logits [seq_len, vocab_size]
            target_ids: Target token IDs [seq_len]
        
        Returns:
            Tuple of (baseline_perplexity, qeff_perplexity)
        """
        # Compute cross-entropy loss
        baseline_loss = F.cross_entropy(baseline_logits[:-1], target_ids[1:], reduction='mean')
        qeff_loss = F.cross_entropy(qeff_logits[:-1], target_ids[1:], reduction='mean')
        
        self.baseline_losses.append(baseline_loss.item())
        self.qeff_losses.append(qeff_loss.item())
        
        # Compute per-sample perplexity
        baseline_ppl = torch.exp(baseline_loss).item()
        qeff_ppl = torch.exp(qeff_loss).item()
        
        self.baseline_perplexities.append(baseline_ppl)
        self.qeff_perplexities.append(qeff_ppl)
        
        return baseline_ppl, qeff_ppl
    
    def get_summary(self) -> Dict:
        """Get summary statistics for both models."""
        baseline_avg_loss = np.mean(self.baseline_losses)
        qeff_avg_loss = np.mean(self.qeff_losses)
        
        return {
            'baseline': {
                'perplexity': np.exp(baseline_avg_loss),
                'loss': baseline_avg_loss,
                'perplexity_std': np.std([np.exp(l) for l in self.baseline_losses]),
                'per_sample_perplexities': self.baseline_perplexities  # Add per-sample data
            },
            'qefficient': {
                'perplexity': np.exp(qeff_avg_loss),
                'loss': qeff_avg_loss,
                'perplexity_std': np.std([np.exp(l) for l in self.qeff_losses]),
                'per_sample_perplexities': self.qeff_perplexities  # Add per-sample data
            },
            'difference': {
                'perplexity_diff': np.exp(qeff_avg_loss) - np.exp(baseline_avg_loss),
                'relative_change_pct': ((np.exp(qeff_avg_loss) - np.exp(baseline_avg_loss)) / np.exp(baseline_avg_loss)) * 100
            }
        }


class LogitMSECalculator(MetricCalculator):
    """Calculate Mean Squared Error between logits."""
    
    def __init__(self):
        super().__init__("Logit MSE")
    
    def compute(self, baseline_logits: torch.Tensor, qeff_logits: torch.Tensor) -> float:
        """
        Compute MSE between logits.
        
        Args:
            baseline_logits: Baseline logits [seq_len, vocab_size]
            qeff_logits: QEfficient logits [seq_len, vocab_size]
        
        Returns:
            MSE value
        """
        # Ensure same dtype for comparison
        baseline_logits = baseline_logits.float()
        qeff_logits = qeff_logits.float()
        
        mse = F.mse_loss(baseline_logits, qeff_logits).item()
        self.add_result(mse)
        return mse


class KLDivergenceCalculator(MetricCalculator):
    """Calculate KL Divergence between probability distributions."""
    
    def __init__(self):
        super().__init__("KL Divergence")
    
    def compute(self, baseline_logits: torch.Tensor, qeff_logits: torch.Tensor) -> float:
        """
        Compute KL(P_baseline || P_qeff).
        
        Args:
            baseline_logits: Baseline logits [seq_len, vocab_size]
            qeff_logits: QEfficient logits [seq_len, vocab_size]
        
        Returns:
            KL divergence value
        """
        # Convert to log probabilities for numerical stability
        baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
        qeff_log_probs = F.log_softmax(qeff_logits, dim=-1)
        
        # Compute KL divergence: KL(P || Q) = sum(P * log(P/Q))
        kl_div = F.kl_div(qeff_log_probs, baseline_log_probs, reduction='batchmean', log_target=True).item()
        
        self.add_result(kl_div)
        return kl_div


class TopKOverlapCalculator(MetricCalculator):
    """Calculate overlap in top-k predicted tokens."""
    
    def __init__(self, k: int = 10):
        super().__init__(f"Top-{k} Overlap")
        self.k = k
    
    def compute(self, baseline_logits: torch.Tensor, qeff_logits: torch.Tensor) -> float:
        """
        Compute percentage overlap in top-k tokens.
        
        Args:
            baseline_logits: Baseline logits [seq_len, vocab_size]
            qeff_logits: QEfficient logits [seq_len, vocab_size]
        
        Returns:
            Overlap percentage (0-100)
        """
        # Get top-k indices for each position
        baseline_topk = torch.topk(baseline_logits, self.k, dim=-1).indices
        qeff_topk = torch.topk(qeff_logits, self.k, dim=-1).indices
        
        # Calculate overlap for each position
        overlaps = []
        for i in range(baseline_topk.shape[0]):
            baseline_set = set(baseline_topk[i].cpu().numpy())
            qeff_set = set(qeff_topk[i].cpu().numpy())
            overlap = len(baseline_set & qeff_set) / self.k * 100
            overlaps.append(overlap)
        
        avg_overlap = np.mean(overlaps)
        self.add_result(avg_overlap)
        return avg_overlap


class RankCorrelationCalculator(MetricCalculator):
    """Calculate Spearman rank correlation of token probabilities."""
    
    def __init__(self):
        super().__init__("Rank Correlation")
    
    def compute(self, baseline_logits: torch.Tensor, qeff_logits: torch.Tensor) -> float:
        """
        Compute Spearman correlation of token ranks.
        
        Args:
            baseline_logits: Baseline logits [seq_len, vocab_size]
            qeff_logits: QEfficient logits [seq_len, vocab_size]
        
        Returns:
            Spearman correlation coefficient
        """
        # Flatten logits
        baseline_flat = baseline_logits.flatten().cpu().numpy()
        qeff_flat = qeff_logits.flatten().cpu().numpy()
        
        # Compute Spearman correlation
        correlation, _ = spearmanr(baseline_flat, qeff_flat)
        
        self.add_result(correlation)
        return correlation


class NextTokenAccuracyCalculator(MetricCalculator):
    """Calculate next-token prediction accuracy (argmax agreement)."""
    
    def __init__(self):
        super().__init__("Next-Token Accuracy")
    
    def compute(self, baseline_logits: torch.Tensor, qeff_logits: torch.Tensor) -> float:
        """
        Compute argmax agreement percentage.
        
        Args:
            baseline_logits: Baseline logits [seq_len, vocab_size]
            qeff_logits: QEfficient logits [seq_len, vocab_size]
        
        Returns:
            Accuracy percentage (0-100)
        """
        # Get argmax predictions
        baseline_preds = torch.argmax(baseline_logits, dim=-1)
        qeff_preds = torch.argmax(qeff_logits, dim=-1)
        
        # Calculate agreement
        agreement = (baseline_preds == qeff_preds).float().mean().item() * 100
        
        self.add_result(agreement)
        return agreement
