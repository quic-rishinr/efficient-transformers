#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Synthetic Data Generator for Model Comparison Testing

Generates synthetic logits to simulate different quality scenarios:
- GOOD: QEff model closely matches baseline (excellent metrics)
- BAD: QEff model shows noticeable degradation (acceptable metrics)
- WORST: QEff model shows significant degradation (poor metrics)
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic logits for testing model comparison metrics."""
    
    def __init__(self, vocab_size: int = 32000, seq_len: int = 512, scenario: str = "good"):
        """
        Initialize synthetic data generator.
        
        Args:
            vocab_size: Size of vocabulary
            seq_len: Sequence length
            scenario: Quality scenario - "good", "bad", or "worst"
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.scenario = scenario.lower()
        
        if self.scenario not in ["good", "bad", "worst"]:
            raise ValueError(f"Invalid scenario: {scenario}. Must be 'good', 'bad', or 'worst'")
        
        logger.info(f"Initialized SyntheticDataGenerator with scenario: {self.scenario}")
    
    def generate_baseline_logits(self) -> torch.Tensor:
        """
        Generate baseline logits with realistic distribution.
        
        Returns:
            Logits tensor [seq_len, vocab_size]
        """
        # Create realistic logits with some tokens having higher probabilities
        logits = torch.randn(self.seq_len, self.vocab_size, dtype=torch.float32) * 2.0
        
        # Make some tokens more likely (simulate language model behavior)
        for i in range(self.seq_len):
            # Pick 10-20 "likely" tokens and boost their logits
            num_likely = np.random.randint(10, 20)
            likely_indices = np.random.choice(self.vocab_size, num_likely, replace=False)
            boost_values = torch.from_numpy(np.random.uniform(3.0, 8.0, num_likely)).float()
            logits[i, likely_indices] += boost_values
        
        return logits
    
    def generate_qeff_logits(self, baseline_logits: torch.Tensor) -> torch.Tensor:
        """
        Generate QEff logits based on scenario.
        
        Args:
            baseline_logits: Baseline logits to perturb
        
        Returns:
            QEff logits tensor [seq_len, vocab_size]
        """
        if self.scenario == "good":
            return self._generate_good_logits(baseline_logits)
        elif self.scenario == "bad":
            return self._generate_bad_logits(baseline_logits)
        else:  # worst
            return self._generate_worst_logits(baseline_logits)
    
    def _generate_good_logits(self, baseline_logits: torch.Tensor) -> torch.Tensor:
        """
        Generate logits with minimal degradation (excellent metrics).
        
        Target metrics:
        - Perplexity change: < 1%
        - Logit MSE: < 1e-4
        - KL Divergence: < 1e-4
        - Top-K Overlap: > 99%
        - Rank Correlation: > 0.9999
        - Next Token Accuracy: > 99%
        """
        # Add very small noise
        noise = torch.randn_like(baseline_logits) * 0.001
        qeff_logits = baseline_logits + noise
        
        # Ensure top predictions remain mostly the same
        # Only change a few predictions (< 1%)
        num_changes = int(self.seq_len * 0.005)  # 0.5% changes
        change_positions = np.random.choice(self.seq_len, num_changes, replace=False)
        
        for pos in change_positions:
            # Slightly shuffle top-k rankings
            top_k_indices = torch.topk(baseline_logits[pos], k=10).indices
            shuffle_amount = torch.randn(10) * 0.01
            qeff_logits[pos, top_k_indices] += shuffle_amount
        
        return qeff_logits
    
    def _generate_bad_logits(self, baseline_logits: torch.Tensor) -> torch.Tensor:
        """
        Generate logits with noticeable degradation (acceptable metrics).
        
        Target metrics:
        - Perplexity change: 5-10%
        - Logit MSE: 1e-3 to 1e-2
        - KL Divergence: 0.01 to 0.1
        - Top-K Overlap: 90-95%
        - Rank Correlation: 0.95-0.99
        - Next Token Accuracy: 90-95%
        """
        # Add moderate noise
        noise = torch.randn_like(baseline_logits) * 0.1
        qeff_logits = baseline_logits + noise
        
        # Change more predictions (5-10%)
        num_changes = int(self.seq_len * 0.075)  # 7.5% changes
        change_positions = np.random.choice(self.seq_len, num_changes, replace=False)
        
        for pos in change_positions:
            # Significantly shuffle rankings
            top_k_indices = torch.topk(baseline_logits[pos], k=20).indices
            
            # Sometimes swap top prediction with another top-k token
            if np.random.random() < 0.3:  # 30% of changed positions
                swap_idx = np.random.randint(1, 10)
                temp = qeff_logits[pos, top_k_indices[0]].clone()
                qeff_logits[pos, top_k_indices[0]] = qeff_logits[pos, top_k_indices[swap_idx]]
                qeff_logits[pos, top_k_indices[swap_idx]] = temp
            
            # Add noise to top-k
            shuffle_amount = torch.randn(20) * 0.5
            qeff_logits[pos, top_k_indices] += shuffle_amount
        
        return qeff_logits
    
    def _generate_worst_logits(self, baseline_logits: torch.Tensor) -> torch.Tensor:
        """
        Generate logits with significant degradation (poor metrics).
        
        Target metrics:
        - Perplexity change: > 10%
        - Logit MSE: > 1e-2
        - KL Divergence: > 0.1
        - Top-K Overlap: < 90%
        - Rank Correlation: < 0.95
        - Next Token Accuracy: < 90%
        """
        # Add large noise
        noise = torch.randn_like(baseline_logits) * 0.5
        qeff_logits = baseline_logits + noise
        
        # Change many predictions (15-20%)
        num_changes = int(self.seq_len * 0.175)  # 17.5% changes
        change_positions = np.random.choice(self.seq_len, num_changes, replace=False)
        
        for pos in change_positions:
            # Heavily shuffle rankings
            top_k_indices = torch.topk(baseline_logits[pos], k=50).indices
            
            # Frequently swap top prediction with another token
            if np.random.random() < 0.5:  # 50% of changed positions
                swap_idx = np.random.randint(1, 20)
                temp = qeff_logits[pos, top_k_indices[0]].clone()
                qeff_logits[pos, top_k_indices[0]] = qeff_logits[pos, top_k_indices[swap_idx]]
                qeff_logits[pos, top_k_indices[swap_idx]] = temp
            
            # Add large noise to top-k
            shuffle_amount = torch.randn(50) * 1.5
            qeff_logits[pos, top_k_indices] += shuffle_amount
            
            # Sometimes add random noise to entire distribution
            if np.random.random() < 0.3:
                qeff_logits[pos] += torch.randn(self.vocab_size) * 0.3
        
        return qeff_logits
    
    def generate_sample(self) -> tuple:
        """
        Generate a complete sample with baseline and QEff logits.
        
        Returns:
            Tuple of (baseline_logits, qeff_logits, target_ids)
        """
        # Generate baseline logits
        baseline_logits = self.generate_baseline_logits()
        
        # Generate QEff logits based on scenario
        qeff_logits = self.generate_qeff_logits(baseline_logits)
        
        # Generate target IDs (ground truth) from baseline
        target_ids = torch.argmax(baseline_logits, dim=-1)
        
        return baseline_logits, qeff_logits, target_ids
    
    def generate_batch(self, num_samples: int) -> list:
        """
        Generate multiple samples.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            List of (baseline_logits, qeff_logits, target_ids) tuples
        """
        logger.info(f"Generating {num_samples} synthetic samples with scenario: {self.scenario}")
        samples = []
        
        for i in range(num_samples):
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
            samples.append(self.generate_sample())
        
        logger.info(f"Completed generating {num_samples} samples")
        return samples


def get_scenario_description(scenario: str) -> str:
    """Get description of the scenario."""
    descriptions = {
        "good": "Excellent quality - QEff model closely matches baseline (< 1% degradation)",
        "bad": "Acceptable quality - QEff model shows noticeable degradation (5-10% degradation)",
        "worst": "Poor quality - QEff model shows significant degradation (> 10% degradation)"
    }
    return descriptions.get(scenario.lower(), "Unknown scenario")
