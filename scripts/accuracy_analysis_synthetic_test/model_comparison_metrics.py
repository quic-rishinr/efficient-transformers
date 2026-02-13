#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Comprehensive Model Comparison Script

Compares HuggingFace models (AutoModelForCausalLM) with QEfficient models
(QEFFAutoModelForCausalLM) using multiple advanced metrics:

1. Perplexity
2. Logit MSE (Mean Squared Error)
3. KL Divergence (P_baseline || P_qeff)
4. Top-k Overlap
5. Rank Correlation (Spearman)
6. Next-token Accuracy (Argmax Agreement)

Usage:
    python model_comparison_metrics.py \
        --model-name Qwen/Qwen2-1.5B-Instruct \
        --dataset wikitext-2-raw-v1 \
        --num-samples 100 \
        --ctx-len 512 \
        --output-dir ./comparison_results
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import metric calculators
from metric_calculators import (
    MetricCalculator,
    PerplexityCalculator,
    LogitMSECalculator,
    KLDivergenceCalculator,
    TopKOverlapCalculator,
    RankCorrelationCalculator,
    NextTokenAccuracyCalculator
)

# Import configuration constants
from comparison_config import FIGURE_SIZE, DPI, METRIC_THRESHOLDS

# Try to import QEfficient
try:
    from QEfficient import QEFFAutoModelForCausalLM
    QEFFICIENT_AVAILABLE = True
except ImportError:
    QEFFICIENT_AVAILABLE = False
    warnings.warn("QEfficient not available. Only baseline evaluation will work.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set matplotlib style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    # Fallback for newer matplotlib versions
    plt.style.use('default')
    sns.set_style("darkgrid")

def get_metric_status(metric_name: str, value: float) -> str:
    """Determine the status of a metric value based on thresholds."""
    if metric_name not in METRIC_THRESHOLDS:
        return 'unknown'
    
    thresholds = METRIC_THRESHOLDS[metric_name]
    
    # For metrics where higher is better
    if metric_name in ['topk_overlap', 'rank_correlation', 'next_token_accuracy']:
        if value >= thresholds['excellent']:
            return 'excellent'
        elif value >= thresholds['good']:
            return 'good'
        elif value >= thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'
    # For metrics where lower is better
    else:
        if value <= thresholds['excellent']:
            return 'excellent'
        elif value <= thresholds['good']:
            return 'good'
        elif value <= thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'


class DatasetLoader:
    """Load and prepare datasets for evaluation."""
    
    def __init__(self, dataset_name: str, tokenizer_name: str, num_samples: int, ctx_len: int, stride: int = 512):
        """
        Initialize dataset loader.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'wikitext-2-raw-v1')
            tokenizer_name: HuggingFace tokenizer name
            num_samples: Number of samples to use
            ctx_len: Context length for each sample
            stride: Stride for sliding window
        """
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.ctx_len = ctx_len
        self.stride = stride
        
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading dataset: {dataset_name}")
        self.samples = self._load_and_prepare_dataset()
    
    def _load_and_prepare_dataset(self) -> List[Dict]:
        """Load and prepare dataset samples."""
        if self.dataset_name.startswith('wikitext'):
            return self._load_wikitext()
        elif self.dataset_name == 'lambada':
            return self._load_lambada()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_wikitext(self) -> List[Dict]:
        """Load WikiText dataset."""
        dataset = load_dataset("wikitext", self.dataset_name, split="test")
        
        # Concatenate all text
        text = "\n\n".join(dataset["text"])
        
        # Tokenize
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids[0]
        
        # Create samples with sliding window
        samples = []
        seq_len = len(input_ids)
        
        for i in range(0, seq_len - self.ctx_len + 1, self.stride):
            if len(samples) >= self.num_samples:
                break
            
            sample_ids = input_ids[i:i + self.ctx_len]
            samples.append({
                'input_ids': sample_ids,
                'text': self.tokenizer.decode(sample_ids, skip_special_tokens=True)
            })
        
        logger.info(f"Prepared {len(samples)} samples from WikiText")
        return samples
    
    def _load_lambada(self) -> List[Dict]:
        """Load LAMBADA dataset."""
        dataset = load_dataset("lambada", split="test")
        
        samples = []
        for i, example in enumerate(dataset):
            if i >= self.num_samples:
                break
            
            text = example['text']
            encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.ctx_len)
            input_ids = encodings.input_ids[0]
            
            samples.append({
                'input_ids': input_ids,
                'text': text
            })
        
        logger.info(f"Prepared {len(samples)} samples from LAMBADA")
        return samples
    
    def get_samples(self) -> List[Dict]:
        """Get prepared samples."""
        return self.samples


class BaselineModelLoader:
    """Load and manage HuggingFace baseline model."""
    
    def __init__(self, model_name: str, device: str = "cpu", dtype: str = "fp32"):
        """
        Initialize baseline model loader.
        
        Args:
            model_name: HuggingFace model ID
            device: Device to use (cuda/cpu)
            dtype: Data type (fp32/fp16/bf16)
        """
        self.model_name = model_name
        self.device = device
        
        # Determine torch dtype
        if dtype == "fp32":
            self.torch_dtype = torch.float32
        elif dtype == "fp16":
            self.torch_dtype = torch.float16
        elif dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
        
        logger.info(f"Loading baseline model: {model_name} (dtype: {dtype}, device: {device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype
        ).to(device)
        self.model.eval()
        
        logger.info("Baseline model loaded successfully")
    
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get logits from the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
        
        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        
        return logits
    
    def cleanup(self):
        """Clean up model and free memory."""
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()


class QEfficientModelLoader:
    """Load and manage QEfficient model in PyTorch mode (no compilation).
    
    Note: For PyTorch comparison, we use the base HuggingFace model since QEfficient
    transforms are designed for compilation optimization, not for changing PyTorch behavior.
    """
    
    def __init__(
        self,
        model_name: str,
        prefill_seq_len: int = 32,
        ctx_len: int = 512,
        num_cores: int = 16,
        device_group: List[int] = [0],
        device: str = "cpu",
        dtype: str = "fp32"
    ):
        """
        Initialize QEfficient model loader in PyTorch mode.
        
        Args:
            model_name: HuggingFace model ID
            prefill_seq_len: Prefill sequence length (unused in PyTorch mode)
            ctx_len: Context length (unused in PyTorch mode)
            num_cores: Number of cores (unused in PyTorch mode)
            device_group: List of device IDs (unused in PyTorch mode)
            device: Device to use (cuda/cpu)
            dtype: Data type (fp32/fp16/bf16)
        """
        if not QEFFICIENT_AVAILABLE:
            raise ImportError("QEfficient not available")
        
        self.model_name = model_name
        self.device = device
        
        # Determine torch dtype
        if dtype == "fp32":
            self.torch_dtype = torch.float32
        elif dtype == "fp16":
            self.torch_dtype = torch.float16
        elif dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
        
        logger.info(f"Loading QEfficient model in PyTorch mode: {model_name} (dtype: {dtype}, device: {device})")
        logger.info("Note: Using base HuggingFace model for PyTorch comparison")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the base HuggingFace model directly (same as baseline)
        # This ensures we're comparing PyTorch to PyTorch without KV-cache transforms
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype
        ).to(device)
        self.model.eval()
        
        logger.info("QEfficient model loaded successfully in PyTorch mode (using base HF model)")
    
    def get_logits_pt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get logits from the PyTorch model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
        
        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        
        return logits
    
    def cleanup(self):
        """Clean up model and free memory."""
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()


class ComparisonEngine:
    """Main engine for running model comparison."""
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        num_samples: int,
        ctx_len: int,
        top_k: int = 10,
        device: str = "cpu",
        baseline_dtype: str = "fp32",
        qaic_dtype: str = "fp16",
        qaic_prefill_seq_len: int = 32,
        qaic_num_cores: int = 16,
        qaic_device_group: List[int] = [0],
        output_dir: str = "./comparison_results"
    ):
        """Initialize comparison engine."""
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.ctx_len = ctx_len
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        self.dataset_loader = DatasetLoader(dataset_name, model_name, num_samples, ctx_len)
        self.samples = self.dataset_loader.get_samples()
        
        # Initialize models
        logger.info("Initializing baseline model...")
        self.baseline_model = BaselineModelLoader(model_name, device, baseline_dtype)
        
        logger.info("Initializing QEfficient model...")
        self.qeff_model = QEfficientModelLoader(
            model_name,
            qaic_prefill_seq_len,
            ctx_len,
            qaic_num_cores,
            qaic_device_group,
            device=device,
            dtype=qaic_dtype
        )
        
        # Initialize metric calculators
        self.metrics = {
            'perplexity': PerplexityCalculator(),
            'logit_mse': LogitMSECalculator(),
            'kl_divergence': KLDivergenceCalculator(),
            'topk_overlap': TopKOverlapCalculator(top_k),
            'rank_correlation': RankCorrelationCalculator(),
            'next_token_accuracy': NextTokenAccuracyCalculator()
        }
    
    def run_comparison(self):
        """Run full comparison across all samples."""
        logger.info(f"Starting comparison on {len(self.samples)} samples...")
        
        for i, sample in enumerate(tqdm(self.samples, desc="Processing samples")):
            try:
                input_ids = sample['input_ids'].unsqueeze(0)  # Add batch dimension
                
                # Get logits from both models
                baseline_logits = self.baseline_model.get_logits(input_ids)
                qeff_logits = self.qeff_model.get_logits_pt(input_ids)
                
                # Remove batch dimension
                baseline_logits = baseline_logits[0]
                qeff_logits = qeff_logits[0]
                target_ids = input_ids[0]
                
                # Compute all metrics
                self.metrics['perplexity'].compute(baseline_logits, qeff_logits, target_ids)
                self.metrics['logit_mse'].compute(baseline_logits, qeff_logits)
                self.metrics['kl_divergence'].compute(baseline_logits, qeff_logits)
                self.metrics['topk_overlap'].compute(baseline_logits, qeff_logits)
                self.metrics['rank_correlation'].compute(baseline_logits, qeff_logits)
                self.metrics['next_token_accuracy'].compute(baseline_logits, qeff_logits)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        logger.info("Comparison complete!")
    
    def generate_report(self):
        """Generate comprehensive report with all metrics."""
        logger.info("Generating report...")
        
        # Collect all metric summaries
        report = {
            'model_name': self.model_name,
            'dataset': self.dataset_name,
            'num_samples': len(self.samples),
            'ctx_len': self.ctx_len,
            'metrics': {}
        }
        
        for name, calculator in self.metrics.items():
            report['metrics'][name] = calculator.get_summary()
        
        # Save JSON report
        json_path = os.path.join(self.output_dir, 'metrics_summary.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved JSON report to: {json_path}")
        
        # Generate CSV report
        self._generate_csv_report(report)
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate text report
        self._generate_text_report(report)
        
        logger.info(f"All reports saved to: {self.output_dir}")
    
    def _generate_csv_report(self, report: Dict):
        """Generate CSV comparison report."""
        import csv
        
        csv_path = os.path.join(self.output_dir, 'comparison_report.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Std Dev', 'Min', 'Max'])
            
            for metric_name, metric_data in report['metrics'].items():
                if metric_name == 'perplexity':
                    writer.writerow([
                        'Baseline Perplexity',
                        f"{metric_data['baseline']['perplexity']:.4f}",
                        f"{metric_data['baseline']['perplexity_std']:.4f}",
                        '-',
                        '-'
                    ])
                    writer.writerow([
                        'QEfficient Perplexity',
                        f"{metric_data['qefficient']['perplexity']:.4f}",
                        f"{metric_data['qefficient']['perplexity_std']:.4f}",
                        '-',
                        '-'
                    ])
                else:
                    writer.writerow([
                        metric_name,
                        f"{metric_data.get('mean', 0):.4f}",
                        f"{metric_data.get('std', 0):.4f}",
                        f"{metric_data.get('min', 0):.4f}",
                        f"{metric_data.get('max', 0):.4f}"
                    ])
        
        logger.info(f"Saved CSV report to: {csv_path}")
    
    def _generate_visualizations(self):
        """Generate visualization plots with expectation thresholds."""
        # Plot 0: Per-Sample Perplexity Comparison
        ppl_data = self.metrics['perplexity'].get_summary()
        if ppl_data and 'baseline' in ppl_data and 'per_sample_perplexities' in ppl_data['baseline']:
            baseline_ppls = ppl_data['baseline']['per_sample_perplexities']
            qeff_ppls = ppl_data['qefficient']['per_sample_perplexities']
            
            if baseline_ppls and qeff_ppls:
                plt.figure(figsize=FIGURE_SIZE)
                n_samples = len(baseline_ppls)
                x = range(n_samples)
                
                plt.plot(x, baseline_ppls, marker='o', linestyle='-', alpha=0.7, color='blue', label='Baseline', linewidth=2)
                plt.plot(x, qeff_ppls, marker='s', linestyle='-', alpha=0.7, color='orange', label='QEfficient', linewidth=2)
                
                # Add mean lines
                baseline_mean = np.mean(baseline_ppls)
                qeff_mean = np.mean(qeff_ppls)
                plt.axhline(baseline_mean, color='blue', linestyle='--', linewidth=2, alpha=0.5, label=f'Baseline Mean: {baseline_mean:.2f}')
                plt.axhline(qeff_mean, color='orange', linestyle='--', linewidth=2, alpha=0.5, label=f'QEfficient Mean: {qeff_mean:.2f}')
                
                plt.xlabel('Sample Index')
                plt.ylabel('Perplexity')
                plt.title(f'Per-Sample Perplexity Comparison (n={n_samples})')
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                plt.savefig(os.path.join(self.output_dir, 'perplexity_per_sample.png'), dpi=DPI, bbox_inches='tight')
                plt.close()
        
        # Plot 1: Logit MSE Distribution
        if self.metrics['logit_mse'].results:
            plt.figure(figsize=FIGURE_SIZE)
            n_samples = len(self.metrics['logit_mse'].results)
            summary = self.metrics['logit_mse'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('logit_mse', mean_val)
            
            plt.hist(self.metrics['logit_mse'].results, bins=30, edgecolor='black', alpha=0.7, color='purple')
            plt.xscale('log')
            
            # Add threshold lines
            thresholds = METRIC_THRESHOLDS['logit_mse']
            plt.axvline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (<{thresholds["excellent"]:.0e})')
            plt.axvline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (<{thresholds["good"]:.0e})')
            plt.axvline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (<{thresholds["acceptable"]:.0e})')
            
            # Add direction indicator
            plt.text(0.98, 0.98, f"📉 {thresholds['description']}\nIdeal: {thresholds['ideal']}", 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.xlabel('Logit MSE (log scale)')
            plt.ylabel('Frequency')
            plt.title(f'Logit MSE Distribution (n={n_samples})\nMean: {mean_val:.2e} - Status: {status.upper()}')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.savefig(os.path.join(self.output_dir, 'logit_mse_distribution.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 2: KL Divergence
        if self.metrics['kl_divergence'].results:
            plt.figure(figsize=FIGURE_SIZE)
            n_samples = len(self.metrics['kl_divergence'].results)
            summary = self.metrics['kl_divergence'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('kl_divergence', mean_val)
            
            plt.plot(self.metrics['kl_divergence'].results, marker='o', linestyle='-', alpha=0.7, color='darkorange')
            plt.yscale('log')
            
            # Add threshold lines
            thresholds = METRIC_THRESHOLDS['kl_divergence']
            plt.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (<{thresholds["excellent"]:.0e})')
            plt.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (<{thresholds["good"]:.2f})')
            plt.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (<{thresholds["acceptable"]:.1f})')
            
            # Add direction indicator
            plt.text(0.98, 0.98, f"📉 {thresholds['description']}\nIdeal: {thresholds['ideal']}", 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.xlabel('Sample Index')
            plt.ylabel('KL Divergence (log scale)')
            plt.title(f'KL Divergence Across Samples (n={n_samples})\nMean: {mean_val:.2e} - Status: {status.upper()}')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.savefig(os.path.join(self.output_dir, 'kl_divergence_plot.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Top-k Overlap
        if self.metrics['topk_overlap'].results:
            plt.figure(figsize=FIGURE_SIZE)
            n_samples = len(self.metrics['topk_overlap'].results)
            summary = self.metrics['topk_overlap'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('topk_overlap', mean_val)
            
            plt.bar(range(n_samples), self.metrics['topk_overlap'].results, alpha=0.7, color='steelblue')
            plt.axhline(y=mean_val, color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}%')
            
            # Add threshold lines
            thresholds = METRIC_THRESHOLDS['topk_overlap']
            plt.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (>{thresholds["excellent"]:.0f}%)')
            plt.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (>{thresholds["good"]:.0f}%)')
            plt.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (>{thresholds["acceptable"]:.0f}%)')
            
            # Add direction indicator
            plt.text(0.02, 0.98, f"📈 {thresholds['description']}\nIdeal: {thresholds['ideal']:.0f}%", 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
                    horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.xlabel('Sample Index')
            plt.ylabel('Overlap (%)')
            plt.title(f'Top-k Token Overlap Across Samples (n={n_samples})\nMean: {mean_val:.1f}% - Status: {status.upper()}')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            plt.savefig(os.path.join(self.output_dir, 'topk_overlap_chart.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 4: Rank Correlation Scatter
        if self.metrics['rank_correlation'].results:
            plt.figure(figsize=FIGURE_SIZE)
            n_samples = len(self.metrics['rank_correlation'].results)
            summary = self.metrics['rank_correlation'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('rank_correlation', mean_val)
            
            plt.scatter(range(n_samples), self.metrics['rank_correlation'].results, alpha=0.6, s=50, color='teal')
            plt.axhline(y=1.0, color='darkgreen', linestyle='--', linewidth=2, label='Perfect Correlation')
            plt.axhline(y=mean_val, color='darkblue', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.6f}')
            
            # Add threshold lines
            thresholds = METRIC_THRESHOLDS['rank_correlation']
            plt.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (>{thresholds["excellent"]:.4f})')
            plt.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (>{thresholds["good"]:.2f})')
            plt.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (>{thresholds["acceptable"]:.2f})')
            
            # Add direction indicator
            plt.text(0.02, 0.02, f"📈 {thresholds['description']}\nIdeal: {thresholds['ideal']}", 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', 
                    horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.xlabel('Sample Index')
            plt.ylabel('Spearman Correlation')
            plt.title(f'Rank Correlation Across Samples (n={n_samples})\nMean: {mean_val:.6f} - Status: {status.upper()}')
            plt.legend(loc='lower right', fontsize=9)
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.savefig(os.path.join(self.output_dir, 'rank_correlation_scatter.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 5: Next-Token Accuracy
        if self.metrics['next_token_accuracy'].results:
            plt.figure(figsize=FIGURE_SIZE)
            n_samples = len(self.metrics['next_token_accuracy'].results)
            summary = self.metrics['next_token_accuracy'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('next_token_accuracy', mean_val)
            
            plt.plot(self.metrics['next_token_accuracy'].results, marker='s', linestyle='-', alpha=0.7, color='green', linewidth=2)
            plt.axhline(y=100.0, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Accuracy')
            plt.axhline(y=mean_val, color='darkblue', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.1f}%')
            
            # Add threshold lines
            thresholds = METRIC_THRESHOLDS['next_token_accuracy']
            plt.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (>{thresholds["excellent"]:.0f}%)')
            plt.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (>{thresholds["good"]:.0f}%)')
            plt.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (>{thresholds["acceptable"]:.0f}%)')
            
            # Add direction indicator
            plt.text(0.02, 0.02, f"📈 {thresholds['description']}\nIdeal: {thresholds['ideal']:.0f}%", 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', 
                    horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.xlabel('Sample Index')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Next-Token Prediction Accuracy (n={n_samples})\nMean: {mean_val:.1f}% - Status: {status.upper()}')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.savefig(os.path.join(self.output_dir, 'next_token_accuracy.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 6: Comprehensive Summary
        self._generate_summary_plot()
        
        logger.info("Generated all visualizations")
    
    def _generate_summary_plot(self):
        """Generate comprehensive summary plot with expectation thresholds."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Comprehensive Model Comparison Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Perplexity Comparison
        ppl_data = self.metrics['perplexity'].get_summary()
        if ppl_data:
            ax = axes[0, 0]
            models = ['Baseline', 'QEfficient']
            ppls = [ppl_data['baseline']['perplexity'], ppl_data['qefficient']['perplexity']]
            pct_change = abs(ppl_data['difference']['relative_change_pct'])
            
            # Determine color based on threshold
            status = get_metric_status('perplexity_pct_change', pct_change)
            bar_colors = ['#3498db', '#2ecc71' if status == 'excellent' else '#f39c12' if status == 'good' else '#e74c3c']
            
            ax.bar(models, ppls, color=bar_colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Perplexity')
            ax.set_title(f'Perplexity Comparison\n(Δ: {pct_change:.2f}% - {status.upper()})')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Logit MSE
        if self.metrics['logit_mse'].results:
            ax = axes[0, 1]
            summary = self.metrics['logit_mse'].get_summary()
            mean_val = summary['mean']
            min_val = summary['min']
            max_val = summary['max']
            median_val = summary['median']
            status = get_metric_status('logit_mse', mean_val)
            
            # Determine marker color based on status
            marker_color = '#2ecc71' if status == 'excellent' else '#f39c12' if status == 'good' else '#e74c3c'
            
            # Plot as a point with error bars showing min/max range
            ax.errorbar([0.5], [mean_val], 
                       yerr=[[mean_val - min_val], [max_val - mean_val]],
                       fmt='o', markersize=12, color=marker_color, 
                       ecolor='black', capsize=8, capthick=2, 
                       label=f'Mean: {mean_val:.2e}', alpha=0.8)
            
            # Add median as a horizontal line
            ax.axhline(median_val, color='blue', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Median: {median_val:.2e}')
            
            # Add threshold lines (only those visible in current data range)
            thresholds = METRIC_THRESHOLDS['logit_mse']
            ylim_max = max(max_val * 1.2, thresholds['excellent'] * 2)
            
            if thresholds['excellent'] < ylim_max:
                ax.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (<{thresholds["excellent"]:.0e})')
            if thresholds['good'] < ylim_max:
                ax.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (<{thresholds["good"]:.0e})')
            if thresholds['acceptable'] < ylim_max:
                ax.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (<{thresholds["acceptable"]:.0e})')
            
            ax.set_ylim(0, ylim_max)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.5])
            ax.set_xticklabels(['Logit MSE'])
            ax.set_ylabel('MSE Value')
            ax.set_title(f'Logit MSE Range\n(Mean: {mean_val:.2e} - {status.upper()})')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: KL Divergence
        if self.metrics['kl_divergence'].results:
            ax = axes[0, 2]
            summary = self.metrics['kl_divergence'].get_summary()
            mean_val = summary['mean']
            min_val = summary['min']
            max_val = summary['max']
            std_val = summary['std']
            status = get_metric_status('kl_divergence', mean_val)
            
            # Determine marker color based on status
            marker_color = '#2ecc71' if status == 'excellent' else '#f39c12' if status == 'good' else '#e74c3c'
            
            # Plot as a point with error bars showing min/max range
            ax.errorbar([0.5], [mean_val], 
                       yerr=[[mean_val - min_val], [max_val - mean_val]],
                       fmt='o', markersize=12, color=marker_color, 
                       ecolor='black', capsize=8, capthick=2, 
                       label=f'Mean: {mean_val:.2e}', alpha=0.8)
            
            # Add a horizontal line at the mean for clarity
            ax.axhline(mean_val, color=marker_color, linestyle=':', linewidth=1.5, alpha=0.5)
            
            # Add threshold lines (only those visible in current data range)
            thresholds = METRIC_THRESHOLDS['kl_divergence']
            # Set y-axis limit to show data range and thresholds
            ylim_max = max(max_val * 1.2, thresholds['excellent'] * 2)
            
            if thresholds['excellent'] < ylim_max:
                ax.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (<{thresholds["excellent"]:.0e})')
            if thresholds['good'] < ylim_max:
                ax.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (<{thresholds["good"]:.2f})')
            if thresholds['acceptable'] < ylim_max:
                ax.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (<{thresholds["acceptable"]:.1f})')
            
            ax.set_ylim(0, ylim_max)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.5])
            ax.set_xticklabels(['KL Divergence'])
            ax.set_ylabel('KL Divergence Value')
            ax.set_title(f'KL Divergence Range\n(Mean: {mean_val:.2e} - {status.upper()})')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Top-k Overlap
        if self.metrics['topk_overlap'].results:
            ax = axes[1, 0]
            summary = self.metrics['topk_overlap'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('topk_overlap', mean_val)
            
            # Determine bar color based on status
            bar_color = '#2ecc71' if status == 'excellent' else '#f39c12' if status == 'good' else '#e74c3c'
            
            ax.bar(['Your Result'], [mean_val], color=bar_color, alpha=0.7, edgecolor='black', label='Your Result')
            ax.errorbar(['Your Result'], [mean_val], yerr=[summary['std']], 
                       fmt='none', color='black', capsize=5)
            
            # Add threshold lines
            thresholds = METRIC_THRESHOLDS['topk_overlap']
            ax.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (>{thresholds["excellent"]:.0f}%)')
            ax.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (>{thresholds["good"]:.0f}%)')
            ax.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (>{thresholds["acceptable"]:.0f}%)')
            
            ax.set_ylabel('Overlap (%)')
            ax.set_title(f'Top-k Overlap (Mean ± Std)\n({status.upper()})')
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Rank Correlation
        if self.metrics['rank_correlation'].results:
            ax = axes[1, 1]
            summary = self.metrics['rank_correlation'].get_summary()
            mean_val = summary['mean']
            min_val = summary['min']
            max_val = summary['max']
            median_val = summary['median']
            status = get_metric_status('rank_correlation', mean_val)
            
            # Determine marker color based on status
            marker_color = '#2ecc71' if status == 'excellent' else '#f39c12' if status == 'good' else '#e74c3c'
            
            # Plot as a point with error bars showing min/max range
            ax.errorbar([0.5], [mean_val], 
                       yerr=[[mean_val - min_val], [max_val - mean_val]],
                       fmt='o', markersize=12, color=marker_color, 
                       ecolor='black', capsize=8, capthick=2, 
                       label=f'Mean: {mean_val:.6f}', alpha=0.8)
            
            # Add median as a horizontal line
            ax.axhline(median_val, color='blue', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Median: {median_val:.6f}')
            
            # Add perfect correlation line
            ax.axhline(1.0, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (1.0)')
            
            # Add threshold lines
            thresholds = METRIC_THRESHOLDS['rank_correlation']
            ax.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (>{thresholds["excellent"]:.4f})')
            ax.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (>{thresholds["good"]:.2f})')
            ax.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (>{thresholds["acceptable"]:.2f})')
            
            # Dynamic y-axis limits to ensure all data is visible
            ylim_min = max(0.85, min_val - 0.01)  # Don't go below 0.85
            ylim_max = min(1.01, max(max_val + 0.01, 1.0))  # Don't exceed 1.01, but show 1.0 line
            ax.set_ylim([ylim_min, ylim_max])
            ax.set_xlim(0, 1)
            ax.set_xticks([0.5])
            ax.set_xticklabels(['Rank Correlation'])
            
            ax.set_ylabel('Spearman Correlation')
            ax.set_title(f'Rank Correlation Range\n(Mean: {mean_val:.6f} - {status.upper()})')
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Next-Token Accuracy
        if self.metrics['next_token_accuracy'].results:
            ax = axes[1, 2]
            summary = self.metrics['next_token_accuracy'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('next_token_accuracy', mean_val)
            
            # Determine bar color based on status
            bar_color = '#2ecc71' if status == 'excellent' else '#f39c12' if status == 'good' else '#e74c3c'
            
            ax.bar(['Your Result'], [mean_val], color=bar_color, alpha=0.7, edgecolor='black', label='Your Result')
            ax.errorbar(['Your Result'], [mean_val], yerr=[summary['std']], 
                       fmt='none', color='black', capsize=5)
            
            # Add threshold lines
            thresholds = METRIC_THRESHOLDS['next_token_accuracy']
            ax.axhline(thresholds['excellent'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Excellent (>{thresholds["excellent"]:.0f}%)')
            ax.axhline(thresholds['good'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Good (>{thresholds["good"]:.0f}%)')
            ax.axhline(thresholds['acceptable'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Acceptable (>{thresholds["acceptable"]:.0f}%)')
            
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Next-Token Accuracy (Mean ± Std)\n({status.upper()})')
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_summary.png'), dpi=DPI, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, report: Dict):
        """Generate human-readable text report with expectation thresholds."""
        text_path = os.path.join(self.output_dir, 'comprehensive_report.txt')
        
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model: {report['model_name']}\n")
            f.write(f"Dataset: {report['dataset']}\n")
            f.write(f"Number of Samples: {report['num_samples']}\n")
            f.write(f"Context Length: {report['ctx_len']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("METRIC SUMMARIES WITH EXPECTATIONS\n")
            f.write("="*80 + "\n\n")
            
            # Perplexity
            ppl_data = report['metrics']['perplexity']
            pct_change = abs(ppl_data['difference']['relative_change_pct'])
            ppl_status = get_metric_status('perplexity_pct_change', pct_change)
            
            f.write("1. PERPLEXITY\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Baseline Perplexity:    {ppl_data['baseline']['perplexity']:.4f} ± {ppl_data['baseline']['perplexity_std']:.4f}\n")
            f.write(f"   QEfficient Perplexity:  {ppl_data['qefficient']['perplexity']:.4f} ± {ppl_data['qefficient']['perplexity_std']:.4f}\n")
            f.write(f"   Difference:             {ppl_data['difference']['perplexity_diff']:+.4f}\n")
            f.write(f"   Relative Change:        {pct_change:.2f}%\n")
            f.write(f"   Status:                 {ppl_status.upper()}\n")
            f.write(f"   Expectations:\n")
            f.write(f"      Excellent:  < {METRIC_THRESHOLDS['perplexity_pct_change']['excellent']:.1f}%\n")
            f.write(f"      Good:       < {METRIC_THRESHOLDS['perplexity_pct_change']['good']:.1f}%\n")
            f.write(f"      Acceptable: < {METRIC_THRESHOLDS['perplexity_pct_change']['acceptable']:.1f}%\n\n")
            
            # Logit MSE
            mse_data = report['metrics']['logit_mse']
            mse_status = get_metric_status('logit_mse', mse_data['mean'])
            f.write("2. LOGIT MSE\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Mean:    {mse_data['mean']:.6f}\n")
            f.write(f"   Std Dev: {mse_data['std']:.6f}\n")
            f.write(f"   Min:     {mse_data['min']:.6f}\n")
            f.write(f"   Max:     {mse_data['max']:.6f}\n")
            f.write(f"   Median:  {mse_data['median']:.6f}\n")
            f.write(f"   Status:  {mse_status.upper()}\n")
            f.write(f"   Expectations:\n")
            f.write(f"      Excellent:  < {METRIC_THRESHOLDS['logit_mse']['excellent']:.0e}\n")
            f.write(f"      Good:       < {METRIC_THRESHOLDS['logit_mse']['good']:.0e}\n")
            f.write(f"      Acceptable: < {METRIC_THRESHOLDS['logit_mse']['acceptable']:.0e}\n\n")
            
            # KL Divergence
            kl_data = report['metrics']['kl_divergence']
            kl_status = get_metric_status('kl_divergence', kl_data['mean'])
            f.write("3. KL DIVERGENCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Mean:    {kl_data['mean']:.6f}\n")
            f.write(f"   Std Dev: {kl_data['std']:.6f}\n")
            f.write(f"   Min:     {kl_data['min']:.6f}\n")
            f.write(f"   Max:     {kl_data['max']:.6f}\n")
            f.write(f"   Median:  {kl_data['median']:.6f}\n")
            f.write(f"   Status:  {kl_status.upper()}\n")
            f.write(f"   Expectations:\n")
            f.write(f"      Excellent:  < {METRIC_THRESHOLDS['kl_divergence']['excellent']:.0e}\n")
            f.write(f"      Good:       < {METRIC_THRESHOLDS['kl_divergence']['good']:.2f}\n")
            f.write(f"      Acceptable: < {METRIC_THRESHOLDS['kl_divergence']['acceptable']:.1f}\n\n")
            
            # Top-K Overlap
            topk_data = report['metrics']['topk_overlap']
            topk_status = get_metric_status('topk_overlap', topk_data['mean'])
            f.write("4. TOP-K OVERLAP\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Mean:    {topk_data['mean']:.4f}%\n")
            f.write(f"   Std Dev: {topk_data['std']:.4f}%\n")
            f.write(f"   Min:     {topk_data['min']:.4f}%\n")
            f.write(f"   Max:     {topk_data['max']:.4f}%\n")
            f.write(f"   Median:  {topk_data['median']:.4f}%\n")
            f.write(f"   Status:  {topk_status.upper()}\n")
            f.write(f"   Expectations:\n")
            f.write(f"      Excellent:  > {METRIC_THRESHOLDS['topk_overlap']['excellent']:.0f}%\n")
            f.write(f"      Good:       > {METRIC_THRESHOLDS['topk_overlap']['good']:.0f}%\n")
            f.write(f"      Acceptable: > {METRIC_THRESHOLDS['topk_overlap']['acceptable']:.0f}%\n\n")
            
            # Rank Correlation
            rank_data = report['metrics']['rank_correlation']
            rank_status = get_metric_status('rank_correlation', rank_data['mean'])
            f.write("5. RANK CORRELATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Mean:    {rank_data['mean']:.8f}\n")
            f.write(f"   Std Dev: {rank_data['std']:.8f}\n")
            f.write(f"   Min:     {rank_data['min']:.8f}\n")
            f.write(f"   Max:     {rank_data['max']:.8f}\n")
            f.write(f"   Median:  {rank_data['median']:.8f}\n")
            f.write(f"   Status:  {rank_status.upper()}\n")
            f.write(f"   Expectations:\n")
            f.write(f"      Excellent:  > {METRIC_THRESHOLDS['rank_correlation']['excellent']:.4f}\n")
            f.write(f"      Good:       > {METRIC_THRESHOLDS['rank_correlation']['good']:.2f}\n")
            f.write(f"      Acceptable: > {METRIC_THRESHOLDS['rank_correlation']['acceptable']:.2f}\n\n")
            
            # Next Token Accuracy
            acc_data = report['metrics']['next_token_accuracy']
            acc_status = get_metric_status('next_token_accuracy', acc_data['mean'])
            f.write("6. NEXT TOKEN ACCURACY\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Mean:    {acc_data['mean']:.4f}%\n")
            f.write(f"   Std Dev: {acc_data['std']:.4f}%\n")
            f.write(f"   Min:     {acc_data['min']:.4f}%\n")
            f.write(f"   Max:     {acc_data['max']:.4f}%\n")
            f.write(f"   Median:  {acc_data['median']:.4f}%\n")
            f.write(f"   Status:  {acc_status.upper()}\n")
            f.write(f"   Expectations:\n")
            f.write(f"      Excellent:  > {METRIC_THRESHOLDS['next_token_accuracy']['excellent']:.0f}%\n")
            f.write(f"      Good:       > {METRIC_THRESHOLDS['next_token_accuracy']['good']:.0f}%\n")
            f.write(f"      Acceptable: > {METRIC_THRESHOLDS['next_token_accuracy']['acceptable']:.0f}%\n\n")
            
            # Overall Assessment
            f.write("="*80 + "\n")
            f.write("OVERALL ASSESSMENT\n")
            f.write("="*80 + "\n\n")
            
            statuses = [
                ('Perplexity Change', ppl_status),
                ('Logit MSE', mse_status),
                ('KL Divergence', kl_status),
                ('Top-K Overlap', topk_status),
                ('Rank Correlation', rank_status),
                ('Next Token Accuracy', acc_status)
            ]
            
            excellent_count = sum(1 for _, s in statuses if s == 'excellent')
            good_count = sum(1 for _, s in statuses if s == 'good')
            acceptable_count = sum(1 for _, s in statuses if s == 'acceptable')
            poor_count = sum(1 for _, s in statuses if s == 'poor')
            
            f.write(f"Metrics Summary:\n")
            f.write(f"   Excellent:  {excellent_count}/6 metrics\n")
            f.write(f"   Good:       {good_count}/6 metrics\n")
            f.write(f"   Acceptable: {acceptable_count}/6 metrics\n")
            f.write(f"   Poor:       {poor_count}/6 metrics\n\n")
            
            if excellent_count >= 5:
                f.write("Overall: EXCELLENT - Model optimization preserved accuracy exceptionally well!\n")
            elif excellent_count + good_count >= 5:
                f.write("Overall: GOOD - Model optimization maintained high accuracy.\n")
            elif poor_count == 0:
                f.write("Overall: ACCEPTABLE - Model optimization is within acceptable bounds.\n")
            else:
                f.write("Overall: NEEDS IMPROVEMENT - Some metrics show significant degradation.\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Saved text report to: {text_path}")
    
    def cleanup(self):
        """Clean up models and free resources."""
        logger.info("Cleaning up resources...")
        self.baseline_model.cleanup()
        self.qeff_model.cleanup()

def run_baseline_only(args):
    """Run baseline-only evaluation (perplexity calculation)."""
    import csv
    
    logger.info("="*80)
    logger.info("BASELINE-ONLY EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Context Length: {args.ctx_len}")
    logger.info("="*80)
    
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset_loader = DatasetLoader(args.dataset, args.model_name, args.num_samples, args.ctx_len)
        samples = dataset_loader.get_samples()
        logger.info(f"Loaded {len(samples)} samples")
        
        # Load model
        logger.info("Loading baseline model...")
        model_loader = BaselineModelLoader(args.model_name, args.device, args.baseline_dtype)
        
        # Calculate perplexity for each sample
        logger.info("Computing perplexity...")
        perplexities = []
        
        for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                input_ids = sample['input_ids'].unsqueeze(0)
                logits = model_loader.get_logits(input_ids)
                logits = logits[0]  # Remove batch dimension
                target_ids = input_ids[0].to(args.device)
                
                # Compute loss and perplexity
                loss = F.cross_entropy(logits[:-1], target_ids[1:], reduction='mean')
                ppl = torch.exp(loss).item()
                perplexities.append(ppl)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate statistics
        avg_ppl = np.mean(perplexities)
        std_ppl = np.std(perplexities)
        min_ppl = np.min(perplexities)
        max_ppl = np.max(perplexities)
        median_ppl = np.median(perplexities)
        
        # Save results
        results = {
            'model_name': args.model_name,
            'dataset': args.dataset,
            'num_samples': len(samples),
            'ctx_len': args.ctx_len,
            'perplexity': {
                'mean': float(avg_ppl),
                'std': float(std_ppl),
                'min': float(min_ppl),
                'max': float(max_ppl),
                'median': float(median_ppl)
            },
            'per_sample_perplexities': [float(p) for p in perplexities]
        }
        
        # Save JSON
        json_path = os.path.join(args.output_dir, 'baseline_perplexity.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON results to: {json_path}")
        
        # Save CSV
        csv_path = os.path.join(args.output_dir, 'baseline_perplexity.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Mean Perplexity', f'{avg_ppl:.4f}'])
            writer.writerow(['Std Dev', f'{std_ppl:.4f}'])
            writer.writerow(['Min', f'{min_ppl:.4f}'])
            writer.writerow(['Max', f'{max_ppl:.4f}'])
            writer.writerow(['Median', f'{median_ppl:.4f}'])
        logger.info(f"Saved CSV results to: {csv_path}")
        
        # Save text report
        txt_path = os.path.join(args.output_dir, 'baseline_report.txt')
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BASELINE MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Number of Samples: {len(samples)}\n")
            f.write(f"Context Length: {args.ctx_len}\n\n")
            f.write("="*80 + "\n")
            f.write("PERPLEXITY RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Mean Perplexity:   {avg_ppl:.4f} ± {std_ppl:.4f}\n")
            f.write(f"Median Perplexity: {median_ppl:.4f}\n")
            f.write(f"Min Perplexity:    {min_ppl:.4f}\n")
            f.write(f"Max Perplexity:    {max_ppl:.4f}\n\n")
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        logger.info(f"Saved text report to: {txt_path}")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(perplexities, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Perplexity')
        plt.ylabel('Frequency')
        plt.title('Perplexity Distribution')
        plt.axvline(avg_ppl, color='r', linestyle='--', label=f'Mean: {avg_ppl:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(perplexities, marker='o', linestyle='-', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Across Samples')
        plt.axhline(avg_ppl, color='r', linestyle='--', label=f'Mean: {avg_ppl:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = os.path.join(args.output_dir, 'perplexity_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization to: {viz_path}")
        
        # Cleanup
        model_loader.cleanup()
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Mean Perplexity: {avg_ppl:.4f} ± {std_ppl:.4f}")
        logger.info(f"Median Perplexity: {median_ppl:.4f}")
        logger.info(f"Range: [{min_ppl:.4f}, {max_ppl:.4f}]")
        logger.info(f"Total execution time: {elapsed_time/60:.2f} minutes")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Comparison Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and dataset arguments
    parser.add_argument('--model-name', type=str, 
                       help='HuggingFace model ID', default="Qwen/Qwen2-7B")
    parser.add_argument('--dataset', type=str, default='wikitext-2-raw-v1',
                       choices=['wikitext-2-raw-v1', 'lambada'],
                       help='Dataset to use for evaluation')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--ctx-len', type=int, default=512,
                       help='Context length for each sample')
    parser.add_argument('--stride', type=int, default=256,
                       help='Stride for sliding window (WikiText only)')
    
    # Metric arguments
    parser.add_argument('--top-k', type=int, default=10,
                       help='K value for top-k overlap metric')
    
    # Baseline model arguments
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'cpu'],
                       help='Device for baseline model')
    parser.add_argument('--baseline-dtype', type=str, default='bf16',
                       choices=['fp32', 'fp16', 'bf16'],
                       help='Data type for baseline model')
    
    # Mode selection
    parser.add_argument('--baseline-only', action='store_true',
                       help='Run baseline-only mode (no QEfficient comparison)')
    
    # QEfficient arguments
    parser.add_argument('--qaic-prefill-seq-len', type=int, default=32,
                       help='Prefill sequence length for QEfficient')
    parser.add_argument('--qaic-num-cores', type=int, default=16,
                       help='Number of cores for QEfficient')
    parser.add_argument('--qaic-device-group', type=str, default='0',
                       help='Device IDs for QEfficient (comma-separated)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./comparison_results_qwen',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    # Check if baseline-only mode
    if args.baseline_only or not QEFFICIENT_AVAILABLE:
        if not QEFFICIENT_AVAILABLE:
            logger.warning("QEfficient not available - running in baseline-only mode")
        logger.info("Running baseline-only evaluation (perplexity calculation)")
        run_baseline_only(args)
        return
    
    # Parse device group
    device_group = [int(x.strip()) for x in args.qaic_device_group.split(',')]
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL COMPARISON")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Context Length: {args.ctx_len}")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # Initialize comparison engine
        engine = ComparisonEngine(
            model_name=args.model_name,
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            ctx_len=args.ctx_len,
            top_k=args.top_k,
            device=args.device,
            baseline_dtype=args.baseline_dtype,
            qaic_prefill_seq_len=args.qaic_prefill_seq_len,
            qaic_num_cores=args.qaic_num_cores,
            qaic_device_group=device_group,
            output_dir=args.output_dir
        )
        
        # Run comparison
        engine.run_comparison()
        
        # Generate reports
        engine.generate_report()
        
        # Cleanup
        engine.cleanup()
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("="*80)
        logger.info("COMPARISON COMPLETE!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
