#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Synthetic Model Comparison Script for Testing

This script uses synthetic data to test the model comparison metrics
without requiring actual models or datasets. It generates three scenarios:
- GOOD: Excellent quality (< 1% degradation)
- BAD: Acceptable quality (5-10% degradation)
- WORST: Poor quality (> 10% degradation)

Usage:
    python model_comparison_synthetic.py --scenario good --num-samples 100
    python model_comparison_synthetic.py --scenario bad --num-samples 500
    python model_comparison_synthetic.py --scenario worst --num-samples 1000
"""

import argparse
import json
import logging
import os
import time
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import metric calculators from the original script
from metric_calculators import (
    PerplexityCalculator,
    LogitMSECalculator,
    KLDivergenceCalculator,
    TopKOverlapCalculator,
    RankCorrelationCalculator,
    NextTokenAccuracyCalculator
)

# Import configuration constants
from comparison_config import FIGURE_SIZE, DPI, METRIC_THRESHOLDS

# Import synthetic data generator
from synthetic_data_generator import SyntheticDataGenerator, get_scenario_description

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


class SyntheticComparisonEngine:
    """Engine for running synthetic model comparison."""
    
    def __init__(
        self,
        scenario: str,
        num_samples: int,
        vocab_size: int = 32000,
        seq_len: int = 512,
        top_k: int = 10,
        output_dir: str = "./synthetic_results"
    ):
        """Initialize synthetic comparison engine."""
        self.scenario = scenario
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize synthetic data generator
        logger.info(f"Initializing synthetic data generator for scenario: {scenario}")
        logger.info(get_scenario_description(scenario))
        self.data_generator = SyntheticDataGenerator(vocab_size, seq_len, scenario)
        
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
        """Run comparison on synthetic data."""
        logger.info(f"Starting synthetic comparison with {self.num_samples} samples...")
        logger.info(f"Scenario: {self.scenario.upper()}")
        logger.info(f"Vocab size: {self.vocab_size}, Sequence length: {self.seq_len}")
        
        # Generate all samples
        samples = self.data_generator.generate_batch(self.num_samples)
        
        # Process each sample
        logger.info("Computing metrics for all samples...")
        for i, (baseline_logits, qeff_logits, target_ids) in enumerate(tqdm(samples, desc="Processing samples")):
            try:
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
            'scenario': self.scenario,
            'scenario_description': get_scenario_description(self.scenario),
            'num_samples': self.num_samples,
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
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
            writer.writerow(['Metric', 'Value', 'Std Dev', 'Min', 'Max', 'Status'])
            
            for metric_name, metric_data in report['metrics'].items():
                if metric_name == 'perplexity':
                    pct_change = abs(metric_data['difference']['relative_change_pct'])
                    status = get_metric_status('perplexity_pct_change', pct_change)
                    writer.writerow([
                        'Baseline Perplexity',
                        f"{metric_data['baseline']['perplexity']:.4f}",
                        f"{metric_data['baseline']['perplexity_std']:.4f}",
                        '-',
                        '-',
                        '-'
                    ])
                    writer.writerow([
                        'QEfficient Perplexity',
                        f"{metric_data['qefficient']['perplexity']:.4f}",
                        f"{metric_data['qefficient']['perplexity_std']:.4f}",
                        '-',
                        '-',
                        '-'
                    ])
                    writer.writerow([
                        'Perplexity % Change',
                        f"{pct_change:.2f}%",
                        '-',
                        '-',
                        '-',
                        status.upper()
                    ])
                else:
                    mean_val = metric_data.get('mean', 0)
                    status = get_metric_status(metric_name, mean_val)
                    writer.writerow([
                        metric_name,
                        f"{mean_val:.6f}",
                        f"{metric_data.get('std', 0):.6f}",
                        f"{metric_data.get('min', 0):.6f}",
                        f"{metric_data.get('max', 0):.6f}",
                        status.upper()
                    ])
        
        logger.info(f"Saved CSV report to: {csv_path}")
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        # Import the visualization methods from the original script
        # We'll create simplified versions here
        
        # Plot 0: Per-Sample Perplexity Comparison
        ppl_data = self.metrics['perplexity'].get_summary()
        if ppl_data and 'baseline' in ppl_data and 'per_sample_perplexities' in ppl_data['baseline']:
            baseline_ppls = ppl_data['baseline']['per_sample_perplexities']
            qeff_ppls = ppl_data['qefficient']['per_sample_perplexities']
            
            if baseline_ppls and qeff_ppls:
                plt.figure(figsize=FIGURE_SIZE)
                n_samples = len(baseline_ppls)
                x = range(n_samples)
                
                plt.plot(x, baseline_ppls, marker='o', linestyle='-', alpha=0.7, color='blue', label='Baseline', linewidth=2, markersize=3)
                plt.plot(x, qeff_ppls, marker='s', linestyle='-', alpha=0.7, color='orange', label='QEfficient', linewidth=2, markersize=3)
                
                baseline_mean = np.mean(baseline_ppls)
                qeff_mean = np.mean(qeff_ppls)
                plt.axhline(baseline_mean, color='blue', linestyle='--', linewidth=2, alpha=0.5, label=f'Baseline Mean: {baseline_mean:.2f}')
                plt.axhline(qeff_mean, color='orange', linestyle='--', linewidth=2, alpha=0.5, label=f'QEfficient Mean: {qeff_mean:.2f}')
                
                plt.xlabel('Sample Index')
                plt.ylabel('Perplexity')
                plt.title(f'Per-Sample Perplexity Comparison - {self.scenario.upper()} Scenario (n={n_samples})')
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.output_dir, 'perplexity_per_sample.png'), dpi=DPI, bbox_inches='tight')
                plt.close()
        
        # Plot 1: Logit MSE Distribution
        if self.metrics['logit_mse'].results:
            plt.figure(figsize=FIGURE_SIZE)
            summary = self.metrics['logit_mse'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('logit_mse', mean_val)
            
            plt.hist(self.metrics['logit_mse'].results, bins=30, edgecolor='black', alpha=0.7, color='purple')
            plt.xscale('log')
            plt.xlabel('Logit MSE (log scale)')
            plt.ylabel('Frequency')
            plt.title(f'Logit MSE Distribution - {self.scenario.upper()} Scenario\nMean: {mean_val:.2e} - Status: {status.upper()}')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'logit_mse_distribution.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 2: KL Divergence
        if self.metrics['kl_divergence'].results:
            plt.figure(figsize=FIGURE_SIZE)
            summary = self.metrics['kl_divergence'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('kl_divergence', mean_val)
            
            plt.plot(self.metrics['kl_divergence'].results, marker='o', linestyle='-', alpha=0.7, color='darkorange', markersize=3)
            plt.yscale('log')
            plt.xlabel('Sample Index')
            plt.ylabel('KL Divergence (log scale)')
            plt.title(f'KL Divergence - {self.scenario.upper()} Scenario\nMean: {mean_val:.2e} - Status: {status.upper()}')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'kl_divergence_plot.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Top-k Overlap
        if self.metrics['topk_overlap'].results:
            plt.figure(figsize=FIGURE_SIZE)
            summary = self.metrics['topk_overlap'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('topk_overlap', mean_val)
            
            plt.bar(range(len(self.metrics['topk_overlap'].results)), self.metrics['topk_overlap'].results, alpha=0.7, color='steelblue')
            plt.axhline(y=mean_val, color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}%')
            plt.xlabel('Sample Index')
            plt.ylabel('Overlap (%)')
            plt.title(f'Top-k Token Overlap - {self.scenario.upper()} Scenario\nMean: {mean_val:.1f}% - Status: {status.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(os.path.join(self.output_dir, 'topk_overlap_chart.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 4: Rank Correlation
        if self.metrics['rank_correlation'].results:
            plt.figure(figsize=FIGURE_SIZE)
            summary = self.metrics['rank_correlation'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('rank_correlation', mean_val)
            
            plt.scatter(range(len(self.metrics['rank_correlation'].results)), self.metrics['rank_correlation'].results, alpha=0.6, s=50, color='teal')
            plt.axhline(y=1.0, color='darkgreen', linestyle='--', linewidth=2, label='Perfect Correlation')
            plt.axhline(y=mean_val, color='darkblue', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.6f}')
            plt.xlabel('Sample Index')
            plt.ylabel('Spearman Correlation')
            plt.title(f'Rank Correlation - {self.scenario.upper()} Scenario\nMean: {mean_val:.6f} - Status: {status.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'rank_correlation_scatter.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 5: Next-Token Accuracy
        if self.metrics['next_token_accuracy'].results:
            plt.figure(figsize=FIGURE_SIZE)
            summary = self.metrics['next_token_accuracy'].get_summary()
            mean_val = summary['mean']
            status = get_metric_status('next_token_accuracy', mean_val)
            
            plt.plot(self.metrics['next_token_accuracy'].results, marker='s', linestyle='-', alpha=0.7, color='green', linewidth=2, markersize=3)
            plt.axhline(y=100.0, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Accuracy')
            plt.axhline(y=mean_val, color='darkblue', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.1f}%')
            plt.xlabel('Sample Index')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Next-Token Prediction Accuracy - {self.scenario.upper()} Scenario\nMean: {mean_val:.1f}% - Status: {status.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'next_token_accuracy.png'), dpi=DPI, bbox_inches='tight')
            plt.close()
        
        # Plot 6: Comprehensive Summary
        self._generate_summary_plot()
        
        logger.info("Generated all visualizations")
    
    def _generate_summary_plot(self):
        """Generate comprehensive summary plot."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Comprehensive Model Comparison Summary - {self.scenario.upper()} Scenario', fontsize=16, fontweight='bold')
        
        # Plot 1: Perplexity Comparison
        ppl_data = self.metrics['perplexity'].get_summary()
        if ppl_data:
            ax = axes[0, 0]
            models = ['Baseline', 'QEfficient']
            ppls = [ppl_data['baseline']['perplexity'], ppl_data['qefficient']['perplexity']]
            pct_change = abs(ppl_data['difference']['relative_change_pct'])
            status = get_metric_status('perplexity_pct_change', pct_change)
            
            bar_colors = ['#3498db', '#2ecc71' if status == 'excellent' else '#f39c12' if status == 'good' else '#e74c3c']
            ax.bar(models, ppls, color=bar_colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Perplexity')
            ax.set_title(f'Perplexity Comparison\n(Δ: {pct_change:.2f}% - {status.upper()})')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2-6: Other metrics (simplified versions)
        metric_names = ['logit_mse', 'kl_divergence', 'topk_overlap', 'rank_correlation', 'next_token_accuracy']
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for metric_name, (row, col) in zip(metric_names, positions):
            ax = axes[row, col]
            if self.metrics[metric_name].results:
                summary = self.metrics[metric_name].get_summary()
                mean_val = summary['mean']
                std_val = summary['std']
                status = get_metric_status(metric_name, mean_val)
                
                bar_color = '#2ecc71' if status == 'excellent' else '#f39c12' if status == 'good' else '#e74c3c'
                ax.bar(['Result'], [mean_val], color=bar_color, alpha=0.7, edgecolor='black')
                ax.errorbar(['Result'], [mean_val], yerr=[std_val], fmt='none', color='black', capsize=5)
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.set_title(f'{metric_name.replace("_", " ").title()}\n({status.upper()})')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_summary.png'), dpi=DPI, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, report: Dict):
        """Generate human-readable text report."""
        text_path = os.path.join(self.output_dir, 'comprehensive_report.txt')
        
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SYNTHETIC MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Scenario: {report['scenario'].upper()}\n")
            f.write(f"Description: {report['scenario_description']}\n")
            f.write(f"Number of Samples: {report['num_samples']}\n")
            f.write(f"Vocabulary Size: {report['vocab_size']}\n")
            f.write(f"Sequence Length: {report['seq_len']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("METRIC SUMMARIES\n")
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
            f.write(f"   Status:                 {ppl_status.upper()}\n\n")
            
            # Other metrics
            metric_names = ['logit_mse', 'kl_divergence', 'topk_overlap', 'rank_correlation', 'next_token_accuracy']
            metric_titles = ['LOGIT MSE', 'KL DIVERGENCE', 'TOP-K OVERLAP', 'RANK CORRELATION', 'NEXT TOKEN ACCURACY']
            
            for i, (metric_name, title) in enumerate(zip(metric_names, metric_titles), start=2):
                metric_data = report['metrics'][metric_name]
                status = get_metric_status(metric_name, metric_data['mean'])
                
                f.write(f"{i}. {title}\n")
                f.write("-" * 40 + "\n")
                f.write(f"   Mean:    {metric_data['mean']:.6f}\n")
                f.write(f"   Std Dev: {metric_data['std']:.6f}\n")
                f.write(f"   Min:     {metric_data['min']:.6f}\n")
                f.write(f"   Max:     {metric_data['max']:.6f}\n")
                f.write(f"   Median:  {metric_data['median']:.6f}\n")
                f.write(f"   Status:  {status.upper()}\n\n")
            
            # Overall Assessment
            f.write("="*80 + "\n")
            f.write("OVERALL ASSESSMENT\n")
            f.write("="*80 + "\n\n")
            
            statuses = [
                ('Perplexity Change', ppl_status),
                ('Logit MSE', get_metric_status('logit_mse', report['metrics']['logit_mse']['mean'])),
                ('KL Divergence', get_metric_status('kl_divergence', report['metrics']['kl_divergence']['mean'])),
                ('Top-K Overlap', get_metric_status('topk_overlap', report['metrics']['topk_overlap']['mean'])),
                ('Rank Correlation', get_metric_status('rank_correlation', report['metrics']['rank_correlation']['mean'])),
                ('Next Token Accuracy', get_metric_status('next_token_accuracy', report['metrics']['next_token_accuracy']['mean']))
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
            
            f.write(f"Expected Scenario: {report['scenario'].upper()}\n")
            f.write(f"Scenario Description: {report['scenario_description']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Saved text report to: {text_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthetic Model Comparison Script for Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--scenario', type=str, required=True,
                       choices=['good', 'bad', 'worst'],
                       help='Quality scenario to simulate')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--vocab-size', type=int, default=32000,
                       help='Vocabulary size')
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--top-k', type=int, default=10,
                       help='K value for top-k overlap metric')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: ./synthetic_results_{scenario}_{num_samples})')
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"./synthetic_results_{args.scenario}_{args.num_samples}"
    
    logger.info("="*80)
    logger.info("SYNTHETIC MODEL COMPARISON")
    logger.info("="*80)
    logger.info(f"Scenario: {args.scenario.upper()}")
    logger.info(f"Description: {get_scenario_description(args.scenario)}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Vocab Size: {args.vocab_size}")
    logger.info(f"Sequence Length: {args.seq_len}")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # Initialize comparison engine
        engine = SyntheticComparisonEngine(
            scenario=args.scenario,
            num_samples=args.num_samples,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            top_k=args.top_k,
            output_dir=args.output_dir
        )
        
        # Run comparison
        engine.run_comparison()
        
        # Generate reports
        engine.generate_report()
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("="*80)
        logger.info("COMPARISON COMPLETE!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
