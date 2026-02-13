# Accuracy Analysis Tools

This directory contains tools for comprehensive accuracy analysis and comparison between HuggingFace baseline models and QEfficient-optimized models running on Qualcomm Cloud AI 100.

## Overview

The accuracy analysis toolkit provides detailed metrics to evaluate how well QEfficient models preserve the accuracy of the original HuggingFace models. This is crucial for understanding the trade-offs between performance optimization and model fidelity.

## Contents

- `model_comparison_metrics.py` - Main comparison script with 6 comprehensive metrics
- `example_usage.sh` - Example usage scripts
- `README.md` - This file

---

## 📊 Understanding the Metrics

### 1. PERPLEXITY → "How confused is the model?"

**Simple:** Lower = better. Measures how well the model predicts text.

**Technical:** Exponential of average negative log-likelihood. Quantifies prediction uncertainty.

**Example:**
```
Text: "The cat sat on the ___"

Good model (perplexity ~5):
  "mat": 60%, "floor": 25%, "chair": 10%, "table": 5%
  → Confident, focused predictions

Bad model (perplexity ~50):
  "mat": 15%, "elephant": 14%, "quantum": 13%, "purple": 12%...
  → Confused, scattered predictions
```

**Interpretation:**
- **Excellent:** < 20 for general language models
- **Acceptable:** 20-50
- **Poor:** > 50

---

### 2. LOGIT MSE → "How different are the raw scores?"

**Simple:** Measures numerical difference in model outputs before converting to probabilities.

**Technical:** Mean squared error of logit vectors. Captures pre-softmax divergence.

**Example:**
```
For 4 possible next words:

Baseline logits:    [5.2,  3.1,  1.8, -0.5]
QEfficient logits:  [5.20002, 3.10001, 1.80001, -0.49999]

Differences:        [0.00002, 0.00001, 0.00001, 0.00001]
MSE = average of squared differences = 0.00000000025
```

**Interpretation:**
- **Excellent:** < 1.0
- **Good:** 1.0-5.0
- **Acceptable:** 5.0-15.0
- **Poor:** > 15.0

---

### 3. KL DIVERGENCE → "How different are the probability distributions?"

**Simple:** Measures information loss between two probability distributions.

**Technical:** Kullback-Leibler divergence: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))

**Example:**
```
Predicting next word after "I love":

Baseline:    {"you": 35%, "it": 30%, "this": 20%, "that": 15%}
QEfficient:  {"you": 35%, "it": 30%, "this": 20%, "that": 15%}
KL Divergence = 0 (identical)

If QEfficient was: {"you": 40%, "it": 25%, "this": 20%, "that": 15%}
KL Divergence = 0.015 (noticeable difference)
```

**Why negative values?** Numerical precision in floating-point calculations. Values near ±0.00005 are effectively zero.

**Interpretation:**
- **Excellent:** < 0.1
- **Good:** 0.1-1.0
- **Acceptable:** 1.0-5.0
- **Poor:** > 5.0

---

### 4. TOP-K OVERLAP → "Do models agree on most likely tokens?"

**Simple:** % of top candidates both models agree on.

**Technical:** Intersection over union of top-K token sets (typically K=10-100).

**Example:**
```
After "The quick brown", predict next word:

Baseline top-10:
  1. fox (25%)    6. deer (3%)
  2. dog (20%)    7. wolf (2%)
  3. cat (15%)    8. bear (2%)
  4. horse (10%)  9. lion (1%)
  5. rabbit (5%)  10. tiger (1%)

QEfficient top-10:
  1. fox (25%)    6. deer (3%)
  2. dog (20%)    7. wolf (2%)
  3. cat (15%)    8. bear (2%)
  4. horse (10%)  9. lion (1%)
  5. rabbit (5%)  10. mouse (1%)  ← Only difference

Overlap: 9/10 = 90%
```

**Interpretation:**
- **Excellent:** > 90%
- **Good:** 70-90%
- **Acceptable:** 50-70%
- **Poor:** < 50%

---

### 5. RANK CORRELATION → "Is the entire ordering preserved?"

**Simple:** Do models rank ALL tokens similarly? (Not just top ones)

**Technical:** Spearman's rank correlation coefficient. Measures monotonic relationship between rankings.

**Example:**
```
Vocabulary: 10,000 tokens

Baseline ranking:
  "the"(1st), "a"(2nd), "is"(3rd)... "xylophone"(9,847th)... "zephyr"(10,000th)

QEfficient ranking:
  "the"(1st), "a"(2nd), "is"(3rd)... "xylophone"(9,847th)... "zephyr"(10,000th)

If rankings match exactly → correlation = 1.0
If completely random → correlation = 0.0
If opposite order → correlation = -1.0
```

**Interpretation:**
- **Excellent:** > 0.95
- **Good:** 0.85-0.95
- **Acceptable:** 0.70-0.85
- **Poor:** < 0.70

---

### 6. NEXT TOKEN ACCURACY → "How often do models pick the same winner?"

**Simple:** % of times both models choose the same most-likely token.

**Technical:** Argmax agreement rate. Binary match on highest-probability token.

**Example:**
```
Sequence of 256 tokens to predict:

Position 1: "The" → Both pick "cat" ✓
Position 2: "cat" → Both pick "sat" ✓
Position 3: "sat" → Both pick "on" ✓
...
Position 255: "the" → Both pick "mat" ✓
Position 256: "mat" → Baseline picks ".", QEfficient picks "!" ✗

Accuracy: 255/256 = 99.61%
```

**Interpretation:**
- **Excellent:** > 95%
- **Good:** 85-95%
- **Acceptable:** 70-85%
- **Poor:** < 70%

---

## 📈 Quick Reference Table

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **Perplexity Δ** | <1% | 1-5% | 5-10% | >10% |
| **Logit MSE** | <1.0 | 1.0-5.0 | 5.0-15.0 | >15.0 |
| **KL Divergence** | <0.1 | 0.1-1.0 | 1.0-5.0 | >5.0 |
| **Top-K Overlap** | >90% | 70-90% | 50-70% | <50% |
| **Rank Correlation** | >0.95 | 0.85-0.95 | 0.70-0.85 | <0.70 |
| **Next Token Acc** | >95% | 85-95% | 70-85% | <70% |

---

## Features

### Supported Metrics

The toolkit computes **6 comprehensive metrics** for model comparison:

1. **Perplexity** - Language modeling quality metric
   - Measures how well the model predicts the next token
   - Lower is better
   - Computed for both baseline and QEfficient models

2. **Logit MSE** (Mean Squared Error)
   - Measures the raw difference between logit values
   - Indicates how closely QEfficient matches baseline outputs
   - Lower values indicate better agreement

3. **KL Divergence** (Kullback-Leibler Divergence)
   - Measures the difference between probability distributions
   - KL(P_baseline || P_qefficient)
   - Lower values indicate more similar distributions

4. **Top-k Overlap**
   - Percentage of overlap in top-k predicted tokens
   - Configurable k value (default: 10)
   - Higher values indicate better agreement on likely tokens

5. **Rank Correlation** (Spearman)
   - Measures correlation of token rankings
   - Values range from -1 to 1 (1 = perfect correlation)
   - Indicates how well relative token probabilities are preserved

6. **Next-Token Accuracy** (Argmax Agreement)
   - Percentage of exact matches in top-1 predictions
   - Most stringent metric
   - Higher values indicate better agreement

### Supported Datasets

- **WikiText-2** (`wikitext-2-raw-v1`) - General language modeling
- **LAMBADA** (`lambada`) - Long-range dependency evaluation

### Operating Modes

1. **Full Comparison Mode** - Compares baseline vs QEfficient with all 6 metrics
2. **Baseline-Only Mode** - Computes perplexity for baseline model only (useful when QEfficient is not available)

## Installation

### Prerequisites

```bash
# Install QEfficient (required for full comparison)
pip install QEfficient

# Install additional dependencies
pip install transformers datasets torch scipy matplotlib seaborn
```

### Verify Installation

```bash
# Check if QEfficient is available
python -c "from QEfficient import QEFFAutoModelForCausalLM; print('QEfficient available')"

# Check QAIC devices
python -c "from QEfficient.utils.device_utils import get_available_device_id; print(f'Devices: {get_available_device_id()}')"
```

## Usage

### Basic Usage - Full Comparison

```bash
python model_comparison_metrics.py \
  --model-name meta-llama/Llama-3.2-1B \
  --dataset wikitext-2-raw-v1 \
  --num-samples 100 \
  --ctx-len 512 \
  --top-k 10 \
  --device cuda \
  --baseline-dtype fp32 \
  --qaic-prefill-seq-len 32 \
  --qaic-num-cores 16 \
  --qaic-device-group 0 \
  --output-dir ./comparison_results
```

### Baseline-Only Mode

```bash
python model_comparison_metrics.py \
  --model-name meta-llama/Llama-3.2-1B \
  --dataset wikitext-2-raw-v1 \
  --num-samples 100 \
  --ctx-len 512 \
  --baseline-only \
  --device cuda \
  --output-dir ./baseline_results
```

### CPU Mode (No CUDA)

```bash
python model_comparison_metrics.py \
  --model-name meta-llama/Llama-3.2-1B \
  --dataset wikitext-2-raw-v1 \
  --num-samples 50 \
  --ctx-len 256 \
  --device cpu \
  --baseline-dtype fp32 \
  --qaic-prefill-seq-len 32 \
  --qaic-num-cores 16 \
  --qaic-device-group 0 \
  --output-dir ./cpu_comparison_results
```

### Quick Test (Small Sample)

```bash
python model_comparison_metrics.py \
  --model-name meta-llama/Llama-3.2-1B \
  --dataset wikitext-2-raw-v1 \
  --num-samples 5 \
  --ctx-len 128 \
  --device cpu \
  --output-dir ./quick_test
```

## Command-Line Arguments

### Required Arguments

- `--model-name` - HuggingFace model ID (e.g., `meta-llama/Llama-3.2-1B`)

### Dataset Arguments

- `--dataset` - Dataset name (default: `wikitext-2-raw-v1`)
  - Choices: `wikitext-2-raw-v1`, `lambada`
- `--num-samples` - Number of samples to evaluate (default: 100)
- `--ctx-len` - Context length for each sample (default: 512)
- `--stride` - Stride for sliding window, WikiText only (default: 256)

### Metric Arguments

- `--top-k` - K value for top-k overlap metric (default: 10)

### Baseline Model Arguments

- `--device` - Device for baseline model (default: `cuda`)
  - Choices: `cuda`, `cpu`
- `--baseline-dtype` - Data type for baseline model (default: `fp32`)
  - Choices: `fp32`, `fp16`, `bf16`

### Mode Selection

- `--baseline-only` - Run baseline-only mode (no QEfficient comparison)

### QEfficient Arguments

- `--qaic-prefill-seq-len` - Prefill sequence length (default: 32)
- `--qaic-num-cores` - Number of cores (default: 16)
- `--qaic-device-group` - Device IDs, comma-separated (default: `0`)

### Output Arguments

- `--output-dir` - Output directory for results (default: `./comparison_results`)

## Output Files

The script generates comprehensive reports in multiple formats:

### JSON Report
- **File**: `metrics_summary.json`
- **Content**: Machine-readable results with all metrics and statistics
- **Use Case**: Programmatic analysis, CI/CD integration

### CSV Report
- **File**: `comparison_report.csv`
- **Content**: Spreadsheet-friendly format with key metrics
- **Use Case**: Excel analysis, data processing

### Text Report
- **File**: `comprehensive_report.txt`
- **Content**: Human-readable summary with formatted statistics
- **Use Case**: Quick review, documentation

### Visualizations

Six PNG files with detailed plots:

1. `logit_mse_distribution.png` - Distribution of MSE values
2. `kl_divergence_plot.png` - KL divergence across samples
3. `topk_overlap_chart.png` - Top-k overlap percentages
4. `rank_correlation_scatter.png` - Rank correlation scatter plot
5. `next_token_accuracy.png` - Next-token accuracy trend
6. `comprehensive_summary.png` - 6-panel summary dashboard

### Baseline-Only Mode Output

When running in baseline-only mode:
- `baseline_perplexity.json` - Perplexity results in JSON
- `baseline_perplexity.csv` - Perplexity results in CSV
- `baseline_report.txt` - Text summary
- `perplexity_visualization.png` - Perplexity distribution and trend

## Example Workflows

### 1. Quick Model Validation

Test a model quickly with a small sample:

```bash
python model_comparison_metrics.py \
  --model-name Qwen/Qwen2-1.5B-Instruct \
  --num-samples 10 \
  --ctx-len 128 \
  --device cpu \
  --output-dir ./quick_validation
```

### 2. Comprehensive Accuracy Analysis

Full evaluation with 100 samples:

```bash
python model_comparison_metrics.py \
  --model-name meta-llama/Llama-3.2-1B \
  --dataset wikitext-2-raw-v1 \
  --num-samples 100 \
  --ctx-len 512 \
  --top-k 10 \
  --device cuda \
  --baseline-dtype fp32 \
  --qaic-prefill-seq-len 32 \
  --qaic-num-cores 16 \
  --qaic-device-group 0 \
  --output-dir ./full_analysis
```

### 3. Multi-Device Comparison

Compare across multiple QAIC devices:

```bash
# Device 0
python model_comparison_metrics.py \
  --model-name meta-llama/Llama-3.2-1B \
  --qaic-device-group 0 \
  --output-dir ./device_0_results

# Device 1
python model_comparison_metrics.py \
  --model-name meta-llama/Llama-3.2-1B \
  --qaic-device-group 1 \
  --output-dir ./device_1_results
```

### 4. Baseline Perplexity Benchmark

Establish baseline metrics without QEfficient:

```bash
python model_comparison_metrics.py \
  --model-name meta-llama/Llama-3.2-1B \
  --dataset wikitext-2-raw-v1 \
  --num-samples 200 \
  --ctx-len 512 \
  --baseline-only \
  --device cuda \
  --output-dir ./baseline_benchmark
```

## Performance Considerations

### Memory Usage

- **Baseline Model**: Depends on model size and dtype
  - FP32: ~4GB for 1B parameter model
  - FP16: ~2GB for 1B parameter model
- **QEfficient Model**: Compiled QPC size + runtime memory
- **Dataset**: Minimal (samples loaded on-demand)

### Execution Time

Approximate times for 100 samples on Llama-3.2-1B:

- **Baseline-only mode**: 5-10 minutes (GPU), 20-30 minutes (CPU)
- **Full comparison mode**: 15-25 minutes (includes compilation)
- **First run**: +5-10 minutes for model compilation

### Optimization Tips

1. **Use GPU for baseline** when available (`--device cuda`)
2. **Start with small samples** for testing (`--num-samples 5`)
3. **Use FP16** for faster baseline inference (`--baseline-dtype fp16`)
4. **Adjust context length** based on model capabilities (`--ctx-len`)
5. **Use stride** to control sample overlap (`--stride`)

## Troubleshooting

### Common Issues

#### 1. QEfficient Not Available

**Error**: `ModuleNotFoundError: No module named 'QEfficient'`

**Solution**:
```bash
pip install QEfficient
```

Or run in baseline-only mode:
```bash
python model_comparison_metrics.py --baseline-only ...
```

#### 2. CUDA Not Available

**Error**: `AssertionError: Torch not compiled with CUDA enabled`

**Solution**: Use CPU mode:
```bash
python model_comparison_metrics.py --device cpu ...
```

#### 3. Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `--num-samples`
- Reduce `--ctx-len`
- Use `--baseline-dtype fp16`
- Use `--device cpu`

#### 4. No QAIC Devices

**Error**: `No QAIC devices found`

**Solution**: Verify QAIC installation:
```bash
/opt/qti-aic/tools/qaic-util -q
```

#### 5. Model Download Issues

**Error**: `OSError: Can't load tokenizer`

**Solution**: Set HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```

## Advanced Usage

### Custom Dataset

To use a custom dataset, modify the `DatasetLoader` class in `model_comparison_metrics.py`:

```python
def _load_custom_dataset(self) -> List[Dict]:
    # Your custom loading logic
    pass
```

### Custom Metrics

Add new metrics by extending the `MetricCalculator` base class:

```python
class CustomMetricCalculator(MetricCalculator):
    def __init__(self):
        super().__init__("Custom Metric")
    
    def compute(self, baseline_logits, qeff_logits):
        # Your metric computation
        pass
```

### Batch Processing

Process multiple models:

```bash
#!/bin/bash
MODELS=("meta-llama/Llama-3.2-1B" "Qwen/Qwen2-1.5B-Instruct")

for model in "${MODELS[@]}"; do
    python model_comparison_metrics.py \
        --model-name "$model" \
        --num-samples 100 \
        --output-dir "./results_$(basename $model)"
done
```

## Integration with CI/CD

### Example GitHub Actions Workflow

```yaml
name: Accuracy Analysis

on: [push, pull_request]

jobs:
  accuracy-test:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v2
      
      - name: Run Accuracy Analysis
        run: |
          python efficient-transformers/scripts/accuracy_analysis/model_comparison_metrics.py \
            --model-name meta-llama/Llama-3.2-1B \
            --num-samples 50 \
            --baseline-only \
            --output-dir ./accuracy_results
      
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: accuracy-results
          path: ./accuracy_results/
```

## Contributing

To contribute improvements or new metrics:

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Submit a pull request

## Support

For issues or questions:
- Open an issue on GitHub
- Contact the QEfficient team
- Check the main QEfficient documentation

## License

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause

## References

- [QEfficient Documentation](https://github.com/quic/efficient-transformers)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Qualcomm Cloud AI 100](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence)

## Changelog

### Version 1.1.0 (2026-02-12)
- Added comprehensive metrics explanation section
- Added practical examples for each metric
- Added quick reference table for metric interpretation
- Improved documentation structure

### Version 1.0.0 (2026-02-09)
- Initial release
- 6 comprehensive metrics
- Support for WikiText-2 and LAMBADA datasets
- Baseline-only mode
- Full comparison mode with QEfficient
- Comprehensive visualization suite
- JSON, CSV, and text report generation
